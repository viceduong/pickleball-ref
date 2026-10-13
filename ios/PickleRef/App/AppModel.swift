import AVFoundation
import Combine
import Foundation
import SwiftUI
import UIKit

final class AppModel: ObservableObject {
    @Published var status = "Starting camera…"
    @Published var geometry: CourtGeometry?
    @Published var calls: [RefCall] = []
    @Published var snapshot = EngineSnapshot()
    @Published var fps = 0.0
    @Published var settings = CaptureSettings() { didSet { capture.apply(settings) } }
    @Published var announce = true
    @Published var kitchenCalls = true { didSet { engine.setKitchen(kitchenCalls) } }
    @Published var linkPhones = false { didSet { linkPhones ? peers.start() : peers.stop() } }
    @Published var cameraDenied = false
    @Published var cameraInfo = ""

    let engine: RefereeEngine
    let capture: CaptureEngine
    let peers = PeerLink()
    private let speech = AVSpeechSynthesizer()
    private var contacts: [UUID: (ContactMeasurement, Court.Region)] = [:]
    private var remoteBuffer: [(t: Double, m: ContactMeasurement, kind: String)] = []
    private var cancellables = Set<AnyCancellable>()

    var imageSize: Vec2 { geometry?.imageSize ?? Vec2(1920, 1080) }
    var neuralAvailable: Bool { capture.hasNeuralModel }

    init() {
        let e = RefereeEngine()
        engine = e
        capture = CaptureEngine(engine: e)
        capture.onCalibration = { [weak self] g, msg in
            self?.geometry = g
            self?.status = g.map { Self.describe($0) } ?? msg
        }
        capture.onStats = { [weak self] fps, _ in self?.fps = fps }
        capture.onFormat = { [weak self] info in self?.cameraInfo = info }
        engine.onCall = { [weak self] call, contact in
            DispatchQueue.main.async { self?.receive(call, contact: contact) }
        }
        engine.onSnapshot = { [weak self] s in DispatchQueue.main.async { self?.snapshot = s } }
        peers.onContact = { [weak self] t, m, kind, _ in
            DispatchQueue.main.async { self?.remote(t: t, m: m, kind: kind) }
        }
        peers.objectWillChange.sink { [weak self] _ in self?.objectWillChange.send() }.store(in: &cancellables)
        try? AVAudioSession.sharedInstance().setCategory(.playback, options: [.mixWithOthers, .duckOthers])
    }

    static func describe(_ g: CourtGeometry) -> String {
        var s = "Court locked · \(g.view.rawValue)"
        if let c = g.cameraCenter { s += String(format: " · cam %.1f ft high", c.z) }
        if let advice = g.setupAdvice { s += " · \(advice)" }
        return s
    }

    func start() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized: capture.start()
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { ok in
                DispatchQueue.main.async {
                    if ok { self.capture.start() } else { self.cameraDenied = true }
                }
            }
        default: cameraDenied = true
        }
        UIApplication.shared.isIdleTimerDisabled = true
    }

    func recalibrate() { capture.recalibrate() }

    var latestCall: RefCall? { calls.last }

    // MARK: Calls

    private func region(for call: RefCall) -> Court.Region {
        switch call.kind {
        case .serve:
            let p = call.courtPoint
            return Court.serviceCourt(half: Court.half(of: p), side: Court.side(of: p))
        default: return Court.fullCourt
        }
    }

    private func receive(_ call: RefCall, contact: ContactMeasurement?) {
        var c = call
        if let m = contact {
            contacts[c.id] = (m, region(for: c))
            if linkPhones { peers.broadcast(contact: m, kind: c.kind.rawValue) }
            // Fuse with any matching remote observation already received.
            let matches = remoteBuffer.filter { abs($0.t - m.t) < 0.06 }
            if !matches.isEmpty { c = fused(c, local: m, remote: matches.map(\.m)) }
        }
        calls.append(c)
        if calls.count > 200 { calls.removeFirst(50) }
        speak(c)
    }

    private func remote(t: Double, m: ContactMeasurement, kind: String) {
        remoteBuffer.append((t, m, kind))
        remoteBuffer.removeAll { $0.t < t - 5 }
        if let i = calls.lastIndex(where: { abs($0.t - t) < 0.06 && $0.kind != .kitchen }), let entry = contacts[calls[i].id] {
            let local = entry.0
            calls[i] = fused(calls[i], local: local, remote: [m])
            contacts[calls[i].id] = (LineJudge.fuse([local, m]) ?? local, region(for: calls[i]))
        }
    }

    private func fused(_ call: RefCall, local: ContactMeasurement, remote: [ContactMeasurement]) -> RefCall {
        guard let f = LineJudge.fuse([local] + remote) else { return call }
        let r = LineJudge.judge(f, region: region(for: call))
        var c = call
        c.courtX = f.point.x; c.courtY = f.point.y
        c.verdict = r.verdict; c.marginCm = r.marginCm; c.sigmaCm = r.sigmaCm; c.confidence = r.confidence
        c.fusedPeers += remote.count
        return c
    }

    private func speak(_ c: RefCall) {
        guard announce else { return }
        let text: String
        switch c.verdict {
        case .out: text = "Out"
        case .fault: text = c.kind == .kitchen ? "Kitchen fault" : "Fault"
        default: return
        }
        let u = AVSpeechUtterance(string: text)
        u.rate = 0.55
        speech.speak(u)
    }
}
