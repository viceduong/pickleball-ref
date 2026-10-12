import Foundation
import simd

/// Snapshot published to the UI.
struct EngineSnapshot {
    var trail: [Vec2] = []
    var lastEvents: [BallEvent] = []
    var rallyActive = false
}

/// Time-ordered fusion of detector outputs → tracking → events → calls. Runs on its own serial queue.
final class RefereeEngine {
    let queue = DispatchQueue(label: "referee.engine", qos: .userInitiated)

    var onCall: ((RefCall, ContactMeasurement?) -> Void)?
    var onSnapshot: ((EngineSnapshot) -> Void)?

    private let tracker = BallTracker()
    private let detector = EventDetector()
    private let rules = RuleEngine()
    private var geometry: CourtGeometry?
    private var frameTimes: [Double] = []
    private var pending: [BallCandidate] = []
    private var feet: [RuleEngine.FootSample] = []
    private var emittedUpTo = -Double.infinity
    private var lastAnalysis = 0.0
    private var lastSnapshot = 0.0
    private var events: [BallEvent] = []
    private var currentRally = -1
    private(set) var fps = 120.0

    let reorderDelay = 0.15

    // MARK: Inputs (any thread)

    func setGeometry(_ g: CourtGeometry?) { queue.async { self.geometry = g; self.tracker.reset() } }
    func setKitchen(_ on: Bool) { queue.async { self.rules.kitchenEnabled = on } }
    func setFps(_ f: Double) { queue.async {
        self.fps = f
        self.detector.sidePoints = max(4, Int((f / 24).rounded()))
    } }

    func addCandidates(_ c: [BallCandidate]) {
        guard !c.isEmpty else { return }
        queue.async { self.pending.append(contentsOf: c) }
    }

    func addPose(t: Double, players: [PoseDetector.Player]) {
        queue.async {
            guard let g = self.geometry else { return }
            var ground: [Vec2] = [], wrists: [Vec2] = []
            for p in players {
                ground.append(contentsOf: p.anklesPx.map { g.footGround(anklePx: $0) })
                wrists.append(contentsOf: p.wristsPx)
            }
            self.feet.append(.init(t: t, ground: ground, wristsPx: wrists))
            self.feet.removeAll { $0.t < t - 4 }
        }
    }

    func frame(_ t: Double) {
        queue.async {
            self.frameTimes.append(t)
            self.drain(upTo: t - self.reorderDelay)
        }
    }

    // MARK: Processing

    private func drain(upTo limit: Double) {
        while let ft = frameTimes.first, ft <= limit {
            frameTimes.removeFirst()
            let tol = 0.5 / max(30, fps)
            var mine: [BallCandidate] = []
            pending.removeAll { c in
                if abs(c.t - ft) <= tol { mine.append(c); return true }
                return c.t < ft - tol   // stale
            }
            let started = tracker.ingest(frameTime: ft, candidates: mine)
            if tracker.rallyId != currentRally || started {
                currentRally = tracker.rallyId
                let first = tracker.rally.first.map { geometry?.toCourt($0.p) ?? .zero }
                rules.startRally(id: currentRally, firstBallCourt: first)
                emittedUpTo = max(emittedUpTo, tracker.rally.first?.t ?? ft)
                events.removeAll()
            }
            if ft - lastAnalysis > 0.05 { lastAnalysis = ft; analyse(now: ft) }
            if ft - lastSnapshot > 1.0 / 30 { lastSnapshot = ft; publish(now: ft) }
        }
    }

    private func analyse(now: Double) {
        let pts = tracker.rally.filter { $0.t >= now - 3.0 }
        guard pts.count >= 10, let lastT = pts.last?.t else { return }
        let found = detector.detect(pts, geometry: geometry)
        let stableBefore = tracker.hasActiveTrack ? lastT - 0.15 : lastT
        for e in found where e.t > emittedUpTo + 0.04 && e.t < stableBefore {
            emittedUpTo = e.t
            events.append(e)
            let contact = e.kind == .bounce ? measure(e) : nil
            for call in rules.handle(e, contact: contact, feet: feet) {
                onCall?(call, contact)
            }
        }
    }

    /// Ball-contact location and uncertainty from the arc-intersection image point.
    func measure(_ e: BallEvent) -> ContactMeasurement? {
        guard let g = geometry else { return nil }
        let ground = g.groundContact(ballCenterPx: e.imagePoint)
        let blurPx = 0.25 * g.ballRadiusPx(at: ground)
        let sigmaPx = (e.residualPx * e.residualPx + g.calibrationRmsPx * g.calibrationRmsPx + 0.6 * 0.6 + blurPx * blurPx).squareRoot()
        let skid = 0.4 / Court.feetToCm
        let sx = (pow(g.groundSigma(atPx: e.imagePoint, along: Vec2(1, 0), sigmaPx: sigmaPx), 2) + skid * skid).squareRoot()
        let sy = (pow(g.groundSigma(atPx: e.imagePoint, along: Vec2(0, 1), sigmaPx: sigmaPx), 2) + skid * skid).squareRoot()
        return ContactMeasurement(point: ground, sigma: Vec2(sx, sy), t: e.t, imagePoint: e.imagePoint)
    }

    private func publish(now: Double) {
        let trail = tracker.rally.filter { $0.t >= now - 0.6 }.map(\.p)
        let snap = EngineSnapshot(trail: trail, lastEvents: events.suffix(6), rallyActive: tracker.hasActiveTrack)
        onSnapshot?(snap)
    }
}
