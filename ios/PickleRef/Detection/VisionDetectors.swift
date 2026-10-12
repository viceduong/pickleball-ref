import AVFoundation
import CoreML
import CoreImage
import Foundation
import Vision
import simd

/// Apple Vision parabolic-trajectory detector (iOS 14+). Stateful: fed every frame in order.
final class VisionTrajectoryDetector {
    private var request: VNDetectTrajectoriesRequest
    private let handler = VNSequenceRequestHandler()
    private var emitted: [UUID: Double] = [:]

    init() { request = Self.makeRequest() }

    private static func makeRequest() -> VNDetectTrajectoriesRequest {
        let r = VNDetectTrajectoriesRequest(frameAnalysisSpacing: CMTime(value: 1, timescale: 120), trajectoryLength: 6)
        r.objectMinimumNormalizedRadius = 0.0008
        r.objectMaximumNormalizedRadius = 0.02
        return r
    }

    func reset() { request = Self.makeRequest(); emitted.removeAll() }

    /// - Parameters:
    ///   - roi: full-resolution pixel ROI (top-left origin).
    ///   - hostOffset: add to sample-buffer seconds to get host-clock seconds.
    func process(_ sb: CMSampleBuffer, roi: CGRect, imageSize: Vec2, hostOffset: Double) -> [BallCandidate] {
        // Vision ROI: normalised, lower-left origin.
        let nroi = CGRect(x: roi.minX / imageSize.x, y: 1 - roi.maxY / imageSize.y,
                          width: roi.width / imageSize.x, height: roi.height / imageSize.y)
            .intersection(CGRect(x: 0, y: 0, width: 1, height: 1))
        var used = CGRect(x: 0, y: 0, width: 1, height: 1)
        if !nroi.isNull && nroi.width > 0.05 && nroi.height > 0.05 { used = nroi }
        request.regionOfInterest = used
        do { try handler.perform([request], on: sb, orientation: .up) } catch { return [] }
        guard let obs = request.results else { return [] }
        var out: [BallCandidate] = []
        for o in obs {
            let pts = o.detectedPoints
            guard pts.count >= 2 else { continue }
            let start = o.timeRange.start.seconds, dur = o.timeRange.duration.seconds
            let last = emitted[o.uuid] ?? -.infinity
            for (i, p) in pts.enumerated() {
                let t = start + dur * Double(i) / Double(pts.count - 1) + hostOffset
                guard t > last + 1e-4 else { continue }
                // Vision results are normalised to the region of interest (lower-left origin).
                let nx = Double(used.minX) + p.x * Double(used.width)
                let ny = Double(used.minY) + p.y * Double(used.height)
                let px = Vec2(nx * imageSize.x, (1 - ny) * imageSize.y)
                out.append(BallCandidate(t: t, p: px, score: Double(o.confidence), source: .vision))
            }
            emitted[o.uuid] = start + dur + hostOffset
        }
        if emitted.count > 200 { emitted.removeAll() }
        return out
    }
}

/// Body pose (19 joints, iOS 14+) → ankles projected to the court and wrists for hitter attribution.
final class PoseDetector {
    private let handler = VNSequenceRequestHandler()

    struct Player { var anklesPx: [Vec2]; var wristsPx: [Vec2]; var box: CGRect }

    /// `buffer` is any downscaled copy of the frame; coordinates are mapped back to `imageSize`.
    func detect(_ buffer: CVPixelBuffer, imageSize: Vec2) -> [Player] {
        let req = VNDetectHumanBodyPoseRequest()
        do { try handler.perform([req], on: buffer, orientation: .up) } catch { return [] }
        guard let results = req.results else { return [] }
        var players: [Player] = []
        for o in results {
            guard let pts = try? o.recognizedPoints(.all) else { continue }
            func px(_ j: VNHumanBodyPoseObservation.JointName) -> Vec2? {
                guard let p = pts[j], p.confidence > 0.25 else { return nil }
                return Vec2(Double(p.location.x) * imageSize.x, (1 - Double(p.location.y)) * imageSize.y)
            }
            let la = px(.leftAnkle), ra = px(.rightAnkle)
            let lw = px(.leftWrist), rw = px(.rightWrist)
            guard let a0 = la ?? ra, let w0 = lw ?? rw else { continue }
            let ankles = [la ?? a0, ra ?? a0]
            let wrists = [lw ?? w0, rw ?? w0]
            let all = pts.values.filter { $0.confidence > 0.25 }
                .map { Vec2(Double($0.location.x) * imageSize.x, (1 - Double($0.location.y)) * imageSize.y) }
            let minX = all.map(\.x).min() ?? 0, maxX = all.map(\.x).max() ?? 0
            let minY = all.map(\.y).min() ?? 0, maxY = all.map(\.y).max() ?? 0
            let box = CGRect(x: minX, y: minY, width: maxX - minX, height: maxY - minY).insetBy(dx: -12, dy: -12)
            players.append(Player(anklesPx: ankles, wristsPx: wrists, box: box))
        }
        return players
    }
}

/// WASB (BMVC'23, HRNet heatmap, 3-frame input) converted to Core ML at build time.
/// Inputs: frame0…frame2 (BGRA 512×288, oldest first); output: `heatmaps` [1,3,288,512] (sigmoid).
final class NeuralBallDetector {
    static let inputSize = (w: 512, h: 288)
    private let model: MLModel
    var threshold: Float = 0.35

    static var isBundled: Bool { Bundle.main.url(forResource: "WASB", withExtension: "mlmodelc") != nil }

    init?() {
        guard let url = Bundle.main.url(forResource: "WASB", withExtension: "mlmodelc") else { return nil }
        let cfg = MLModelConfiguration()
        if #available(iOS 16.0, *) { cfg.computeUnits = .cpuAndNeuralEngine } else { cfg.computeUnits = .all }
        guard let m = try? MLModel(contentsOf: url, configuration: cfg) else { return nil }
        model = m
    }

    /// Returns the ball position (512×288 crop coordinates) in the middle frame — the channel with
    /// temporal context on both sides, most reliable in tests — with its score.
    func predict(frames: [CVPixelBuffer]) -> (Vec2, Double)? {
        guard frames.count == 3 else { return nil }
        let input = try? MLDictionaryFeatureProvider(dictionary: [
            "frame0": MLFeatureValue(pixelBuffer: frames[0]),
            "frame1": MLFeatureValue(pixelBuffer: frames[1]),
            "frame2": MLFeatureValue(pixelBuffer: frames[2]),
        ])
        guard let inp = input, let out = try? model.prediction(from: inp),
              let hm = out.featureValue(for: "heatmaps")?.multiArrayValue else { return nil }
        let (w, h) = Self.inputSize
        let plane = w * h
        guard hm.count >= 3 * plane else { return nil }
        let base = plane
        var best: Float = 0
        var bi = -1
        func value(_ i: Int) -> Float {
            switch hm.dataType {
            case .float32: return hm.dataPointer.assumingMemoryBound(to: Float.self)[i]
            case .double: return Float(hm.dataPointer.assumingMemoryBound(to: Double.self)[i])
            default: return hm[i].floatValue
            }
        }
        for i in 0..<plane {
            let v = value(base + i)
            if v > best { best = v; bi = i }
        }
        guard bi >= 0, best >= threshold else { return nil }
        // Weighted centroid around the peak.
        let px = bi % w, py = bi / w
        var sx = 0.0, sy = 0.0, sw = 0.0
        for y in max(0, py - 4)...min(h - 1, py + 4) {
            for x in max(0, px - 4)...min(w - 1, px + 4) {
                let v = Double(value(base + y * w + x))
                if v >= Double(threshold) * 0.5 { sx += Double(x) * v; sy += Double(y) * v; sw += v }
            }
        }
        guard sw > 0 else { return nil }
        return (Vec2(sx / sw, sy / sw), Double(best))
    }
}
