import Foundation
import simd

enum DetectionSource: String, Codable {
    case motion, vision, neural
    var weight: Double {
        switch self {
        case .neural: return 1.0
        case .vision: return 0.9
        case .motion: return 0.6
        }
    }
}

/// Ball hypothesis in full-resolution image pixels.
struct BallCandidate {
    var t: Double
    var p: Vec2
    var score: Double
    var source: DetectionSource
}

struct TrackPoint: Codable {
    var t: Double
    var x: Double
    var y: Double
    var source: DetectionSource
    var p: Vec2 { Vec2(x, y) }
    init(t: Double, p: Vec2, source: DetectionSource) { self.t = t; x = p.x; y = p.y; self.source = source }
}

/// Single-ball tracker: fuses candidates from all detectors, confirms tentative tracks by
/// ballistic consistency, follows the active track with a quadratic predictor.
final class BallTracker {
    private struct Track {
        var points: [TrackPoint]
        var lastT: Double { points.last!.t }
    }

    var confirmLength = 5
    var maxConfirmRms = 3.0
    var minSpeedPx = 120.0       // px/s, rejects static blobs
    var lostAfter = 0.12         // s without detection ends the active track
    var rallyGap = 1.5           // s without any track ends the rally

    private var active: Track?
    private var tentatives: [Track] = []
    private(set) var rally: [TrackPoint] = []
    private(set) var rallyId = 0
    private var lastAccepted: Double = -.infinity

    /// Returns true when this frame started a new rally (previous one finished).
    @discardableResult
    func ingest(frameTime t: Double, candidates raw: [BallCandidate]) -> Bool {
        var started = false
        let cands = merge(raw)
        if t - lastAccepted > rallyGap && !rally.isEmpty && active == nil {
            rally.removeAll()
            rallyId += 1
        }

        if var tr = active {
            let pred = predict(tr, at: t)
            let dt = t - tr.lastT
            let speed = velocity(tr).length
            let gate = 14 + 0.35 * speed * dt
            var best: (Int, Double)?
            for (i, c) in cands.enumerated() {
                let d = (c.p - pred).length
                if d > gate { continue }
                let cost = d / gate - 0.3 * c.score * c.source.weight
                if best == nil || cost < best!.1 { best = (i, cost) }
            }
            if let b = best {
                let c = cands[b.0]
                tr.points.append(TrackPoint(t: t, p: c.p, source: c.source))
                active = tr
                accept(tr.points.last!)
                return started
            }
            if dt > lostAfter { active = nil } else { active = tr; return started }
        }

        // No active track: grow tentatives.
        var used = Set<Int>()
        for k in tentatives.indices {
            let tr = tentatives[k]
            let pred = tr.points.count >= 3 ? predict(tr, at: t) : linearPredict(tr, at: t)
            let gate = tr.points.count == 1 ? 90.0 : 20.0
            var best: (Int, Double)?
            for (i, c) in cands.enumerated() where !used.contains(i) {
                let d = (c.p - pred).length
                if d < gate && (best == nil || d < best!.1) { best = (i, d) }
            }
            if let b = best {
                used.insert(b.0)
                tentatives[k].points.append(TrackPoint(t: t, p: cands[b.0].p, source: cands[b.0].source))
            }
        }
        tentatives.removeAll { t - $0.lastT > 0.06 }
        for (i, c) in cands.enumerated() where !used.contains(i) && tentatives.count < 12 {
            tentatives.append(Track(points: [TrackPoint(t: t, p: c.p, source: c.source)]))
        }
        if let idx = tentatives.indices.filter({ isConfirmed(tentatives[$0]) })
            .max(by: { tentatives[$0].points.count < tentatives[$1].points.count }) {
            let tr = tentatives[idx]
            if rally.isEmpty { started = true }
            active = tr
            tentatives.removeAll()
            for p in tr.points { accept(p) }
        }
        return started
    }

    func reset() {
        active = nil; tentatives.removeAll(); rally.removeAll(); rallyId += 1; lastAccepted = -.infinity
    }

    var hasActiveTrack: Bool { active != nil }

    // MARK: - Helpers

    private func accept(_ p: TrackPoint) {
        if let last = rally.last, p.t <= last.t { return }
        rally.append(p)
        lastAccepted = p.t
        if rally.count > 4000 { rally.removeFirst(1000) }
    }

    private func merge(_ cands: [BallCandidate]) -> [BallCandidate] {
        var out: [BallCandidate] = []
        for c in cands.sorted(by: { $0.score * $0.source.weight > $1.score * $1.source.weight }) {
            if let i = out.firstIndex(where: { ($0.p - c.p).length < 6 }) {
                let a = out[i]
                let wa = a.score * a.source.weight, wc = c.score * c.source.weight
                out[i].p = (a.p * wa + c.p * wc) / max(1e-6, wa + wc)
                out[i].score = min(1, a.score + 0.25 * c.score)
            } else {
                out.append(c)
            }
        }
        return out
    }

    private func isConfirmed(_ tr: Track) -> Bool {
        guard tr.points.count >= confirmLength else { return false }
        let pts = tr.points.suffix(8)
        guard let fit = QuadFit.fit(pts.map(\.t), pts.map(\.p), t0: pts.last!.t) else { return false }
        return fit.rms <= maxConfirmRms && fit.velocity(pts.last!.t).length >= minSpeedPx
    }

    private func velocity(_ tr: Track) -> Vec2 {
        let pts = tr.points.suffix(6)
        guard pts.count >= 2 else { return .zero }
        if pts.count >= 3, let f = QuadFit.fit(pts.map(\.t), pts.map(\.p), t0: pts.last!.t) { return f.velocity(pts.last!.t) }
        let a = pts.first!, b = pts.last!
        return (b.p - a.p) / max(1e-3, b.t - a.t)
    }

    private func predict(_ tr: Track, at t: Double) -> Vec2 {
        let pts = tr.points.suffix(6)
        if pts.count >= 4, let f = QuadFit.fit(pts.map(\.t), pts.map(\.p), t0: pts.last!.t) { return f.position(t) }
        return linearPredict(tr, at: t)
    }

    private func linearPredict(_ tr: Track, at t: Double) -> Vec2 {
        guard tr.points.count >= 2 else { return tr.points.last!.p }
        let a = tr.points[tr.points.count - 2], b = tr.points.last!
        let v = (b.p - a.p) / max(1e-3, b.t - a.t)
        return b.p + v * (t - b.t)
    }
}
