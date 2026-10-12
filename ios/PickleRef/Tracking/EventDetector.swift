import Foundation
import simd

struct BallEvent: Codable {
    enum Kind: String, Codable { case bounce, hit }
    var kind: Kind
    var t: Double
    var imageX: Double
    var imageY: Double
    /// Fit residual / arc intersection gap (px) — feeds the uncertainty model.
    var residualPx: Double
    /// Along-court travel direction after the event (+1 toward far baseline, -1 toward near, 0 unknown).
    var travelY: Double
    /// Ball-centre height at the event from the 3D arc (ft), if recovered.
    var height3D: Double?
    var imagePoint: Vec2 { Vec2(imageX, imageY) }
}

/// Detects trajectory discontinuities (bounces, paddle hits) by fitting independent ballistic arcs on
/// each side of every split and intersecting them (sub-frame contact estimate), then classifies them
/// with monocular physics-constrained 3D arcs.
final class EventDetector {
    var sidePoints = 5
    var minKink = 0.35
    var maxWindow = 0.22

    struct Kink {
        var j: Int
        var kink: Double
        var t: Double
        var p: Vec2
        var gap: Double
        var rms: Double
        var vL: Vec2
        var vR: Vec2
    }

    func detect(_ pts: [TrackPoint], geometry: CourtGeometry?) -> [BallEvent] {
        let kinks = detectKinks(pts)
        return kinks.enumerated().map { idx, k in
            let prevJ = idx > 0 ? kinks[idx - 1].j + 1 : 0
            let nextJ = idx + 1 < kinks.count ? kinks[idx + 1].j : pts.count - 1
            return classify(k, pts: pts, segBefore: prevJ...k.j, segAfter: (k.j + 1)...max(k.j + 1, nextJ), geometry: geometry)
        }
    }

    func detectKinks(_ pts: [TrackPoint]) -> [Kink] {
        let m = sidePoints
        guard pts.count >= 2 * m else { return [] }
        var cands: [Kink] = []
        for j in (m - 1)..<(pts.count - m) {
            let left = Array(pts[(j - m + 1)...j])
            let right = Array(pts[(j + 1)...(j + m)])
            if left.last!.t - left.first!.t > maxWindow || right.last!.t - right.first!.t > maxWindow { continue }
            if right.first!.t - left.last!.t > 0.08 { continue }
            let tSplit = 0.5 * (left.last!.t + right.first!.t)
            guard let fl = QuadFit.fit(left.map(\.t), left.map(\.p), t0: tSplit),
                  let fr = QuadFit.fit(right.map(\.t), right.map(\.p), t0: tSplit) else { continue }
            let ta = left.last!.t, tb = right.first!.t
            var bestT = tSplit, bestGap = Double.infinity
            for s in 0...40 {
                let tt = ta + (tb - ta) * Double(s) / 40
                let g = (fl.position(tt) - fr.position(tt)).length
                if g < bestGap { bestGap = g; bestT = tt }
            }
            let vL = fl.velocity(bestT), vR = fr.velocity(bestT)
            let kink = (vR - vL).length / max(60, max(vL.length, vR.length))
            let p = 0.5 * (fl.position(bestT) + fr.position(bestT))
            cands.append(Kink(j: j, kink: kink, t: bestT, p: p, gap: bestGap, rms: max(fl.rms, fr.rms), vL: vL, vR: vR))
        }
        var out: [Kink] = []
        for (i, c) in cands.enumerated() {
            guard c.kink >= minKink, c.gap <= 4 + 3 * c.rms else { continue }
            var isMax = true
            for k in max(0, i - m)...min(cands.count - 1, i + m) where k != i && abs(cands[k].j - c.j) < m {
                if cands[k].kink > c.kink || (cands[k].kink == c.kink && k < i) { isMax = false; break }
            }
            if isMax { out.append(c) }
        }
        return out
    }

    private func classify(_ k: Kink, pts: [TrackPoint], segBefore: ClosedRange<Int>, segAfter: ClosedRange<Int>,
                          geometry: CourtGeometry?) -> BallEvent {
        var kind: BallEvent.Kind
        var travel = 0.0
        var height: Double?
        let residual = (k.rms * k.rms + k.gap * k.gap / 4).squareRoot()

        // 1. Physics: 3D arcs before/after.
        if let g = geometry, let pose = g.pose,
           let before = Trajectory3D.fit(Array(pts[segBefore].suffix(40)), pose: pose),
           let after = Trajectory3D.fit(Array(pts[segAfter].prefix(40)), pose: pose),
           before.rmsPx < 6, after.rmsPx < 6 {
            let vb = before.velocity(k.t), va = after.velocity(k.t)
            let z = before.position(k.t).z
            height = z
            travel = va.y >= 0 ? 1 : -1
            let reversed = vb.y * va.y < 0 && abs(vb.y) > 3 && abs(va.y) > 3
            let verticalFlip = vb.z < 0 && va.z > 0
            kind = (!reversed && verticalFlip && z < 1.2) ? .bounce : .hit
            return BallEvent(kind: kind, t: k.t, imageX: k.p.x, imageY: k.p.y, residualPx: residual, travelY: travel, height3D: height)
        }

        // 2. Fallback: velocity change relative to the local image "up" direction.
        var up = Vec2(0, -1)
        if let g = geometry {
            let c = g.toCourt(k.p)
            if let a = g.project(Vec3(c.x, c.y, 0)), let b = g.project(Vec3(c.x, c.y, 1)) { up = (b - a).normalized }
            let c2 = g.toCourt(k.p + k.vR * 0.02)
            travel = c2.y - c.y >= 0 ? 1 : -1
        }
        let dv = k.vR - k.vL
        let upKick = simd_dot(dv, up)
        let lateral = abs(simd_dot(dv, Vec2(-up.y, up.x)))
        kind = upKick > 0.6 * lateral ? .bounce : .hit
        return BallEvent(kind: kind, t: k.t, imageX: k.p.x, imageY: k.p.y, residualPx: residual, travelY: travel, height3D: height)
    }
}
