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
    /// Half-window of the hinge regression (s).
    var window = 0.1
    /// Significance threshold for the velocity jump (synthetic noise: false ≤ 49, true ≥ 91).
    var minChi2 = 60.0
    var noiseFloorPx = 0.5

    struct Kink {
        var j: Int
        var kink: Double      // χ² of the velocity jump
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

    /// Hinge regression around every split: p(t) = a + b·d + c·d² + Δv·max(0, d), d = t − τ, fitted jointly over
    /// ±`window` seconds with the contact time τ scanned between frames. Continuity is built in, the whole window
    /// constrains the contact point, and Δv is tested for significance (χ² against the residual noise level).
    func detectKinks(_ pts: [TrackPoint]) -> [Kink] {
        guard pts.count >= 10 else { return [] }
        let span = pts.last!.t - pts.first!.t
        let fps = Double(pts.count - 1) / max(1e-3, span)
        let m = max(sidePoints, Int((window * fps).rounded()))
        guard pts.count >= 2 * m else { return [] }
        var cands: [Kink] = []
        for j in (m - 1)..<(pts.count - m) {
            let seg = pts[(j - m + 1)...(j + m)]
            if seg.last!.t - seg.first!.t > 2.6 * window + 0.05 { continue }   // gap in the track
            var best: (rss: Double, tau: Double, coef: [Vec2], c00: Double, c33: Double)?
            for s in 0...8 {
                let tau = pts[j].t + (pts[j + 1].t - pts[j].t) * Double(s) / 8
                guard let f = hingeFit(seg, tau: tau) else { continue }
                if best == nil || f.rss < best!.rss { best = (f.rss, tau, f.coef, f.c00, f.c33) }
            }
            guard let b = best else { continue }
            let dof = Double(2 * seg.count - 8)
            let sigma = max(noiseFloorPx, (b.rss / max(1, dof)).squareRoot())
            let dv = b.coef[3]
            let chi2 = simd_length_squared(dv) / (sigma * sigma * b.c33)
            let vL = b.coef[1], vR = b.coef[1] + dv
            cands.append(Kink(j: j, kink: chi2, t: b.tau, p: b.coef[0], gap: sigma * b.c00.squareRoot(), rms: sigma, vL: vL, vR: vR))
        }
        var out: [Kink] = []
        let sep = max(3, m / 2)
        for (i, c) in cands.enumerated() where c.kink >= minChi2 {
            var isMax = true
            for k in max(0, i - sep)...min(cands.count - 1, i + sep) where k != i {
                if cands[k].kink > c.kink || (cands[k].kink == c.kink && k < i) { isMax = false; break }
            }
            if isMax { out.append(c) }
        }
        return out
    }

    private func hingeFit(_ seg: ArraySlice<TrackPoint>, tau: Double) -> (rss: Double, coef: [Vec2], c00: Double, c33: Double)? {
        var xtx = [Double](repeating: 0, count: 16)
        var xtpx = [Double](repeating: 0, count: 4), xtpy = [Double](repeating: 0, count: 4)
        for p in seg {
            let d = p.t - tau
            let row = [1, d, d * d, max(0, d)]
            for a in 0..<4 {
                xtpx[a] += row[a] * p.x; xtpy[a] += row[a] * p.y
                for b in 0..<4 { xtx[a * 4 + b] += row[a] * row[b] }
            }
        }
        guard let cx = LinearSolver.solve(xtx, xtpx, n: 4), let cy = LinearSolver.solve(xtx, xtpy, n: 4),
              let e0 = LinearSolver.solve(xtx, [1, 0, 0, 0], n: 4), let e3 = LinearSolver.solve(xtx, [0, 0, 0, 1], n: 4) else { return nil }
        var rss = 0.0
        for p in seg {
            let d = p.t - tau
            let row = [1, d, d * d, max(0, d)]
            var ex = p.x, ey = p.y
            for a in 0..<4 { ex -= row[a] * cx[a]; ey -= row[a] * cy[a] }
            rss += ex * ex + ey * ey
        }
        return (rss, (0..<4).map { Vec2(cx[$0], cy[$0]) }, e0[0], e3[3])
    }

    private func classify(_ k: Kink, pts: [TrackPoint], segBefore: ClosedRange<Int>, segAfter: ClosedRange<Int>,
                          geometry: CourtGeometry?) -> BallEvent {
        var kind: BallEvent.Kind
        var travel = 0.0
        var height: Double?
        let residual = k.gap   // 1σ image position of the contact from the regression

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
