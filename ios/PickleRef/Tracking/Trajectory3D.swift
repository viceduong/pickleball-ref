import Foundation
import simd

/// Monocular 3D ball arc with a physics prior: X(t) = p0 + v0·φ(t) + G(t), gravity + linearised drag.
/// With a calibrated camera every image observation gives two equations that are linear in (p0, v0),
/// so the arc is recovered by weighted linear least squares (gravity fixes the metric depth).
struct Trajectory3D {
    static let gravity = 32.174          // ft/s²
    static let quadDrag = 0.0106         // 1/ft  (½ρ·Cd·A / m for a pickleball, Cd≈0.35)

    var t0: Double
    var p0: Vec3
    var v0: Vec3
    var k: Double                         // linear drag coefficient (1/s)
    var rmsPx: Double

    private static func phi(_ t: Double, _ k: Double) -> Double { k < 1e-6 ? t : (1 - exp(-k * t)) / k }
    private static func gz(_ t: Double, _ k: Double) -> Double {
        k < 1e-6 ? -0.5 * gravity * t * t : -gravity * (t - phi(t, k)) / k
    }

    func position(_ t: Double) -> Vec3 {
        let d = t - t0
        return p0 + v0 * Self.phi(d, k) + Vec3(0, 0, Self.gz(d, k))
    }

    func velocity(_ t: Double) -> Vec3 {
        let d = t - t0
        let e = exp(-k * d)
        let vzg = k < 1e-6 ? -Self.gravity * d : -Self.gravity * (1 - e) / k
        return v0 * e + Vec3(0, 0, vzg)
    }

    /// First time ≥ `after` where the ball centre reaches height `z` while descending (bisection).
    func timeAtHeight(_ z: Double, after: Double, before: Double) -> Double? {
        let f = { (t: Double) in self.position(t).z - z }
        var a = after, b = before
        guard f(a) > 0, f(b) <= 0 else { return nil }
        for _ in 0..<50 {
            let m = 0.5 * (a + b)
            if f(m) > 0 { a = m } else { b = m }
        }
        return 0.5 * (a + b)
    }

    static func fit(_ pts: [TrackPoint], pose: CameraPose) -> Trajectory3D? {
        guard pts.count >= 6, let first = pts.first, let last = pts.last, last.t - first.t >= 0.08 else { return nil }
        let a = pose.k * pose.r
        let b = pose.k * pose.t
        let rows = (a.transpose.columns.0, a.transpose.columns.1, a.transpose.columns.2)  // rows of A
        let t0 = first.t
        var k = 0.0
        var sol: (Vec3, Vec3)?
        var weights = [Double](repeating: 1, count: pts.count)
        for iteration in 0..<4 {
            var ata = [Double](repeating: 0, count: 36)
            var atb = [Double](repeating: 0, count: 6)
            for (i, o) in pts.enumerated() {
                let d = o.t - t0
                let ph = phi(d, k)
                let g = Vec3(0, 0, gz(d, k))
                for (row, bb, obs) in [(rows.0, b.x, o.x), (rows.1, b.y, o.y)] {
                    let c = row - obs * rows.2
                    let rhs = -(bb - obs * b.z) - simd_dot(c, g)
                    let r = [c.x, c.y, c.z, ph * c.x, ph * c.y, ph * c.z]
                    let w = weights[i] * weights[i]
                    for p in 0..<6 {
                        atb[p] += w * r[p] * rhs
                        for q in 0..<6 { ata[p * 6 + q] += w * r[p] * r[q] }
                    }
                }
            }
            guard let x = LinearSolver.solve(ata, atb, n: 6) else { return nil }
            let p0 = Vec3(x[0], x[1], x[2]), v0 = Vec3(x[3], x[4], x[5])
            sol = (p0, v0)
            // Reweight by inverse depth so equations approximate reprojection error; update drag from speed.
            let tmp = Trajectory3D(t0: t0, p0: p0, v0: v0, k: k, rmsPx: 0)
            for (i, o) in pts.enumerated() {
                let cam = pose.r * tmp.position(o.t) + pose.t
                weights[i] = cam.z > 0.1 ? 1 / cam.z : 1
            }
            if iteration >= 1 && iteration < 3 { k = quadDrag * simd_length(v0) }
        }
        guard let (p0, v0) = sol else { return nil }
        var traj = Trajectory3D(t0: t0, p0: p0, v0: v0, k: k, rmsPx: 0)
        var err = 0.0
        for o in pts {
            guard let p = pose.project(traj.position(o.t)) else { return nil }
            err += simd_length_squared(p - o.p)
        }
        traj.rmsPx = (err / Double(pts.count)).squareRoot()
        return traj
    }
}
