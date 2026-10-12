import Foundation
import simd

/// Planar projective transform. `m` maps source → destination in homogeneous coordinates.
struct Homography: Equatable {
    var m: simd_double3x3

    static let identity = Homography(m: matrix_identity_double3x3)

    init(m: simd_double3x3) { self.m = m }

    init(rows r: [[Double]]) {
        m = simd_double3x3(rows: [Vec3(r[0][0], r[0][1], r[0][2]),
                                  Vec3(r[1][0], r[1][1], r[1][2]),
                                  Vec3(r[2][0], r[2][1], r[2][2])])
    }

    /// Element at row i, column j.
    func e(_ i: Int, _ j: Int) -> Double { m[j][i] }

    func homogeneous(_ p: Vec2) -> Vec3 { m * Vec3(p.x, p.y, 1) }

    func apply(_ p: Vec2) -> Vec2 {
        let v = homogeneous(p)
        return Vec2(v.x / v.z, v.y / v.z)
    }

    var inverse: Homography { Homography(m: m.inverse) }

    /// self followed by `next`.
    func then(_ next: Homography) -> Homography { Homography(m: next.m * m) }

    static func scale(_ sx: Double, _ sy: Double) -> Homography {
        Homography(rows: [[sx, 0, 0], [0, sy, 0], [0, 0, 1]])
    }

    /// Jacobian of the mapping at p as (du/dx, du/dy, dv/dx, dv/dy).
    func jacobian(at p: Vec2) -> (Double, Double, Double, Double) {
        let v = homogeneous(p)
        let w = v.z
        let u = v.x / w, y = v.y / w
        return ((e(0, 0) - u * e(2, 0)) / w, (e(0, 1) - u * e(2, 1)) / w,
                (e(1, 0) - y * e(2, 0)) / w, (e(1, 1) - y * e(2, 1)) / w)
    }

    /// Normalised DLT (Hartley) least-squares fit, h33 = 1.
    static func fit(from src: [Vec2], to dst: [Vec2]) -> Homography? {
        let n = src.count
        guard n >= 4, dst.count == n else { return nil }
        guard let (ts, sn) = normalize(src), let (td, dn) = normalize(dst) else { return nil }
        var ata = [Double](repeating: 0, count: 64)
        var atb = [Double](repeating: 0, count: 8)
        func add(_ row: [Double], _ b: Double) {
            for i in 0..<8 {
                atb[i] += row[i] * b
                for j in 0..<8 { ata[i * 8 + j] += row[i] * row[j] }
            }
        }
        for i in 0..<n {
            let x = sn[i].x, y = sn[i].y, u = dn[i].x, v = dn[i].y
            add([x, y, 1, 0, 0, 0, -u * x, -u * y], u)
            add([0, 0, 0, x, y, 1, -v * x, -v * y], v)
        }
        guard let h = LinearSolver.solve(ata, atb, n: 8) else { return nil }
        let hn = simd_double3x3(rows: [Vec3(h[0], h[1], h[2]), Vec3(h[3], h[4], h[5]), Vec3(h[6], h[7], 1)])
        var full = td.inverse * hn * ts
        let s = full[2][2]
        guard abs(s) > 1e-12, full.determinant.isFinite, abs(full.determinant) > 1e-15 else { return nil }
        full = full * (1 / s)
        return Homography(m: full)
    }

    private static func normalize(_ pts: [Vec2]) -> (simd_double3x3, [Vec2])? {
        var c = Vec2.zero
        for p in pts { c += p }
        c /= Double(pts.count)
        var d = 0.0
        for p in pts { d += (p - c).length }
        d /= Double(pts.count)
        guard d > 1e-12 else { return nil }
        let s = 2.0.squareRoot() / d
        let t = simd_double3x3(rows: [Vec3(s, 0, -s * c.x), Vec3(0, s, -s * c.y), Vec3(0, 0, 1)])
        return (t, pts.map { ($0 - c) * s })
    }
}

/// Pinhole camera looking at the court plane (court frame: x across, y along, z up; feet).
struct CameraPose {
    var f: Double
    var principal: Vec2
    var r: simd_double3x3   // world → camera rotation
    var t: Vec3             // world → camera translation

    var k: simd_double3x3 {
        simd_double3x3(rows: [Vec3(f, 0, principal.x), Vec3(0, f, principal.y), Vec3(0, 0, 1)])
    }

    /// Camera centre in court coordinates.
    var center: Vec3 { -(r.transpose * t) }

    func project(_ p: Vec3) -> Vec2? {
        let c = r * p + t
        guard c.z > 1e-6 else { return nil }
        return Vec2(f * c.x / c.z + principal.x, f * c.y / c.z + principal.y)
    }

    /// Ground plane (z = 0) → image homography.
    var groundHomography: Homography {
        let m = k * simd_double3x3(columns: (r.columns.0, r.columns.1, t))
        return Homography(m: m * (1 / m[2][2]))
    }

    /// Recover the pose from a court → image homography and a known focal length.
    static func from(homography h: Homography, f: Double, principal: Vec2) -> CameraPose? {
        let kInv = simd_double3x3(rows: [Vec3(1 / f, 0, -principal.x / f), Vec3(0, 1 / f, -principal.y / f), Vec3(0, 0, 1)])
        let mm = kInv * h.m
        var c0 = mm.columns.0, c1 = mm.columns.1, c2 = mm.columns.2
        let lambda = 2 / (simd_length(c0) + simd_length(c1))
        guard lambda.isFinite else { return nil }
        c0 *= lambda; c1 *= lambda; c2 *= lambda
        if c2.z < 0 { c0 = -c0; c1 = -c1; c2 = -c2 }
        let r1 = simd_normalize(c0)
        let r2 = simd_normalize(c1 - simd_dot(c1, r1) * r1)
        var r3 = simd_cross(r1, r2)
        var cols = (r1, r2, r3)
        var pose = CameraPose(f: f, principal: principal, r: simd_double3x3(columns: cols), t: c2)
        // The court's +z must point toward the camera (camera above the court).
        if pose.center.z < 0 {
            // Mirrored solution: keep x/y, point court +z toward the camera (orthogonal, det = -1).
            r3 = -r3
            cols = (r1, r2, r3)
            pose = CameraPose(f: f, principal: principal, r: simd_double3x3(columns: cols), t: c2)
        }
        return pose
    }

    /// Focal length (pixels) from horizontal field of view.
    static func focal(imageWidth: Double, hfovDegrees: Double) -> Double {
        (imageWidth / 2) / tan(hfovDegrees * .pi / 360)
    }

    /// Camera at `eye` looking at `target`, z-up world; used for tests and simulation.
    static func lookAt(eye: Vec3, target: Vec3, f: Double, principal: Vec2) -> CameraPose {
        let fwd = simd_normalize(target - eye)
        var right = simd_cross(fwd, Vec3(0, 0, 1))
        if simd_length(right) < 1e-6 { right = Vec3(1, 0, 0) }
        right = simd_normalize(right)
        let down = simd_cross(fwd, right)
        let r = simd_double3x3(rows: [right, down, fwd])
        return CameraPose(f: f, principal: principal, r: r, t: -(r * eye))
    }
}
