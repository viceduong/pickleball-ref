import Foundation
import simd

typealias Vec2 = SIMD2<Double>
typealias Vec3 = SIMD3<Double>

extension SIMD2 where Scalar == Double {
    var length: Double { (x * x + y * y).squareRoot() }
    var normalized: Vec2 { let l = length; return l > 0 ? self / l : self }
}

enum LinearSolver {
    /// Solves A x = b for a dense n x n system (row-major `a`) with partial pivoting.
    static func solve(_ a: [Double], _ b: [Double], n: Int) -> [Double]? {
        var m = a
        var r = b
        for col in 0..<n {
            var pivot = col
            var best = abs(m[col * n + col])
            for row in (col + 1)..<n where abs(m[row * n + col]) > best {
                best = abs(m[row * n + col])
                pivot = row
            }
            if best < 1e-12 { return nil }
            if pivot != col {
                for k in 0..<n { m.swapAt(col * n + k, pivot * n + k) }
                r.swapAt(col, pivot)
            }
            let d = m[col * n + col]
            for row in (col + 1)..<n {
                let f = m[row * n + col] / d
                if f == 0 { continue }
                for k in col..<n { m[row * n + k] -= f * m[col * n + k] }
                r[row] -= f * r[col]
            }
        }
        var x = [Double](repeating: 0, count: n)
        for row in stride(from: n - 1, through: 0, by: -1) {
            var s = r[row]
            for k in (row + 1)..<n { s -= m[row * n + k] * x[k] }
            x[row] = s / m[row * n + row]
        }
        return x
    }
}

/// Infinite 2D line n·p = c with |n| = 1.
struct Line2D: Equatable {
    var n: Vec2
    var c: Double

    init(n: Vec2, c: Double) {
        let l = n.length
        self.n = n / l
        self.c = c / l
    }

    init(through a: Vec2, _ b: Vec2) {
        let d = (b - a).normalized
        self.init(n: Vec2(-d.y, d.x), c: simd_dot(Vec2(-d.y, d.x), a))
    }

    /// Signed distance.
    func distance(_ p: Vec2) -> Double { simd_dot(n, p) - c }

    var direction: Vec2 { Vec2(-n.y, n.x) }

    /// Direction angle in [0, π).
    var angle: Double {
        var a = atan2(direction.y, direction.x)
        if a < 0 { a += .pi }
        if a >= .pi { a -= .pi }
        return a
    }

    func intersect(_ o: Line2D) -> Vec2? {
        let det = n.x * o.n.y - n.y * o.n.x
        if abs(det) < 1e-9 { return nil }
        return Vec2((c * o.n.y - n.y * o.c) / det, (n.x * o.c - c * o.n.x) / det)
    }

    /// Returns the same line with the normal oriented to agree with `other`.
    func aligned(to other: Line2D) -> Line2D {
        simd_dot(n, other.n) < 0 ? Line2D(n: -n, c: -c) : self
    }

    /// Total-least-squares fit.
    static func fit(_ pts: [Vec2]) -> Line2D? {
        guard pts.count >= 2 else { return nil }
        var mean = Vec2.zero
        for p in pts { mean += p }
        mean /= Double(pts.count)
        var sxx = 0.0, sxy = 0.0, syy = 0.0
        for p in pts {
            let d = p - mean
            sxx += d.x * d.x; sxy += d.x * d.y; syy += d.y * d.y
        }
        let theta = 0.5 * atan2(2 * sxy, sxx - syy)
        let n = Vec2(-sin(theta), cos(theta))
        return Line2D(n: n, c: simd_dot(n, mean))
    }
}

/// Least-squares quadratic p(t) = a + b t + c t² fitted per axis, t relative to `t0`.
struct QuadFit {
    let t0: Double
    let a: Vec2, b: Vec2, c: Vec2
    let rms: Double

    func position(_ t: Double) -> Vec2 { let d = t - t0; return a + b * d + c * (d * d) }
    func velocity(_ t: Double) -> Vec2 { let d = t - t0; return b + c * (2 * d) }

    static func fit(_ ts: [Double], _ ps: [Vec2], t0: Double) -> QuadFit? {
        let n = ts.count
        guard n >= 3, ps.count == n else { return nil }
        var s = [Double](repeating: 0, count: 5)
        var bx = [Double](repeating: 0, count: 3)
        var by = [Double](repeating: 0, count: 3)
        for i in 0..<n {
            let d = ts[i] - t0
            var pw = 1.0
            for k in 0..<5 { s[k] += pw; pw *= d }
            let d2 = d * d
            bx[0] += ps[i].x; bx[1] += ps[i].x * d; bx[2] += ps[i].x * d2
            by[0] += ps[i].y; by[1] += ps[i].y * d; by[2] += ps[i].y * d2
        }
        let m = [s[0], s[1], s[2], s[1], s[2], s[3], s[2], s[3], s[4]]
        guard let cx = LinearSolver.solve(m, bx, n: 3), let cy = LinearSolver.solve(m, by, n: 3) else { return nil }
        let a = Vec2(cx[0], cy[0]), b = Vec2(cx[1], cy[1]), c = Vec2(cx[2], cy[2])
        var err = 0.0
        for i in 0..<n {
            let d = ts[i] - t0
            let e = ps[i] - (a + b * d + c * (d * d))
            err += simd_length_squared(e)
        }
        let dof = max(1, n - 3)
        return QuadFit(t0: t0, a: a, b: b, c: c, rms: (err / Double(dof)).squareRoot())
    }
}

enum Stats {
    static func median(_ v: [Double]) -> Double? {
        guard !v.isEmpty else { return nil }
        let s = v.sorted()
        return s.count % 2 == 1 ? s[s.count / 2] : 0.5 * (s[s.count / 2 - 1] + s[s.count / 2])
    }

    /// Standard normal CDF.
    static func phi(_ x: Double) -> Double { 0.5 * erfc(-x / 2.0.squareRoot()) }
}
