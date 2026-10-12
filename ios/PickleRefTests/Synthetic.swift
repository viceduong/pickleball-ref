import Foundation
import simd
@testable import PickleRef

/// Synthetic scenes for deterministic tests.
enum Synthetic {
    struct LCG {
        var s: UInt64
        mutating func next() -> Double {
            s = s &* 6364136223846793005 &+ 1442695040888963407
            return Double(s >> 11) / Double(1 << 53)
        }
        mutating func gauss() -> Double {
            let u = max(1e-12, next()), v = next()
            return (-2 * log(u)).squareRoot() * cos(2 * .pi * v)
        }
    }

    /// Camera behind the near baseline (or beside the court).
    static func camera(eye: Vec3, target: Vec3, width: Int, height: Int, hfov: Double = 69) -> CameraPose {
        CameraPose.lookAt(eye: eye, target: target, f: CameraPose.focal(imageWidth: Double(width), hfovDegrees: hfov),
                          principal: Vec2(Double(width) / 2, Double(height) / 2))
    }

    /// Renders court lines (anti-aliased by 3×3 supersampling near lines) on a noisy court surface.
    static func renderCourt(_ cam: CameraPose, width: Int, height: Int, seed: UInt64 = 1,
                            surround: UInt8 = 70, surface: UInt8 = 95, line: UInt8 = 225, noise: Double = 4) -> GrayImage {
        let h = cam.groundHomography
        let inv = h.inverse
        var rng = LCG(s: seed)
        var img = GrayImage(width: width, height: height, fill: 0)
        // Image-space segments for a cheap proximity test.
        let segs: [(Vec2, Vec2)] = Court.segments.compactMap { s in
            guard let a = cam.project(Vec3(s.a.x, s.a.y, 0)), let b = cam.project(Vec3(s.b.x, s.b.y, 0)) else { return nil }
            return (a, b)
        }
        func nearLine(_ p: Vec2) -> Bool {
            for (a, b) in segs {
                let d = b - a
                let u = max(0, min(1, simd_dot(p - a, d) / max(1e-9, simd_length_squared(d))))
                if (a + d * u - p).length < 3 { return true }
            }
            return false
        }
        func onLine(_ g: Vec2) -> Bool {
            for s in Court.segments {
                let hw = Court.lineWidth / 2
                if s.family == .across {
                    if abs(g.y - s.value) <= hw && g.x >= s.from - hw && g.x <= s.to + hw { return true }
                } else {
                    if abs(g.x - s.value) <= hw && g.y >= s.from - hw && g.y <= s.to + hw { return true }
                }
            }
            return false
        }
        for y in 0..<height {
            for x in 0..<width {
                let p = Vec2(Double(x), Double(y))
                let v = inv.homogeneous(p)
                var value: Double
                if v.z <= 0 {
                    value = Double(surround)   // above horizon
                } else {
                    let g = Vec2(v.x / v.z, v.y / v.z)
                    let inside = g.x > -6 && g.x < Court.width + 6 && g.y > -8 && g.y < Court.length + 8
                    value = Double(inside ? surface : surround)
                    if nearLine(p) {
                        var cover = 0.0
                        for sy in 0..<3 {
                            for sx in 0..<3 {
                                let q = inv.apply(Vec2(Double(x) + (Double(sx) - 1) / 3, Double(y) + (Double(sy) - 1) / 3))
                                if onLine(q) { cover += 1 }
                            }
                        }
                        cover /= 9
                        value = value * (1 - cover) + Double(line) * cover
                    }
                }
                value += rng.gauss() * noise
                img.pixels[y * width + x] = UInt8(max(0, min(255, value.rounded())))
            }
        }
        return img
    }

    /// Simulates a ball flight with gravity, quadratic drag and a bounce (COR 0.7, friction 0.85).
    /// Returns image observations (t, px) at `fps` with Gaussian pixel noise, plus true bounce point.
    static func flight(cam: CameraPose, start: Vec3, velocity: Vec3, fps: Double, duration: Double,
                       noisePx: Double, seed: UInt64 = 7, hitAt: Double? = nil) -> (pts: [TrackPoint], bounce: Vec2?, bounceT: Double?) {
        var rng = LCG(s: seed)
        var p = start, v = velocity
        let dt = 1.0 / 2000
        var t = 0.0
        var nextFrame = 0.0
        var out: [TrackPoint] = []
        var bounce: Vec2?
        var bounceT: Double?
        var hit = false
        while t < duration {
            if t >= nextFrame {
                if let px = cam.project(p) {
                    out.append(TrackPoint(t: t, p: px + Vec2(rng.gauss(), rng.gauss()) * noisePx, source: .motion))
                }
                nextFrame += 1 / fps
            }
            let speed = simd_length(v)
            let a = Vec3(0, 0, -Trajectory3D.gravity) - v * (Trajectory3D.quadDrag * speed)
            v += a * dt
            p += v * dt
            if p.z <= Court.ballRadius && v.z < 0 {
                if bounce == nil { bounce = Vec2(p.x, p.y); bounceT = t }
                p.z = Court.ballRadius
                v = Vec3(v.x * 0.85, v.y * 0.85, -v.z * 0.7)
            }
            if let h = hitAt, !hit, t >= h {
                hit = true
                v = Vec3(-v.x * 0.3, -v.y * 0.9, 8)
            }
            t += dt
        }
        return (out, bounce, bounceT)
    }
}
