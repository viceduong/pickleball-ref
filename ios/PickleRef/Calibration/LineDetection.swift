import Foundation
import simd

/// 8-bit luminance image, row-major.
struct GrayImage {
    let width: Int
    let height: Int
    var pixels: [UInt8]

    init(width: Int, height: Int, pixels: [UInt8]) {
        self.width = width; self.height = height; self.pixels = pixels
    }

    init(width: Int, height: Int, fill: UInt8) {
        self.init(width: width, height: height, pixels: [UInt8](repeating: fill, count: width * height))
    }

    @inline(__always) func at(_ x: Int, _ y: Int) -> UInt8 { pixels[y * width + x] }
}

/// Binary mask (0/1) with dimensions.
struct BinaryMask {
    let width: Int
    let height: Int
    var bits: [UInt8]

    @inline(__always) func at(_ x: Int, _ y: Int) -> Bool { bits[y * width + x] != 0 }

    func contains(_ p: Vec2) -> Bool {
        let x = Int(p.x.rounded()), y = Int(p.y.rounded())
        return x >= 0 && y >= 0 && x < width && y < height && bits[y * width + x] != 0
    }

    var points: [Vec2] {
        var out: [Vec2] = []
        out.reserveCapacity(bits.count / 20)
        for y in 0..<height {
            let row = y * width
            for x in 0..<width where bits[row + x] != 0 { out.append(Vec2(Double(x), Double(y))) }
        }
        return out
    }

    /// Square dilation with radius r (separable max filter).
    func dilated(_ r: Int) -> BinaryMask {
        var tmp = [UInt8](repeating: 0, count: bits.count)
        for y in 0..<height {
            let row = y * width
            for x in 0..<width where bits[row + x] != 0 {
                for k in max(0, x - r)...min(width - 1, x + r) { tmp[row + k] = 1 }
            }
        }
        var out = [UInt8](repeating: 0, count: bits.count)
        for y in 0..<height {
            let row = y * width
            for x in 0..<width where tmp[row + x] != 0 {
                for k in max(0, y - r)...min(height - 1, y + r) { out[k * width + x] = 1 }
            }
        }
        return BinaryMask(width: width, height: height, bits: out)
    }
}

enum LineMask {
    /// Court-line pixel test (Farin et al.): a line pixel is brighter than both neighbours at distance τ,
    /// horizontally or vertically. Wide bright areas (walls, sky) fail the test.
    static func detect(_ img: GrayImage, taus: [Int] = [2, 4, 7], contrast: Int = 16, minLuma: Int = 80) -> BinaryMask {
        let w = img.width, h = img.height
        var bits = [UInt8](repeating: 0, count: w * h)
        img.pixels.withUnsafeBufferPointer { p in
            for y in 0..<h {
                let row = y * w
                for x in 0..<w {
                    let l = Int(p[row + x])
                    if l < minLuma { continue }
                    var hit = false
                    for tau in taus {
                        if x - tau >= 0 && x + tau < w {
                            if l - Int(p[row + x - tau]) > contrast && l - Int(p[row + x + tau]) > contrast { hit = true; break }
                        }
                        if y - tau >= 0 && y + tau < h {
                            if l - Int(p[row - tau * w + x]) > contrast && l - Int(p[row + tau * w + x]) > contrast { hit = true; break }
                        }
                    }
                    if hit { bits[row + x] = 1 }
                }
            }
        }
        return BinaryMask(width: w, height: h, bits: bits)
    }
}

struct DetectedLine {
    var line: Line2D
    var support: Int
}

enum HoughLines {
    /// Standard Hough transform + peak NMS + total-least-squares refinement on supporting pixels.
    static func detect(points: [Vec2], width: Int, height: Int, maxLines: Int = 24, minVotes: Int = 40,
                       thetaStep: Double = 0.5) -> [DetectedLine] {
        guard points.count >= minVotes else { return [] }
        let cx = Double(width) / 2, cy = Double(height) / 2
        let nTheta = Int(180 / thetaStep)
        let dMax = Int((Double(width * width + height * height)).squareRoot() / 2) + 1
        let nRho = 2 * dMax + 1
        var cosT = [Double](repeating: 0, count: nTheta), sinT = cosT
        for i in 0..<nTheta {
            let th = Double(i) * thetaStep * .pi / 180
            cosT[i] = cos(th); sinT[i] = sin(th)
        }
        // Subsample very dense masks to bound cost.
        let stride = max(1, points.count / 60_000)
        var acc = [Int32](repeating: 0, count: nTheta * nRho)
        acc.withUnsafeMutableBufferPointer { a in
            var i = 0
            while i < points.count {
                let px = points[i].x - cx, py = points[i].y - cy
                for t in 0..<nTheta {
                    let r = Int((px * cosT[t] + py * sinT[t]).rounded()) + dMax
                    a[t * nRho + r] += 1
                }
                i += stride
            }
        }
        let vMin = Int32(max(8, minVotes / stride))
        var peaks: [(Int32, Int, Int)] = []
        let wt = 4, wr = 5
        for t in 0..<nTheta {
            for r in 0..<nRho {
                let v = acc[t * nRho + r]
                if v < vMin { continue }
                var isMax = true
                outer: for dt in -wt...wt {
                    let tt = t + dt
                    if tt < 0 || tt >= nTheta { continue }
                    for dr in -wr...wr {
                        let rr = r + dr
                        if rr < 0 || rr >= nRho || (dt == 0 && dr == 0) { continue }
                        let o = acc[tt * nRho + rr]
                        if o > v || (o == v && (dt < 0 || (dt == 0 && dr < 0))) { isMax = false; break outer }
                    }
                }
                if isMax { peaks.append((v, t, r)) }
            }
        }
        peaks.sort { $0.0 > $1.0 }

        var lines: [DetectedLine] = []
        for (_, t, r) in peaks.prefix(maxLines * 3) {
            let n = Vec2(cosT[t], sinT[t])
            let rho = Double(r - dMax)
            let raw = Line2D(n: n, c: rho + simd_dot(n, Vec2(cx, cy)))
            guard let refined = refine(raw, points: points, band: 2.5) else { continue }
            if lines.contains(where: { similar($0.line, refined.line) }) { continue }
            if refined.support < minVotes { continue }
            lines.append(refined)
            if lines.count >= maxLines { break }
        }
        return lines.sorted { $0.support > $1.support }
    }

    static func refine(_ line: Line2D, points: [Vec2], band: Double) -> DetectedLine? {
        var current = line
        var support: [Vec2] = []
        for _ in 0..<2 {
            support = points.filter { abs(current.distance($0)) <= band }
            guard support.count >= 2, let l = Line2D.fit(support) else { return nil }
            current = l
        }
        let count = points.reduce(0) { abs(current.distance($1)) <= 1.5 ? $0 + 1 : $0 }
        return DetectedLine(line: current, support: count)
    }

    static func similar(_ a: Line2D, _ b: Line2D, angleTol: Double = 2.0 * .pi / 180, offsetTol: Double = 6) -> Bool {
        let bb = b.aligned(to: a)
        let cosA = min(1, abs(simd_dot(a.n, bb.n)))
        return acos(cosA) < angleTol && abs(a.c - bb.c) < offsetTol
    }
}
