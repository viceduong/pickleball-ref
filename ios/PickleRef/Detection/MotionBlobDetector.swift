import CoreGraphics
import Foundation
import simd

/// Three-frame differencing + connected components on a downscaled luma image.
/// Cheap enough for every frame at 120–240 fps; yields small fast-moving blob candidates.
final class MotionBlobDetector {
    var threshold = 22
    var maxBlobs = 40

    private var prev1: GrayImage?
    private var prev2: GrayImage?
    private var visited: [UInt8] = []
    private var stack: [Int] = []

    /// - Parameters:
    ///   - roi: region in the small image's pixel coordinates.
    ///   - scale: small → full-resolution factor.
    ///   - maxArea: largest plausible ball blob area (small px), incl. motion blur.
    ///   - exclusion: full-resolution boxes (players) where blobs are down-weighted.
    func detect(_ cur: GrayImage, t: Double, roi: CGRect, scale: Double, maxArea: Int, exclusion: [CGRect]) -> [BallCandidate] {
        defer { prev2 = prev1; prev1 = cur }
        guard let p1 = prev1, let p2 = prev2, p1.width == cur.width, p2.width == cur.width else { return [] }
        let w = cur.width, h = cur.height
        let x0 = max(0, Int(roi.minX)), x1 = min(w - 1, Int(roi.maxX))
        let y0 = max(0, Int(roi.minY)), y1 = min(h - 1, Int(roi.maxY))
        guard x1 > x0, y1 > y0 else { return [] }
        if visited.count != w * h { visited = [UInt8](repeating: 0, count: w * h) } else {
            for i in visited.indices { visited[i] = 0 }
        }
        let thr = threshold
        var mask = [UInt8](repeating: 0, count: w * h)
        cur.pixels.withUnsafeBufferPointer { c in
            p1.pixels.withUnsafeBufferPointer { a in
                p2.pixels.withUnsafeBufferPointer { b in
                    for y in y0...y1 {
                        let row = y * w
                        for x in x0...x1 {
                            let i = row + x
                            let v = Int(c[i])
                            if abs(v - Int(a[i])) > thr && abs(v - Int(b[i])) > thr { mask[i] = 1 }
                        }
                    }
                }
            }
        }
        var out: [BallCandidate] = []
        var blobs = 0
        for y in y0...y1 {
            for x in x0...x1 {
                let i = y * w + x
                guard mask[i] != 0, visited[i] == 0 else { continue }
                // Flood fill.
                var area = 0, sx = 0, sy = 0, minX = x, maxX = x, minY = y, maxY = y, lum = 0
                stack.removeAll(keepingCapacity: true)
                stack.append(i); visited[i] = 1
                while let j = stack.popLast() {
                    let jx = j % w, jy = j / w
                    area += 1; sx += jx; sy += jy; lum += Int(cur.pixels[j])
                    minX = min(minX, jx); maxX = max(maxX, jx); minY = min(minY, jy); maxY = max(maxY, jy)
                    if area > maxArea * 3 { continue }
                    for n in [j - 1, j + 1, j - w, j + w] {
                        guard n >= 0, n < w * h else { continue }
                        let nx = n % w
                        if abs(nx - jx) > 1 { continue }
                        if mask[n] != 0 && visited[n] == 0 { visited[n] = 1; stack.append(n) }
                    }
                }
                blobs += 1
                if blobs > maxBlobs * 4 { return [] }   // global change (exposure, shake)
                guard area >= 1, area <= maxArea else { continue }
                let bw = maxX - minX + 1, bh = maxY - minY + 1
                let elong = Double(max(bw, bh)) / Double(min(bw, bh))
                guard elong <= 6 else { continue }
                let fill = Double(area) / Double(bw * bh)
                let centre = Vec2(Double(sx) / Double(area) + 0.5, Double(sy) / Double(area) + 0.5) * scale
                var score = 0.35 + 0.35 * min(1, fill) + 0.3 * min(1, Double(lum) / Double(area) / 200)
                let cp = CGPoint(x: centre.x, y: centre.y)
                if exclusion.contains(where: { $0.contains(cp) }) { score *= 0.4 }
                out.append(BallCandidate(t: t, p: centre, score: min(1, score), source: .motion))
            }
        }
        if out.count > maxBlobs { out = Array(out.sorted { $0.score > $1.score }.prefix(maxBlobs)) }
        return out
    }

    func reset() { prev1 = nil; prev2 = nil }
}
