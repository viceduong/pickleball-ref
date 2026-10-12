import Foundation
import simd

struct CalibrationResult {
    /// Court (feet) → image pixels of the analysed image.
    var courtToImage: Homography
    var hits: Int
    var visibleSamples: Int
    var rmsPx: Double
    var imageWidth: Int
    var imageHeight: Int

    var inlierRatio: Double { visibleSamples > 0 ? Double(hits) / Double(visibleSamples) : 0 }

    /// Same calibration expressed for an image scaled by `s` (e.g. analysed at half resolution).
    func scaled(by s: Double) -> CalibrationResult {
        var r = self
        r.courtToImage = courtToImage.then(.scale(s, s))
        r.rmsPx *= s
        r.imageWidth = Int(Double(imageWidth) * s)
        r.imageHeight = Int(Double(imageHeight) * s)
        return r
    }
}

/// Fully automatic court calibration: line-pixel mask → Hough lines → two line families →
/// exhaustive model-fitting over line/model-line correspondences → refinement on all court lines.
final class CourtCalibrator {
    var maxLinesPerFamily = 7
    var minInlierRatio = 0.5
    var minHits = 60

    private let samples = Court.samples(step: 0.5)
    private let coarseSamples = Court.samples(step: 1.0)
    private var stamp: [Int32] = []
    private var generation: Int32 = 0

    func calibrate(_ img: GrayImage) -> CalibrationResult? {
        let mask = LineMask.detect(img)
        let pts = mask.points
        guard pts.count > 200 else { return nil }
        let minVotes = max(30, min(img.width, img.height) / 10)
        let lines = HoughLines.detect(points: pts, width: img.width, height: img.height, maxLines: 22, minVotes: minVotes)
        guard lines.count >= 4 else { return nil }
        let wide = mask.dilated(2)
        guard let best = search(lines: lines, mask: wide, width: img.width, height: img.height) else { return nil }
        var h = best
        let s0 = score(h, samples: samples, mask: wide)
        var hScore = 2 * s0.hits - s0.visible
        for band in [6.0, 3.0, 3.0] {
            guard let r = refine(h, maskPoints: pts, width: img.width, height: img.height, band: band) else { break }
            let sc = score(r, samples: samples, mask: wide)
            let rScore = 2 * sc.hits - sc.visible
            if rScore + 2 < hScore { break }   // refinement made it worse
            h = r; hScore = rScore
        }
        let (hits, vis) = score(h, samples: samples, mask: wide)
        guard vis > 0 else { return nil }
        let result = CalibrationResult(courtToImage: h, hits: hits, visibleSamples: vis,
                                       rmsPx: residual(h, maskPoints: pts),
                                       imageWidth: img.width, imageHeight: img.height)
        guard result.hits >= minHits, result.inlierRatio >= minInlierRatio else { return nil }
        return result
    }

    // MARK: - Families

    /// Split lines into the two court directions. Strong perspective spreads one family over a wide angle range,
    /// so angles are unreliable; instead lines that physically cross (inside both painted extents) must belong to
    /// different families → 2-colour the crossing graph. Lines that cross nothing go into both families.
    static func families(_ lines: [DetectedLine]) -> ([DetectedLine], [DetectedLine]) {
        let n = lines.count
        var cross = [[Bool]](repeating: [Bool](repeating: false, count: n), count: n)
        for i in 0..<n { for j in (i + 1)..<max(i + 1, n) { let c = lines[i].crosses(lines[j]); cross[i][j] = c; cross[j][i] = c } }
        var colour = [Int?](repeating: nil, count: n)
        for s in 0..<n where colour[s] == nil && cross[s].contains(true) {
            colour[s] = 0
            var queue = [s]
            while let i = queue.popLast() {
                for j in 0..<n where cross[i][j] && colour[j] == nil {
                    colour[j] = 1 - colour[i]!
                    queue.append(j)
                }
            }
        }
        let a = lines.indices.filter { colour[$0] != 1 }.map { lines[$0] }
        let b = lines.indices.filter { colour[$0] != 0 }.map { lines[$0] }
        return (a, b)
    }

    // MARK: - Hypothesis search

    private func search(lines: [DetectedLine], mask: BinaryMask, width: Int, height: Int) -> Homography? {
        let (fa, fb) = Self.families(lines)
        let a = Array(fa.prefix(maxLinesPerFamily)), b = Array(fb.prefix(maxLinesPerFamily))
        guard a.count >= 2, b.count >= 2 else { return nil }

        let acrossPairs = pairs(Court.acrossValues.count)
        let alongPairs = pairs(Court.alongValues.count)
        var bestScore = -Double.infinity
        var bestH: Homography?
        let imgArea = Double(width * height)
        let maxCoord = 6.0 * Double(max(width, height))

        for (acrossImg, alongImg) in [(a, b), (b, a)] {
            for (i, j) in pairs(acrossImg.count) {
                for (k, l) in pairs(alongImg.count) {
                    let li = acrossImg[i].line, lj = acrossImg[j].line, lk = alongImg[k].line, ll = alongImg[l].line
                    guard let pik = li.intersect(lk), let pil = li.intersect(ll),
                          let pjk = lj.intersect(lk), let pjl = lj.intersect(ll) else { continue }
                    if [pik, pil, pjk, pjl].contains(where: { abs($0.x) > maxCoord || abs($0.y) > maxCoord }) { continue }
                    for (wa, wb) in acrossPairs {
                        for (ya, yb) in [(Court.acrossValues[wa], Court.acrossValues[wb]), (Court.acrossValues[wb], Court.acrossValues[wa])] {
                            for (xa, xb) in alongPairs {
                                for (xk, xl) in [(Court.alongValues[xa], Court.alongValues[xb]), (Court.alongValues[xb], Court.alongValues[xa])] {
                                    let src = [Vec2(xk, ya), Vec2(xl, ya), Vec2(xk, yb), Vec2(xl, yb)]
                                    let dst = [pik, pil, pjk, pjl]
                                    guard let h = Homography.fit(from: src, to: dst),
                                          plausible(h, imgArea: imgArea) else { continue }
                                    let (ch, cv) = score(h, samples: coarseSamples, mask: mask)
                                    let s = Double(2 * ch - cv)   // hits − misses
                                    if s > bestScore { bestScore = s; bestH = h }
                                }
                            }
                        }
                    }
                }
            }
        }
        return bestH
    }

    private func pairs(_ n: Int) -> [(Int, Int)] {
        var out: [(Int, Int)] = []
        for i in 0..<n { for j in (i + 1)..<max(i + 1, n) { out.append((i, j)) } }
        return out
    }

    /// Court must be in front of the camera, convex, and of a sane size.
    private func plausible(_ h: Homography, imgArea: Double) -> Bool {
        let hs = Court.corners.map { h.homogeneous($0) }
        let sign = hs[0].z > 0
        guard hs.allSatisfy({ ($0.z > 0) == sign && abs($0.z) > 1e-9 }) else { return false }
        let p = hs.map { Vec2($0.x / $0.z, $0.y / $0.z) }
        var crossSign = 0.0
        var area = 0.0
        for i in 0..<4 {
            let a = p[i], b = p[(i + 1) % 4], c = p[(i + 2) % 4]
            let cr = (b.x - a.x) * (c.y - b.y) - (b.y - a.y) * (c.x - b.x)
            if crossSign == 0 { crossSign = cr } else if cr * crossSign <= 0 { return false }
            area += a.x * b.y - b.x * a.y
        }
        area = abs(area) / 2
        return area > 0.03 * imgArea && area < 60 * imgArea
    }

    /// Counts distinct 2×2-pixel cells (not samples) so degenerate homographies that collapse the court onto a
    /// single image line cannot accumulate hits.
    private func score(_ h: Homography, samples: [Vec2], mask: BinaryMask) -> (hits: Int, visible: Int) {
        let cw = (mask.width + 1) / 2, cells = cw * ((mask.height + 1) / 2)
        if stamp.count != cells { stamp = [Int32](repeating: 0, count: cells); generation = 0 }
        generation &+= 2
        if generation <= 0 { for i in stamp.indices { stamp[i] = 0 }; generation = 2 }
        let seen = generation, hitMark = generation + 1
        var hits = 0, vis = 0
        let w = Double(mask.width - 1), hh = Double(mask.height - 1)
        for s in samples {
            let v = h.homogeneous(s)
            if v.z <= 0 { continue }
            let x = v.x / v.z, y = v.y / v.z
            if x < 0 || y < 0 || x > w || y > hh { continue }
            let xi = Int(x.rounded()), yi = Int(y.rounded())
            let cell = (yi / 2) * cw + xi / 2
            let isHit = mask.bits[yi * mask.width + xi] != 0
            if stamp[cell] < seen { stamp[cell] = seen; vis += 1 }
            if isHit && stamp[cell] != hitMark { stamp[cell] = hitMark; hits += 1 }
        }
        return (hits, vis)
    }

    // MARK: - Refinement

    /// Re-fit every visible court line on the mask pixels it owns (each pixel belongs to its nearest predicted
    /// segment, so junctions don't bias short lines), then re-estimate H from all line intersections.
    private func refine(_ h: Homography, maskPoints: [Vec2], width: Int, height: Int, band: Double) -> Homography? {
        struct Pred { let family: Court.Family; let value: Double; let line: Line2D; let a: Vec2; let dir: Vec2; let len: Double }
        var preds: [Pred] = []
        for s in Court.segments {
            let a = h.apply(s.a), b = h.apply(s.b)
            guard a.x.isFinite, a.y.isFinite, b.x.isFinite, b.y.isFinite, (b - a).length > 1 else { continue }
            preds.append(Pred(family: s.family, value: s.value, line: Line2D(through: a, b), a: a, dir: (b - a).normalized, len: (b - a).length))
        }
        guard !preds.isEmpty else { return nil }
        var owned = [[Vec2]](repeating: [], count: preds.count)
        for p in maskPoints {
            var best = -1, bestD = band
            for (k, pr) in preds.enumerated() {
                let u = simd_dot(p - pr.a, pr.dir)
                if u < -2 || u > pr.len + 2 { continue }
                let d = abs(pr.line.distance(p))
                if d <= bestD { bestD = d; best = k }
            }
            if best >= 0 { owned[best].append(p) }
        }
        var across: [Double: Line2D] = [:]
        var along: [Double: Line2D] = [:]
        var keys: [(Court.Family, Double)] = []
        for pr in preds where !keys.contains(where: { $0.0 == pr.family && $0.1 == pr.value }) { keys.append((pr.family, pr.value)) }
        for (family, value) in keys {
            let idx = preds.indices.filter { preds[$0].family == family && preds[$0].value == value }
            let support = idx.flatMap { owned[$0] }
            // Visible predicted length (px) — require support proportional to it.
            var visible = 0.0
            for k in idx {
                let pr = preds[k]
                var inside = 0
                for i in 0..<100 {
                    let q = pr.a + pr.dir * (pr.len * Double(i) / 99)
                    if q.x >= 0 && q.y >= 0 && q.x < Double(width) && q.y < Double(height) { inside += 1 }
                }
                visible += pr.len * Double(inside) / 100
            }
            guard Double(support.count) >= max(25, 0.5 * visible), let fitted = Line2D.fit(support) else { continue }
            guard HoughLines.similar(preds[idx[0]].line, fitted, angleTol: 4 * .pi / 180, offsetTol: band + 2) else { continue }
            if family == .across { across[value] = fitted } else { along[value] = fitted }
        }
        var src: [Vec2] = [], dst: [Vec2] = []
        let margin = 0.25 * Double(max(width, height))
        for (y, ly) in across {
            for (x, lx) in along {
                guard let p = ly.intersect(lx) else { continue }
                if p.x < -margin || p.y < -margin || p.x > Double(width) + margin || p.y > Double(height) + margin { continue }
                src.append(Vec2(x, y)); dst.append(p)
            }
        }
        guard src.count >= 4, across.count >= 2, along.count >= 2 else { return nil }
        return Homography.fit(from: src, to: dst)
    }

    private func residual(_ h: Homography, maskPoints: [Vec2]) -> Double {
        var lines: [Line2D] = []
        for s in Court.segments {
            let a = h.apply(s.a), b = h.apply(s.b)
            if (b - a).length > 1 { lines.append(Line2D(through: a, b)) }
        }
        var sum = 0.0, n = 0
        for p in maskPoints {
            let d = lines.map { abs($0.distance(p)) }.min() ?? .infinity
            if d <= 3 { sum += d * d; n += 1 }
        }
        return n > 0 ? (sum / Double(n)).squareRoot() : 0
    }
}
