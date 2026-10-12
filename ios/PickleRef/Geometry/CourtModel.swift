import Foundation
import simd

/// USA Pickleball court geometry in feet. Court frame: x across (0…20), y along (0…44), net at y = 22.
/// Boundary coordinates are outer line edges; lines are part of the area they bound (rule 3.A.4).
enum Court {
    static let width = 20.0
    static let length = 44.0
    static let net = 22.0
    static let nvzDepth = 7.0
    static let lineWidth = 2.0 / 12.0
    static let half = lineWidth / 2
    static let ballRadius = 2.92 / 2 / 12.0
    static let feetToCm = 30.48

    enum Family { case across, along }   // across: constant y (baselines, NVZ lines); along: constant x

    struct Segment {
        let family: Family
        let value: Double        // y for `.across`, x for `.along` (line centre)
        let from: Double, to: Double
        var a: Vec2 { family == .across ? Vec2(from, value) : Vec2(value, from) }
        var b: Vec2 { family == .across ? Vec2(to, value) : Vec2(value, to) }
    }

    /// Painted line centrelines.
    static let acrossValues: [Double] = [half, net - nvzDepth + half, net + nvzDepth - half, length - half]
    static let alongValues: [Double] = [half, width / 2, width - half]

    static let segments: [Segment] = [
        Segment(family: .across, value: acrossValues[0], from: 0, to: width),
        Segment(family: .across, value: acrossValues[1], from: 0, to: width),
        Segment(family: .across, value: acrossValues[2], from: 0, to: width),
        Segment(family: .across, value: acrossValues[3], from: 0, to: width),
        Segment(family: .along, value: alongValues[0], from: 0, to: length),
        Segment(family: .along, value: alongValues[2], from: 0, to: length),
        Segment(family: .along, value: alongValues[1], from: 0, to: net - nvzDepth),
        Segment(family: .along, value: alongValues[1], from: net + nvzDepth, to: length),
    ]

    /// Points sampled along every painted line (court frame) for model scoring.
    static func samples(step: Double) -> [Vec2] {
        var out: [Vec2] = []
        for s in segments {
            let len = s.to - s.from
            let n = max(2, Int(len / step))
            for i in 0...n {
                let u = s.from + len * Double(i) / Double(n)
                out.append(s.family == .across ? Vec2(u, s.value) : Vec2(s.value, u))
            }
        }
        return out
    }

    static let corners: [Vec2] = [Vec2(0, 0), Vec2(width, 0), Vec2(width, length), Vec2(0, length)]

    /// Axis-aligned region in court coordinates.
    struct Region: Equatable {
        var x0, x1, y0, y1: Double

        func contains(_ p: Vec2) -> Bool { p.x >= x0 && p.x <= x1 && p.y >= y0 && p.y <= y1 }

        /// Positive inside (distance to nearest edge), negative outside (distance to region).
        func signedMargin(_ p: Vec2) -> (margin: Double, normal: Vec2) {
            if contains(p) {
                let d = [(p.x - x0, Vec2(-1, 0)), (x1 - p.x, Vec2(1, 0)), (p.y - y0, Vec2(0, -1)), (y1 - p.y, Vec2(0, 1))]
                let m = d.min { $0.0 < $1.0 }!
                return (m.0, m.1)
            }
            let dx = max(x0 - p.x, 0, p.x - x1)
            let dy = max(y0 - p.y, 0, p.y - y1)
            let normal = Vec2(p.x < x0 ? -1 : (p.x > x1 ? 1 : 0), p.y < y0 ? -1 : (p.y > y1 ? 1 : 0)).normalized
            return (-(dx * dx + dy * dy).squareRoot(), normal)
        }
    }

    enum Half { case near, far }
    enum Side { case left, right }   // x < 10 is .left

    static func half(of p: Vec2) -> Half { p.y < net ? .near : .far }
    static func side(of p: Vec2) -> Side { p.x < width / 2 ? .left : .right }

    static let fullCourt = Region(x0: 0, x1: width, y0: 0, y1: length)

    static func halfCourt(_ h: Half) -> Region {
        h == .near ? Region(x0: 0, x1: width, y0: 0, y1: net) : Region(x0: 0, x1: width, y0: net, y1: length)
    }

    /// Service court that a serve must land in. NVZ line is NOT part of it (rule 4.A/7 – serve on NVZ line is short),
    /// centreline and sideline/baseline are.
    static func serviceCourt(half h: Half, side s: Side) -> Region {
        let xr: (Double, Double) = s == .left ? (0, width / 2 + half) : (width / 2 - half, width)
        let yr: (Double, Double) = h == .near ? (0, net - nvzDepth) : (net + nvzDepth, length)
        return Region(x0: xr.0, x1: xr.1, y0: yr.0, y1: yr.1)
    }

    /// Non-volley zone including its lines (rule 3.A.4.c, 2-dimensional).
    static func kitchen(_ h: Half) -> Region {
        h == .near ? Region(x0: 0, x1: width, y0: net - nvzDepth, y1: net)
                   : Region(x0: 0, x1: width, y0: net, y1: net + nvzDepth)
    }
}
