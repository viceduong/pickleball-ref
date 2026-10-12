import Foundation
import simd

struct RefCall: Identifiable, Codable {
    enum Kind: String, Codable { case serve = "Serve", line = "Line", kitchen = "Kitchen", doubleBounce = "Double bounce" }
    enum Verdict: String, Codable { case inBounds = "IN", out = "OUT", tooClose = "TOO CLOSE", fault = "FAULT", clean = "CLEAN" }

    var id = UUID()
    var kind: Kind
    var verdict: Verdict
    var t: Double
    var courtX: Double
    var courtY: Double
    /// Signed distance (cm) of the ball contact to the boundary: + inside, − outside.
    var marginCm: Double
    var sigmaCm: Double
    /// Probability that the verdict is correct given the measurement model.
    var confidence: Double
    var rule: String
    var detail: String
    var imageX: Double
    var imageY: Double
    var rallyId: Int
    var fusedPeers: Int = 0

    var courtPoint: Vec2 { Vec2(courtX, courtY) }

    /// What players should do: TOO CLOSE resolves to IN under rule 8.D.
    var headline: String {
        switch verdict {
        case .tooClose: return "TOO CLOSE → IN"
        case .fault: return "\(kind.rawValue.uppercased()) FAULT"
        default: return verdict.rawValue
        }
    }
}

/// Measured ball contact with uncertainty, in court feet.
struct ContactMeasurement {
    var point: Vec2
    /// σ (ft) of the contact along x and y.
    var sigma: Vec2
    var t: Double
    var imagePoint: Vec2
}

enum LineJudge {
    static let contactPatchRadius = 0.6 / 12.0   // ft; compressed ball footprint touching the line counts IN

    /// Verdict for a contact against a region whose boundary lines count as IN.
    static func judge(_ m: ContactMeasurement, region: Court.Region, minSigmaCm: Double = 0.8)
        -> (verdict: RefCall.Verdict, marginCm: Double, sigmaCm: Double, confidence: Double) {
        let (margin, normal) = region.signedMargin(m.point)
        let marginFt = margin + contactPatchRadius
        let sx = m.sigma.x * normal.x, sy = m.sigma.y * normal.y
        var sigmaFt = (sx * sx + sy * sy).squareRoot()
        if normal == .zero { sigmaFt = max(m.sigma.x, m.sigma.y) }
        let sigmaCm = max(minSigmaCm, sigmaFt * Court.feetToCm)
        let marginCm = marginFt * Court.feetToCm
        let z = marginCm / sigmaCm
        let verdict: RefCall.Verdict
        if z >= 2 { verdict = .inBounds } else if z <= -2 { verdict = .out } else { verdict = .tooClose }
        let confidence = verdict == .tooClose ? Stats.phi(abs(z)) : Stats.phi(abs(z))
        return (verdict, marginCm, sigmaCm, confidence)
    }

    /// Inverse-variance fusion of contact measurements from several cameras (per axis).
    static func fuse(_ ms: [ContactMeasurement]) -> ContactMeasurement? {
        guard let first = ms.first else { return nil }
        if ms.count == 1 { return first }
        var wx = 0.0, wy = 0.0, x = 0.0, y = 0.0
        for m in ms {
            let ax = 1 / max(1e-6, m.sigma.x * m.sigma.x), ay = 1 / max(1e-6, m.sigma.y * m.sigma.y)
            wx += ax; wy += ay; x += ax * m.point.x; y += ay * m.point.y
        }
        return ContactMeasurement(point: Vec2(x / wx, y / wy), sigma: Vec2(1 / wx.squareRoot(), 1 / wy.squareRoot()),
                                  t: first.t, imagePoint: first.imagePoint)
    }
}

/// Rally state machine turning ball events (+ player feet) into referee calls.
final class RuleEngine {
    struct FootSample { var t: Double; var ground: [Vec2]; var wristsPx: [Vec2] }

    var kitchenEnabled = true
    private(set) var rallyId = -1
    private var bouncesSinceHit = 0
    private var lastBounceHalf: Court.Half?
    private var shotIndex = 0             // 0 = serve, 1 = return, ...
    private var serverHalf: Court.Half?
    private var serverSide: Court.Side?
    private var expectingServe = true

    func startRally(id: Int, firstBallCourt: Vec2?) {
        rallyId = id
        bouncesSinceHit = 0
        lastBounceHalf = nil
        shotIndex = 0
        expectingServe = true
        if let p = firstBallCourt {
            serverHalf = Court.half(of: p)
            serverSide = Court.side(of: p)
        } else {
            serverHalf = nil; serverSide = nil
        }
    }

    /// Process one event; returns calls it produced.
    func handle(_ e: BallEvent, contact: ContactMeasurement?, feet: [FootSample]) -> [RefCall] {
        var calls: [RefCall] = []
        switch e.kind {
        case .hit:
            if kitchenEnabled, shotIndex >= 2, bouncesSinceHit == 0, let call = kitchenCheck(e, feet: feet) {
                calls.append(call)
            }
            shotIndex += 1
            bouncesSinceHit = 0
            expectingServe = false
        case .bounce:
            guard let c = contact else { break }
            let half = Court.half(of: c.point)
            if bouncesSinceHit == 0 {
                if expectingServe && shotIndex == 0 {
                    calls.append(serveCall(c, event: e))
                    expectingServe = false
                } else {
                    let r = LineJudge.judge(c, region: Court.fullCourt)
                    calls.append(make(.line, r, c, e, rule: "8.B / 8.D", detail: lineDetail(c)))
                }
            } else if bouncesSinceHit >= 1, lastBounceHalf == half, Court.fullCourt.contains(c.point) {
                // Second bounce on the same side before any hit.
                calls.append(RefCall(kind: .doubleBounce, verdict: .fault, t: e.t, courtX: c.point.x, courtY: c.point.y,
                                     marginCm: 0, sigmaCm: 0, confidence: 0.8, rule: "10.B.1",
                                     detail: "Ball bounced twice on the \(half == .near ? "near" : "far") side",
                                     imageX: e.imageX, imageY: e.imageY, rallyId: rallyId))
            }
            bouncesSinceHit += 1
            lastBounceHalf = half
            if shotIndex == 0 { shotIndex = 1 }
        }
        return calls
    }

    private func serveCall(_ c: ContactMeasurement, event e: BallEvent) -> RefCall {
        // Serve must land diagonally in the opposite half; infer the server from the first ball position
        // or, failing that, from travel direction and landing side.
        let landingHalf = Court.half(of: c.point)
        let targetHalf: Court.Half = serverHalf.map { (h: Court.Half) -> Court.Half in h == .near ? .far : .near } ?? landingHalf
        let targetSide: Court.Side
        if let s = serverSide {
            targetSide = s == .left ? .right : .left
        } else {
            targetSide = Court.side(of: c.point)
        }
        let region = Court.serviceCourt(half: targetHalf, side: targetSide)
        let r = LineJudge.judge(c, region: region)
        var detail = "Service court \(targetHalf == .near ? "near" : "far")-\(targetSide == .left ? "left" : "right")"
        let kitchenLine = targetHalf == .near ? Court.net - Court.nvzDepth : Court.net + Court.nvzDepth
        if abs(c.point.y - kitchenLine) < 0.5 { detail += " · NVZ line counts OUT on serve" }
        return make(.serve, r, c, e, rule: "4.A / 7 / 8.B", detail: detail)
    }

    private func lineDetail(_ c: ContactMeasurement) -> String {
        let (m, n) = Court.fullCourt.signedMargin(c.point)
        let edge: String
        if abs(n.x) > abs(n.y) { edge = "sideline" } else { edge = "baseline" }
        return String(format: "%@ · %.1f cm %@", edge, abs(m + LineJudge.contactPatchRadius) * Court.feetToCm,
                      m + LineJudge.contactPatchRadius >= 0 ? "inside" : "outside")
    }

    private func make(_ kind: RefCall.Kind, _ r: (verdict: RefCall.Verdict, marginCm: Double, sigmaCm: Double, confidence: Double),
                      _ c: ContactMeasurement, _ e: BallEvent, rule: String, detail: String) -> RefCall {
        RefCall(kind: kind, verdict: r.verdict, t: e.t, courtX: c.point.x, courtY: c.point.y, marginCm: r.marginCm,
                sigmaCm: r.sigmaCm, confidence: r.confidence, rule: rule, detail: detail,
                imageX: e.imageX, imageY: e.imageY, rallyId: rallyId)
    }

    /// Volley: hitter's feet must be outside the NVZ at contact and until balance is regained (11.A.1–11.A.2).
    private func kitchenCheck(_ e: BallEvent, feet: [FootSample]) -> RefCall? {
        let window = feet.filter { $0.t >= e.t - 0.12 && $0.t <= e.t + 0.8 }
        guard !window.isEmpty else { return nil }
        // Hitter = player whose wrist is closest to the ball at contact.
        guard let atHit = window.min(by: { abs($0.t - e.t) < abs($1.t - e.t) }), !atHit.wristsPx.isEmpty else { return nil }
        let wristIdx = atHit.wristsPx.indices.min { (atHit.wristsPx[$0] - e.imagePoint).length < (atHit.wristsPx[$1] - e.imagePoint).length }!
        guard (atHit.wristsPx[wristIdx] - e.imagePoint).length < 260 else { return nil }
        let hitterFeet = atHit.ground.indices.filter { $0 / 2 == wristIdx / 2 }.map { atHit.ground[$0] }
        guard let anchor = hitterFeet.first else { return nil }
        let half = Court.half(of: anchor)
        let zone = Court.kitchen(half)
        var track = anchor
        var deepest = -Double.infinity
        var when = e.t
        for s in window.sorted(by: { $0.t < $1.t }) {
            // Follow the same player by nearest feet.
            guard let near = s.ground.min(by: { ($0 - track).length < ($1 - track).length }), (near - track).length < 3.0 else { continue }
            track = near
            let feetHere = s.ground.filter { ($0 - track).length < 2.0 }
            for f in feetHere {
                let m = zone.signedMargin(f).margin
                if m > deepest { deepest = m; when = s.t }
            }
        }
        let footUncertainty = 0.35   // ft: toe/heel extent around the ankle projection
        guard deepest > -footUncertainty else { return nil }
        let margin = deepest * Court.feetToCm
        let sigma = footUncertainty * Court.feetToCm
        let z = (deepest + 0.0) / footUncertainty
        let verdict: RefCall.Verdict = deepest > 0.15 ? .fault : .tooClose
        let momentum = when > e.t + 0.05
        return RefCall(kind: .kitchen, verdict: verdict, t: e.t, courtX: track.x, courtY: track.y, marginCm: margin,
                       sigmaCm: sigma, confidence: Stats.phi(z), rule: momentum ? "11.A.2 (momentum)" : "11.A.1",
                       detail: momentum ? "Stepped into the kitchen after the volley" : "Foot in the kitchen while volleying",
                       imageX: e.imageX, imageY: e.imageY, rallyId: rallyId)
    }
}
