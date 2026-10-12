import simd
import XCTest
@testable import PickleRef

final class GeometryTests: XCTestCase {
    func testHomographyFitRoundTrip() {
        let truth = Homography(rows: [[30, 4, 200], [-2, 12, 500], [0.001, 0.02, 1]])
        let src = [Vec2(0, 0), Vec2(20, 0), Vec2(20, 44), Vec2(0, 44), Vec2(10, 15), Vec2(10, 29)]
        let dst = src.map { truth.apply($0) }
        let h = Homography.fit(from: src, to: dst)!
        for p in [Vec2(3, 7), Vec2(17, 40), Vec2(10, 22)] {
            XCTAssertEqual((h.apply(p) - truth.apply(p)).length, 0, accuracy: 1e-6)
            XCTAssertEqual((h.inverse.apply(h.apply(p)) - p).length, 0, accuracy: 1e-6)
        }
    }

    func testPoseRecoveryFromHomography() {
        let cam = Synthetic.camera(eye: Vec3(10, -15, 9), target: Vec3(10, 26, 0), width: 1920, height: 1080)
        let pose = CameraPose.from(homography: cam.groundHomography, f: cam.f, principal: cam.principal)!
        XCTAssertEqual(pose.center.x, 10, accuracy: 1e-6)
        XCTAssertEqual(pose.center.y, -15, accuracy: 1e-6)
        XCTAssertEqual(pose.center.z, 9, accuracy: 1e-6)
        let p = Vec3(4, 30, 3)
        XCTAssertEqual((pose.project(p)! - cam.project(p)!).length, 0, accuracy: 1e-6)
    }

    func testGroundContactCorrectsBallRadius() {
        let cam = Synthetic.camera(eye: Vec3(10, -15, 9), target: Vec3(10, 26, 0), width: 1920, height: 1080)
        let g = CourtGeometry(courtToImage: cam.groundHomography, imageSize: Vec2(1920, 1080), hfovDegrees: 69, rmsPx: 0.5, inlierRatio: 1)
        let truth = Vec2(19.9, 43.95)
        let centre = cam.project(Vec3(truth.x, truth.y, Court.ballRadius))!
        let naive = g.toCourt(centre)
        let corrected = g.groundContact(ballCenterPx: centre)
        XCTAssertGreaterThan((naive - truth).length, 0.3, "un-corrected projection is biased")
        XCTAssertEqual((corrected - truth).length, 0, accuracy: 0.01)
    }

    func testCanonicalFrameFlipsCameraToNearHalf() {
        // Camera behind the far baseline: canonical frame should put it at y < 0.
        let cam = Synthetic.camera(eye: Vec3(10, 59, 9), target: Vec3(10, 18, 0), width: 1920, height: 1080)
        let g = CourtGeometry(courtToImage: cam.groundHomography, imageSize: Vec2(1920, 1080), hfovDegrees: 69, rmsPx: 0.5, inlierRatio: 1)
        XCTAssertLessThan(g.cameraCenter!.y, 0)
        XCTAssertEqual(g.cameraCenter!.z, 9, accuracy: 1e-6)
        XCTAssertEqual(g.view, .baseline)
    }
}

final class CalibrationTests: XCTestCase {
    private func run(eye: Vec3, target: Vec3, seed: UInt64) throws {
        let w = 960, h = 540
        let cam = Synthetic.camera(eye: eye, target: target, width: w, height: h)
        let img = Synthetic.renderCourt(cam, width: w, height: h, seed: seed)
        let result = try XCTUnwrap(CourtCalibrator().calibrate(img), "court not found")
        let err = AutoCalibration.cornerDistance(cam.groundHomography, result.courtToImage)
        XCTAssertLessThan(err, 3.0, "corner reprojection error \(err) px")
        XCTAssertGreaterThan(result.inlierRatio, 0.6)
    }

    func testAutoCalibrationBaselineView() throws {
        try run(eye: Vec3(10, -16, 10), target: Vec3(10, 24, 0), seed: 3)
    }

    func testAutoCalibrationCornerView() throws {
        try run(eye: Vec3(-6, -14, 9), target: Vec3(11, 22, 0), seed: 5)
    }

    func testAutoCalibrationSidelineView() throws {
        try run(eye: Vec3(-22, 22, 11), target: Vec3(10, 22, 0), seed: 9)
    }
}

final class EventTests: XCTestCase {
    private let cam = Synthetic.camera(eye: Vec3(10, -15, 9), target: Vec3(10, 26, 0), width: 1920, height: 1080)
    private var geometry: CourtGeometry {
        CourtGeometry(courtToImage: cam.groundHomography, imageSize: Vec2(1920, 1080), hfovDegrees: 69, rmsPx: 0.5, inlierRatio: 1)
    }

    func testBounceDetectedAndLocalised() throws {
        // Drive from the near half landing deep near the far sideline.
        let sim = Synthetic.flight(cam: cam, start: Vec3(6, 8, 3), velocity: Vec3(5.5, 52, 6), fps: 120, duration: 1.3, noisePx: 0.5)
        let truth = try XCTUnwrap(sim.bounce)
        let det = EventDetector()
        det.sidePoints = 5
        let events = det.detect(sim.pts, geometry: geometry)
        let bounce = try XCTUnwrap(events.first { $0.kind == .bounce }, "events: \(events.map { ($0.kind, $0.t) })")
        XCTAssertEqual(bounce.t, sim.bounceT!, accuracy: 0.01)
        let contact = geometry.groundContact(ballCenterPx: bounce.imagePoint)
        let errCm = (contact - truth).length * Court.feetToCm
        XCTAssertLessThan(errCm, 12, "contact error \(errCm) cm (truth \(truth), got \(contact))")
    }

    func testHitClassifiedAsHit() throws {
        let sim = Synthetic.flight(cam: cam, start: Vec3(10, 6, 3), velocity: Vec3(0, 40, 8), fps: 120, duration: 1.2,
                                   noisePx: 0.4, hitAt: 0.55)
        let events = EventDetector().detect(sim.pts.filter { $0.t < (sim.bounceT ?? 9) - 0.01 }, geometry: geometry)
        let hit = try XCTUnwrap(events.first { abs($0.t - 0.55) < 0.03 }, "events: \(events.map { ($0.kind, $0.t) })")
        XCTAssertEqual(hit.kind, .hit)
    }

    func testTrajectory3DRecoversArc() throws {
        let sim = Synthetic.flight(cam: cam, start: Vec3(8, 5, 3), velocity: Vec3(2, 45, 10), fps: 120, duration: 0.5, noisePx: 0.3)
        let pts = sim.pts.filter { $0.t < (sim.bounceT ?? 1) - 0.02 }
        let arc = try XCTUnwrap(Trajectory3D.fit(pts, pose: geometry.pose!))
        XCTAssertLessThan(arc.rmsPx, 1.5)
        XCTAssertEqual(arc.position(0).z, 3, accuracy: 0.5)
        XCTAssertEqual(arc.position(0).y, 5, accuracy: 1.0)
    }
}

final class TrackerTests: XCTestCase {
    func testTrackerFollowsBallThroughClutter() {
        let cam = Synthetic.camera(eye: Vec3(10, -15, 9), target: Vec3(10, 26, 0), width: 1920, height: 1080)
        let sim = Synthetic.flight(cam: cam, start: Vec3(6, 8, 3), velocity: Vec3(5.5, 52, 6), fps: 120, duration: 1.0, noisePx: 0.5)
        var rng = Synthetic.LCG(s: 42)
        let tracker = BallTracker()
        for p in sim.pts {
            var cands = [BallCandidate(t: p.t, p: p.p, score: 0.8, source: .motion)]
            for _ in 0..<3 {   // random clutter blobs
                cands.append(BallCandidate(t: p.t, p: Vec2(rng.next() * 1920, rng.next() * 1080), score: 0.7, source: .motion))
            }
            tracker.ingest(frameTime: p.t, candidates: cands)
        }
        XCTAssertGreaterThan(tracker.rally.count, Int(Double(sim.pts.count) * 0.85))
        let truth = Dictionary(uniqueKeysWithValues: sim.pts.map { (Int(($0.t * 1e6).rounded()), $0.p) })
        let wrong = tracker.rally.filter { tp in (truth[Int((tp.t * 1e6).rounded())].map { ($0 - tp.p).length } ?? 99) > 5 }
        XCTAssertLessThan(wrong.count, 3)
    }
}

final class RuleTests: XCTestCase {
    private func m(_ x: Double, _ y: Double, sigma: Double = 0.03) -> ContactMeasurement {
        ContactMeasurement(point: Vec2(x, y), sigma: Vec2(sigma, sigma), t: 0, imagePoint: .zero)
    }

    func testLineVerdicts() {
        XCTAssertEqual(LineJudge.judge(m(10, 43.5), region: Court.fullCourt).verdict, .inBounds)
        XCTAssertEqual(LineJudge.judge(m(10, 44.5), region: Court.fullCourt).verdict, .out)
        // Contact on the outer edge of the baseline: IN by the contact patch → too close given 3 cm sigma.
        XCTAssertEqual(LineJudge.judge(m(10, 44.0), region: Court.fullCourt).verdict, .tooClose)
        // Precise measurement with the ball centre just inside the outer edge: IN.
        XCTAssertEqual(LineJudge.judge(m(10, 43.99, sigma: 0.003), region: Court.fullCourt).verdict, .inBounds)
    }

    func testServeOnKitchenLineIsOut() {
        let svc = Court.serviceCourt(half: .far, side: .right)
        // Middle of the far NVZ line (y 28.83…29.0) is not part of the service court.
        let r = LineJudge.judge(m(15, 28.85, sigma: 0.005), region: svc)
        XCTAssertEqual(r.verdict, .out)
        // Centreline counts IN for both service courts.
        XCTAssertEqual(LineJudge.judge(m(9.96, 36, sigma: 0.005), region: svc).verdict, .inBounds)
    }

    func testFusionReducesUncertainty() {
        let a = ContactMeasurement(point: Vec2(10, 44.1), sigma: Vec2(0.05, 0.2), t: 0, imagePoint: .zero)
        let b = ContactMeasurement(point: Vec2(10.02, 44.02), sigma: Vec2(0.2, 0.04), t: 0, imagePoint: .zero)
        let f = LineJudge.fuse([a, b])!
        XCTAssertLessThan(f.sigma.y, 0.04)
        XCTAssertEqual(f.point.y, 44.02, accuracy: 0.01)
    }

    func testRallyServeThenLineCall() {
        let rules = RuleEngine()
        rules.startRally(id: 1, firstBallCourt: Vec2(15, -1))   // server near-right
        let e = BallEvent(kind: .bounce, t: 1, imageX: 0, imageY: 0, residualPx: 0.5, travelY: 1, height3D: nil)
        let serve = rules.handle(e, contact: m(5, 40, sigma: 0.02), feet: [])
        XCTAssertEqual(serve.first?.kind, .serve)
        XCTAssertEqual(serve.first?.verdict, .inBounds)
        _ = rules.handle(BallEvent(kind: .hit, t: 1.5, imageX: 0, imageY: 0, residualPx: 0.5, travelY: -1, height3D: nil), contact: nil, feet: [])
        let line = rules.handle(BallEvent(kind: .bounce, t: 2, imageX: 0, imageY: 0, residualPx: 0.5, travelY: -1, height3D: nil),
                                contact: m(20.4, 10, sigma: 0.02), feet: [])
        XCTAssertEqual(line.first?.kind, .line)
        XCTAssertEqual(line.first?.verdict, .out)
    }

    func testDoubleBounce() {
        let rules = RuleEngine()
        rules.startRally(id: 1, firstBallCourt: Vec2(5, -1))
        _ = rules.handle(BallEvent(kind: .bounce, t: 1, imageX: 0, imageY: 0, residualPx: 0.5, travelY: 1, height3D: nil), contact: m(15, 38), feet: [])
        let calls = rules.handle(BallEvent(kind: .bounce, t: 1.6, imageX: 0, imageY: 0, residualPx: 0.5, travelY: 1, height3D: nil), contact: m(15, 41), feet: [])
        XCTAssertEqual(calls.first?.kind, .doubleBounce)
    }

    func testKitchenVolleyFault() {
        let rules = RuleEngine()
        rules.startRally(id: 1, firstBallCourt: Vec2(5, -1))
        let bounce = { (t: Double, _: Double) in BallEvent(kind: .bounce, t: t, imageX: 0, imageY: 0, residualPx: 0.5, travelY: 1, height3D: nil) }
        let hit = { (t: Double) in BallEvent(kind: .hit, t: t, imageX: 500, imageY: 300, residualPx: 0.5, travelY: 1, height3D: nil) }
        _ = rules.handle(bounce(1, 38), contact: m(15, 38), feet: [])         // serve lands
        _ = rules.handle(hit(1.5), contact: nil, feet: [])                      // return
        _ = rules.handle(bounce(2, 10), contact: m(8, 10), feet: [])            // return bounces
        _ = rules.handle(hit(2.5), contact: nil, feet: [])                      // third shot
        // Fourth shot is a volley by a player standing 0.5 ft inside the far kitchen.
        let feet = [RuleEngine.FootSample(t: 3.0, ground: [Vec2(10, 28.5), Vec2(11, 28.4)], wristsPx: [Vec2(505, 300), Vec2(520, 310)])]
        let calls = rules.handle(hit(3.0), contact: nil, feet: feet)
        XCTAssertEqual(calls.first?.kind, .kitchen)
        XCTAssertEqual(calls.first?.verdict, .fault)
    }
}

final class ClockSyncTests: XCTestCase {
    func testOffsetFromNoisyExchanges() {
        var sync = ClockSync()
        var rng = Synthetic.LCG(s: 11)
        let trueOffset = 12.345
        var t = 100.0
        for _ in 0..<60 {
            let up = 0.002 + rng.next() * 0.05, down = 0.002 + rng.next() * 0.05, proc = 0.0005
            let t0 = t, t1 = t0 + up + trueOffset, t2 = t1 + proc, t3 = t0 + up + proc + down
            sync.add(.init(t0: t0, t1: t1, t2: t2, t3: t3))
            t += 0.25
        }
        XCTAssertEqual(sync.offset!, trueOffset, accuracy: 0.004)
        XCTAssertEqual(sync.toLocal(remoteTime: 200 + trueOffset)!, 200, accuracy: 0.004)
    }
}
