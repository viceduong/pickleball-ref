import CoreGraphics
import Foundation
import simd

/// Locked calibration in full-resolution capture pixels plus derived camera pose.
struct CourtGeometry {
    enum ViewKind: String { case baseline = "Baseline view", sideline = "Sideline view", corner = "Corner view" }

    let courtToImage: Homography
    let imageToCourt: Homography
    let pose: CameraPose?
    let imageSize: Vec2
    let calibrationRmsPx: Double
    let inlierRatio: Double

    init(courtToImage h: Homography, imageSize: Vec2, hfovDegrees: Double?, rmsPx: Double, inlierRatio: Double) {
        let f = CameraPose.focal(imageWidth: imageSize.x, hfovDegrees: hfovDegrees ?? 69)
        let principal = imageSize / 2
        var hh = h
        var pose = CameraPose.from(homography: hh, f: f, principal: principal)
        // Canonicalise the court frame: camera on the near (y < 22) half, and, for sideline views, on x < 10.
        if let c = pose?.center {
            var flipX = false, flipY = false
            if c.y > Court.net { flipY = true }
            let sideline = c.y > -2 && c.y < Court.length + 2 && (c.x < -2 || c.x > Court.width + 2)
            if sideline && c.x > Court.width / 2 { flipX = true }
            if flipX || flipY {
                let fl = Homography(rows: [[flipX ? -1 : 1, 0, flipX ? Court.width : 0],
                                           [0, flipY ? -1 : 1, flipY ? Court.length : 0],
                                           [0, 0, 1]])
                hh = fl.then(hh)
                pose = CameraPose.from(homography: hh, f: f, principal: principal)
            }
        }
        courtToImage = hh
        imageToCourt = hh.inverse
        self.pose = pose
        self.imageSize = imageSize
        calibrationRmsPx = rmsPx
        self.inlierRatio = inlierRatio
    }

    var cameraCenter: Vec3? { pose?.center }

    var view: ViewKind {
        guard let c = cameraCenter else { return .baseline }
        let behindBaseline = c.y < -1
        let besideSideline = c.x < -1 || c.x > Court.width + 1
        if behindBaseline && besideSideline { return .corner }
        return besideSideline ? .sideline : .baseline
    }

    func toImage(_ p: Vec2) -> Vec2 { courtToImage.apply(p) }
    func toCourt(_ p: Vec2) -> Vec2 { imageToCourt.apply(p) }

    func project(_ p: Vec3) -> Vec2? {
        if let pose = pose { return pose.project(p) }
        return p.z == 0 ? toImage(Vec2(p.x, p.y)) : nil
    }

    /// Ground contact point for a ball whose centre is imaged at `centerPx` while touching the ground.
    func groundContact(ballCenterPx: Vec2) -> Vec2 {
        var g = toCourt(ballCenterPx)
        guard pose != nil else { return g }
        for _ in 0..<3 {
            guard let top = project(Vec3(g.x, g.y, Court.ballRadius)), let bottom = project(Vec3(g.x, g.y, 0)) else { break }
            g = toCourt(ballCenterPx - (top - bottom))
        }
        return g
    }

    /// Ground point beneath an ankle keypoint (ankle ≈ 3 in above the floor).
    func footGround(anklePx: Vec2) -> Vec2 {
        var g = toCourt(anklePx)
        guard pose != nil else { return g }
        for _ in 0..<3 {
            guard let top = project(Vec3(g.x, g.y, 0.25)), let bottom = project(Vec3(g.x, g.y, 0)) else { break }
            g = toCourt(anklePx - (top - bottom))
        }
        return g
    }

    /// Expected ball radius in pixels at a court location.
    func ballRadiusPx(at g: Vec2) -> Double {
        guard let a = project(Vec3(g.x, g.y, Court.ballRadius)),
              let b = project(Vec3(g.x, g.y, Court.ballRadius * 3)) else { return 4 }
        return max(1, (a - b).length / 2)
    }

    /// 1-σ ground uncertainty (feet) along unit direction `u` for isotropic pixel noise `sigmaPx` at image point.
    func groundSigma(atPx p: Vec2, along u: Vec2, sigmaPx: Double) -> Double {
        let (a, b, c, d) = imageToCourt.jacobian(at: p)
        // Σg = σ² J Jᵀ ; σ_u² = uᵀ Σg u
        let jtu = Vec2(a * u.x + c * u.y, b * u.x + d * u.y)
        return sigmaPx * jtu.length
    }

    /// Image-space bounding box of the court and the air above it (up to `height` ft), clamped to the image.
    func regionOfInterest(height: Double = 14, margin: Double = 4) -> CGRect {
        var pts: [Vec2] = []
        for c in [Vec2(-margin, -margin), Vec2(Court.width + margin, -margin),
                  Vec2(Court.width + margin, Court.length + margin), Vec2(-margin, Court.length + margin)] {
            if let p = project(Vec3(c.x, c.y, 0)) { pts.append(p) }
            if let p = project(Vec3(c.x, c.y, height)) { pts.append(p) }
        }
        if pts.isEmpty { return CGRect(x: 0, y: 0, width: imageSize.x, height: imageSize.y) }
        let minX = max(0, pts.map(\.x).min()!), maxX = min(imageSize.x, pts.map(\.x).max()!)
        let minY = max(0, pts.map(\.y).min()!), maxY = min(imageSize.y, pts.map(\.y).max()!)
        if maxX <= minX || maxY <= minY { return CGRect(x: 0, y: 0, width: imageSize.x, height: imageSize.y) }
        return CGRect(x: minX, y: minY, width: maxX - minX, height: maxY - minY)
    }

    /// Setup advice derived from the recovered camera pose.
    var setupAdvice: String? {
        guard let c = cameraCenter else { return nil }
        if c.z < 4.5 { return String(format: "Camera is %.1f ft high — raise to 6+ ft for better far-court accuracy", c.z) }
        return nil
    }
}
