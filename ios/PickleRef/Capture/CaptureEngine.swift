import AVFoundation
import CoreImage
import Foundation
import ImageIO
import UIKit

enum CameraLens: String, CaseIterable, Identifiable {
    case ultraWide = "0.5x Ultra-wide"
    case wide = "1x Wide"
    var id: String { rawValue }
    var deviceType: AVCaptureDevice.DeviceType { self == .ultraWide ? .builtInUltraWideCamera : .builtInWideAngleCamera }
}

struct CaptureSettings {
    var targetFps: Double = 120
    /// Ultra-wide covers the whole court from a tripod close to it.
    var lens: CameraLens = .ultraWide
    /// Auto-exposure shutter cap (s); 0 = uncapped. Short shutters keep the ball sharp, AE raises ISO instead.
    var maxShutter: Double = 1.0 / 500
    var useNeural = true
    var useVisionTrajectories = true
    var usePose = true
}

/// Owns the AVCaptureSession and routes every frame to the detectors, calibrator and engine.
final class CaptureEngine: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    let session = AVCaptureSession()
    private let sessionQueue = DispatchQueue(label: "capture.session")
    private let videoQueue = DispatchQueue(label: "capture.video", qos: .userInteractive)
    private let calibQueue = DispatchQueue(label: "capture.calibration", qos: .utility)
    private let visionQueue = DispatchQueue(label: "capture.vision", qos: .userInitiated)
    private let poseQueue = DispatchQueue(label: "capture.pose", qos: .userInitiated)
    private let nnQueue = DispatchQueue(label: "capture.nn", qos: .userInitiated)
    private let thumbQueue = DispatchQueue(label: "capture.thumbs", qos: .utility)

    private var device: AVCaptureDevice?
    private let output = AVCaptureVideoDataOutput()

    let engine: RefereeEngine
    let ring = FrameRing()
    var settings = CaptureSettings()

    // Callbacks (main thread).
    var onCalibration: ((CourtGeometry?, String) -> Void)?
    var onStats: ((Double, Bool) -> Void)?
    var onFormat: ((String) -> Void)?

    private let ciContext = CIContext(options: [.cacheIntermediates: false])
    private var thumbRenderer: BufferRenderer?
    private lazy var nnRenderer = BufferRenderer(width: NeuralBallDetector.inputSize.w, height: NeuralBallDetector.inputSize.h, context: ciContext)
    private let motion = MotionBlobDetector()
    private let trajectories = VisionTrajectoryDetector()
    private let pose = PoseDetector()
    private lazy var neural: NeuralBallDetector? = NeuralBallDetector()
    private let calibrator = CourtCalibrator()
    private let autoCal = AutoCalibration()

    private var geometry: CourtGeometry?
    private var hfov: Double = 69
    private var imageSize = Vec2(1920, 1080)
    private var hostOffset: Double = 0
    private var calibBusy = false, visionBusy = false, poseBusy = false, nnBusy = false
    private var lastCalib = 0.0, lastPose = 0.0, lastThumb = 0.0, lastVerify = 0.0
    private var nnFrames: [(Double, CVPixelBuffer)] = []
    private var nnCrop = CGRect.zero
    private var playerBoxes: [CGRect] = []
    private var frameCount = 0
    private var fpsWindowStart = 0.0
    private(set) var measuredFps = 0.0
    private var deviceFps = 120.0

    init(engine: RefereeEngine) {
        self.engine = engine
        super.init()
    }

    var hasNeuralModel: Bool { NeuralBallDetector.isBundled }

    // MARK: Session

    func start() {
        sessionQueue.async {
            if !self.session.isRunning {
                if self.session.inputs.isEmpty { self.configure() }
                self.session.startRunning()
            }
        }
    }

    func stop() { sessionQueue.async { self.session.stopRunning() } }

    func recalibrate() {
        videoQueue.async {
            self.geometry = nil
            self.autoCal.reset()
            self.motion.reset()
            self.trajectories.reset()
            self.engine.setGeometry(nil)
            DispatchQueue.main.async { self.onCalibration?(nil, "Searching for court lines…") }
        }
    }

    func apply(_ s: CaptureSettings) {
        sessionQueue.async {
            let old = self.settings
            self.settings = s
            if s.lens != old.lens {
                self.switchInput()
                self.recalibrate()
            } else if s.targetFps != old.targetFps || s.maxShutter != old.maxShutter, let d = self.device {
                self.configureFormat(d)
            }
        }
    }

    /// Back camera for the selected lens, falling back to the 1x wide camera (ultra-wide needs iPhone 11+).
    private func selectDevice() -> AVCaptureDevice? {
        AVCaptureDevice.default(settings.lens.deviceType, for: .video, position: .back)
            ?? AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back)
    }

    private func configure() {
        session.beginConfiguration()
        session.sessionPreset = .inputPriority
        guard let d = selectDevice(), let input = try? AVCaptureDeviceInput(device: d), session.canAddInput(input) else {
            session.commitConfiguration(); return
        }
        session.addInput(input)
        device = d
        output.alwaysDiscardsLateVideoFrames = true
        output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr8BiPlanarFullRange]
        output.setSampleBufferDelegate(self, queue: videoQueue)
        if session.canAddOutput(output) { session.addOutput(output) }
        configureConnection()
        session.commitConfiguration()
        configureFormat(d)
    }

    private func switchInput() {
        guard let d = selectDevice(), d != device, let input = try? AVCaptureDeviceInput(device: d) else { return }
        session.beginConfiguration()
        for i in session.inputs { session.removeInput(i) }
        if session.canAddInput(input) { session.addInput(input); device = d }
        configureConnection()
        session.commitConfiguration()
        if let dev = device { configureFormat(dev) }
    }

    private func configureConnection() {
        if let c = output.connection(with: .video) {
            if c.isVideoOrientationSupported { c.videoOrientation = .landscapeRight }
            if c.isVideoStabilizationSupported { c.preferredVideoStabilizationMode = .off }
        }
    }

    /// Format with the widest vertical field of view (court depth runs vertically in landscape): full-sensor 4:3
    /// formats beat 16:9 crops. Among equals, the highest frame rate up to the target, then ~1080p resolution.
    private func configureFormat(_ d: AVCaptureDevice) {
        let want = settings.targetFps
        struct Cand { let f: AVCaptureDevice.Format; let vfov: Double; let fps: Double; let width: Int32 }
        var cands: [Cand] = []
        for f in d.formats {
            let dims = CMVideoFormatDescriptionGetDimensions(f.formatDescription)
            guard CMFormatDescriptionGetMediaSubType(f.formatDescription) == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
                  dims.width >= 1280, dims.width <= 2112 else { continue }
            let maxFps = f.videoSupportedFrameRateRanges.map(\.maxFrameRate).max() ?? 0
            let fps = [240.0, 120, 60, 30].first { $0 <= min(want, maxFps) + 0.5 } ?? 30
            let hf = Double(f.videoFieldOfView) * .pi / 180
            let vfov = 2 * atan(tan(hf / 2) * Double(dims.height) / Double(dims.width)) * 180 / .pi
            cands.append(Cand(f: f, vfov: vfov, fps: fps, width: dims.width))
        }
        // Prefer ≥60 fps (bounce timing), then the widest vertical view, then frame rate, then resolution near 1920.
        let pool = cands.contains { $0.fps >= 60 } ? cands.filter { $0.fps >= 60 } : cands
        guard let best = pool.max(by: { a, b in
            if abs(a.vfov - b.vfov) > 1 { return a.vfov < b.vfov }
            if a.fps != b.fps { return a.fps < b.fps }
            return abs(Int(a.width) - 1920) > abs(Int(b.width) - 1920)
        }) else { return }
        let format = best.f
        let fps = best.fps
        do {
            try d.lockForConfiguration()
            d.activeFormat = format
            let dur = CMTime(value: 1, timescale: CMTimeScale(fps))
            d.activeVideoMinFrameDuration = dur
            d.activeVideoMaxFrameDuration = dur
            // Rectilinear output so the court homography model holds (matters on the ultra-wide lens).
            if d.isGeometricDistortionCorrectionSupported { d.isGeometricDistortionCorrectionEnabled = true }
            if d.isFocusModeSupported(.continuousAutoFocus) { d.focusMode = .continuousAutoFocus }
            if d.isExposureModeSupported(.continuousAutoExposure) {
                d.exposureMode = .continuousAutoExposure
                // Setting activeFormat resets the cap, so apply it afterwards.
                if settings.maxShutter > 0 {
                    let cap = CMTime(seconds: settings.maxShutter, preferredTimescale: 1_000_000)
                    d.activeMaxExposureDuration = CMTimeClampToRange(cap, range: CMTimeRange(start: format.minExposureDuration,
                                                                                             end: format.maxExposureDuration))
                } else {
                    d.activeMaxExposureDuration = .invalid
                }
            }
            d.unlockForConfiguration()
        } catch { return }
        hfov = Double(format.videoFieldOfView)
        let dims = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
        imageSize = Vec2(Double(dims.width), Double(dims.height))
        deviceFps = fps
        engine.setFps(fps)
        var lensName = d.deviceType == .builtInUltraWideCamera ? "0.5x" : "1x"
        if settings.lens == .ultraWide && d.deviceType != .builtInUltraWideCamera { lensName = "1x (no ultra-wide on this iPhone)" }
        let info = String(format: "%@ · %dx%d@%.0f · %.0f°×%.0f° FOV", lensName, Int(dims.width), Int(dims.height), fps, hfov, best.vfov)
        DispatchQueue.main.async { self.onFormat?(info) }
    }

    // MARK: Per-frame pipeline

    func captureOutput(_ output: AVCaptureOutput, didOutput sb: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pb = CMSampleBufferGetImageBuffer(sb) else { return }
        let pts = CMSampleBufferGetPresentationTimeStamp(sb)
        let clock = session.masterClock ?? CMClockGetHostTimeClock()
        let hostTime = CMSyncConvertTime(pts, from: clock, to: CMClockGetHostTimeClock()).seconds
        hostOffset = hostTime - pts.seconds
        let t = hostTime
        let w = CVPixelBufferGetWidth(pb), h = CVPixelBufferGetHeight(pb)
        imageSize = Vec2(Double(w), Double(h))
        updateFps(t)

        // Replay thumbnails (≤ 60 fps).
        if t - lastThumb >= 1.0 / 61 {
            lastThumb = t
            let thumbH = (640 * h / max(1, w)) & ~1
            if thumbRenderer?.height != thumbH { thumbRenderer = BufferRenderer(width: 640, height: thumbH, context: ciContext) }
            if let small = thumbRenderer?.render(pb) {
                thumbQueue.async {
                    let ci = CIImage(cvPixelBuffer: small)
                    if let data = self.ciContext.jpegRepresentation(of: ci, colorSpace: CGColorSpaceCreateDeviceRGB(),
                                                                    options: [kCGImageDestinationLossyCompressionQuality as CIImageRepresentationOption: 0.55]) {
                        self.ring.append(.init(t: t, jpeg: data))
                    }
                }
                if settings.usePose, geometry != nil, !poseBusy, t - lastPose >= 1.0 / 15 {
                    lastPose = t; poseBusy = true
                    let size = imageSize
                    poseQueue.async {
                        let players = self.pose.detect(small, imageSize: size)
                        self.engine.addPose(t: t, players: players)
                        self.videoQueue.async { self.playerBoxes = players.map(\.box); self.poseBusy = false }
                    }
                }
            }
        }

        guard let g = geometry else {
            calibrateIfDue(pb, t: t)
            return
        }
        verifyIfDue(pb, t: t, g: g)
        let roi = g.regionOfInterest()

        // 1. Motion blobs (every frame, 1/4 resolution).
        let sw = w / 4, sh = h / 4
        if let small = ImageUtils.luma(pb, width: sw, height: sh) {
            let s = 4.0
            let sroi = CGRect(x: roi.minX / s, y: roi.minY / s, width: roi.width / s, height: roi.height / s)
            let nearR = g.ballRadiusPx(at: Vec2(Court.width / 2, 0)) / s
            let maxArea = max(6, Int(Double.pi * nearR * nearR * 5))
            let c = motion.detect(small, t: t, roi: sroi, scale: s, maxArea: maxArea, exclusion: playerBoxes)
            engine.addCandidates(c)
        }

        // 2. Apple Vision trajectories.
        if settings.useVisionTrajectories && !visionBusy {
            visionBusy = true
            let size = imageSize, off = hostOffset
            visionQueue.async {
                let c = self.trajectories.process(sb, roi: roi, imageSize: size, hostOffset: off)
                self.engine.addCandidates(c)
                self.videoQueue.async { self.visionBusy = false }
            }
        }

        // 3. WASB neural detector on frame triplets (~60 fps spacing) cropped to the court ROI; middle-frame heatmap.
        if settings.useNeural, neural != nil {
            if nnCrop == .zero { nnCrop = Self.crop16x9(roi, in: imageSize) }
            if let buf = nnRenderer.render(pb, crop: nnCrop) {
                nnFrames.append((t, buf))
                if nnFrames.count > 5 { nnFrames.removeFirst(nnFrames.count - 5) }
            }
            let step = max(1, Int((deviceFps / 60).rounded()))
            if !nnBusy, nnFrames.count > 2 * step {
                let n = nnFrames.count
                let trip = [nnFrames[n - 1 - 2 * step], nnFrames[n - 1 - step], nnFrames[n - 1]]
                let crop = nnCrop
                nnBusy = true
                nnQueue.async {
                    if let hit = self.neural?.predict(frames: trip.map(\.1)) {
                        let p = hit.0, score = hit.1
                        let sx = crop.width / CGFloat(NeuralBallDetector.inputSize.w)
                        let sy = crop.height / CGFloat(NeuralBallDetector.inputSize.h)
                        let full = Vec2(Double(crop.minX) + p.x * Double(sx), Double(crop.minY) + p.y * Double(sy))
                        self.engine.addCandidates([BallCandidate(t: trip[1].0, p: full, score: score, source: .neural)])
                    }
                    self.videoQueue.async { self.nnBusy = false }
                }
            }
        }

        engine.frame(t)
    }

    private func updateFps(_ t: Double) {
        frameCount += 1
        if t - fpsWindowStart >= 1 {
            measuredFps = Double(frameCount) / (t - fpsWindowStart)
            frameCount = 0
            fpsWindowStart = t
            let fps = measuredFps
            let locked = geometry != nil
            DispatchQueue.main.async { self.onStats?(fps, locked) }
        }
    }

    static func crop16x9(_ r: CGRect, in size: Vec2) -> CGRect {
        var w = r.width, h = r.height
        if w / h > 16.0 / 9 { h = w * 9 / 16 } else { w = h * 16 / 9 }
        w = min(w, CGFloat(size.x)); h = min(h, CGFloat(size.y))
        var x = r.midX - w / 2, y = r.midY - h / 2
        x = max(0, min(CGFloat(size.x) - w, x)); y = max(0, min(CGFloat(size.y) - h, y))
        return CGRect(x: x, y: y, width: w, height: h)
    }

    // MARK: Calibration

    private func calibrateIfDue(_ pb: CVPixelBuffer, t: Double) {
        guard !calibBusy, t - lastCalib > 0.6 else { return }
        let w = CVPixelBufferGetWidth(pb), h = CVPixelBufferGetHeight(pb)
        guard let gray = ImageUtils.luma(pb, width: w / 2, height: h / 2) else { return }
        lastCalib = t
        calibBusy = true
        let fov = hfov, size = imageSize
        calibQueue.async {
            let res = self.calibrator.calibrate(gray)?.scaled(by: 2)
            self.videoQueue.async {
                self.calibBusy = false
                guard self.geometry == nil else { return }
                switch self.autoCal.add(res) {
                case .searching(let msg):
                    DispatchQueue.main.async { self.onCalibration?(nil, msg) }
                case .locked(let r):
                    let g = CourtGeometry(courtToImage: r.courtToImage, imageSize: size, hfovDegrees: fov,
                                          rmsPx: r.rmsPx, inlierRatio: r.inlierRatio)
                    self.geometry = g
                    self.nnCrop = .zero
                    self.lastVerify = t
                    self.engine.setGeometry(g)
                    self.lockFocus()
                    DispatchQueue.main.async { self.onCalibration?(g, "Court locked") }
                }
            }
        }
    }

    /// Periodically re-checks the locked court; recalibrates automatically if the phone was moved.
    private func verifyIfDue(_ pb: CVPixelBuffer, t: Double, g: CourtGeometry) {
        guard !calibBusy, t - lastVerify > 6 else { return }
        let w = CVPixelBufferGetWidth(pb), h = CVPixelBufferGetHeight(pb)
        guard let gray = ImageUtils.luma(pb, width: w / 2, height: h / 2) else { return }
        lastVerify = t
        calibBusy = true
        calibQueue.async {
            let ratio = AutoCalibration.support(of: g.courtToImage.then(.scale(0.5, 0.5)), in: gray)
            self.videoQueue.async {
                self.calibBusy = false
                if ratio < 0.3 * g.inlierRatio { self.recalibrate() }
            }
        }
    }

    private func lockFocus() {
        sessionQueue.async {
            guard let d = self.device, (try? d.lockForConfiguration()) != nil else { return }
            if d.isFocusModeSupported(.locked) { d.focusMode = .locked }
            d.unlockForConfiguration()
        }
    }
}

/// Locks once two calibrations agree on the visible court (or one is unambiguous). Tolerates
/// intermittent failures (players crossing lines) instead of restarting the count.
final class AutoCalibration {
    enum State { case searching(String), locked(CalibrationResult) }
    private var recent: [CalibrationResult] = []
    private var failures = 0
    private var consecutiveFailures = 0

    func reset() { recent.removeAll(); failures = 0; consecutiveFailures = 0 }

    func add(_ r: CalibrationResult?) -> State {
        guard let r = r else {
            failures += 1
            consecutiveFailures += 1
            if consecutiveFailures >= 6 { recent.removeAll() }
            if !recent.isEmpty { return .searching("Court found — verifying…") }
            return .searching(failures > 8
                ? "Court not found — mount phone 6+ ft high behind a baseline or beside the kitchen, lines in view"
                : "Searching for court lines…")
        }
        consecutiveFailures = 0
        // Unambiguous single result: most of the court model is confirmed by line pixels.
        if r.inlierRatio >= 0.85 && r.hits >= 150 { recent.removeAll(); return .locked(r) }
        let tol = max(10, 0.008 * Double(r.imageWidth))
        if let match = recent.last(where: { Self.visibleDistance($0, r) <= tol }) {
            recent.removeAll()
            return .locked(match.inlierRatio > r.inlierRatio ? match : r)
        }
        recent.append(r)
        if recent.count > 4 { recent.removeFirst() }
        return .searching(String(format: "Court found — verifying (%.0f%% lines matched)…", r.inlierRatio * 100))
    }

    /// Max distance between two calibrations over court points visible in `a`'s image, modulo court symmetry.
    static func visibleDistance(_ a: CalibrationResult, _ b: CalibrationResult) -> Double {
        var pts: [Vec2] = []
        for i in 0...8 { for j in 0...22 { pts.append(Vec2(Double(i) * 2.5, Double(j) * 2)) } }
        let w = Double(a.imageWidth), h = Double(a.imageHeight)
        let visible = pts.filter { p in
            let v = a.courtToImage.homogeneous(p)
            guard v.z > 0 else { return false }
            let q = Vec2(v.x / v.z, v.y / v.z)
            return q.x >= 0 && q.y >= 0 && q.x <= w && q.y <= h
        }
        guard visible.count >= 6 else { return .infinity }
        return flips.map { f in
            let bb = f.then(b.courtToImage)
            return visible.map { (a.courtToImage.apply($0) - bb.apply($0)).length }.max() ?? .infinity
        }.min() ?? .infinity
    }

    static let flips: [Homography] = [
        .identity,
        Homography(rows: [[-1, 0, Court.width], [0, 1, 0], [0, 0, 1]]),
        Homography(rows: [[1, 0, 0], [0, -1, Court.length], [0, 0, 1]]),
        Homography(rows: [[-1, 0, Court.width], [0, -1, Court.length], [0, 0, 1]]),
    ]

    /// Fraction of court-line samples supported by line pixels in `img`.
    static func support(of h: Homography, in img: GrayImage) -> Double {
        let mask = LineMask.detect(img).dilated(2)
        var hits = 0, vis = 0
        for s in Court.samples(step: 1) {
            let p = h.apply(s)
            guard p.x >= 0, p.y >= 0, p.x < Double(img.width - 1), p.y < Double(img.height - 1) else { continue }
            vis += 1
            if mask.contains(p) { hits += 1 }
        }
        return vis > 0 ? Double(hits) / Double(vis) : 0
    }
}
