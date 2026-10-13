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
    private lazy var thumbRenderer = BufferRenderer(width: 640, height: 360, context: ciContext)
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

    /// Best format for the lens: 1080p (or 720p) at the requested frame rate, falling back to lower rates
    /// (ultra-wide lenses often top out at 60 fps). Continuous auto-exposure with a shutter cap.
    private func configureFormat(_ d: AVCaptureDevice) {
        let want = settings.targetFps
        func ok(_ f: AVCaptureDevice.Format, _ w: Int32, _ fps: Double) -> Bool {
            let dims = CMVideoFormatDescriptionGetDimensions(f.formatDescription)
            let sub = CMFormatDescriptionGetMediaSubType(f.formatDescription)
            return dims.width == w && sub == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange &&
                f.videoSupportedFrameRateRanges.contains { $0.maxFrameRate >= fps - 0.5 }
        }
        var chosen: AVCaptureDevice.Format?
        var fps = want
        for candidateFps in [want, 120, 60, 30] where candidateFps <= want {
            if let f = d.formats.first(where: { ok($0, 1920, candidateFps) }) ?? d.formats.first(where: { ok($0, 1280, candidateFps) }) {
                chosen = f; fps = candidateFps; break
            }
        }
        guard let format = chosen else { return }
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
        let lensName = d.deviceType == .builtInUltraWideCamera ? "0.5x" : "1x"
        let info = String(format: "%@ · %dp%.0f · %.0f° FOV", lensName, Int(dims.height), fps, hfov)
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
            if let small = thumbRenderer.render(pb) {
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

/// Requires several consecutive agreeing calibrations before locking (robust to players crossing lines).
final class AutoCalibration {
    enum State { case searching(String), locked(CalibrationResult) }
    private var recent: [CalibrationResult] = []
    private var failures = 0

    func reset() { recent.removeAll(); failures = 0 }

    func add(_ r: CalibrationResult?) -> State {
        guard let r = r else {
            failures += 1
            recent.removeAll()
            return .searching(failures > 8
                ? "Court not found — mount phone 6+ ft high behind a baseline or beside the kitchen, all lines in view"
                : "Searching for court lines…")
        }
        if let last = recent.last, Self.cornerDistance(last.courtToImage, r.courtToImage) > 8 { recent.removeAll() }
        recent.append(r)
        if recent.count >= 3 {
            let best = recent.max { $0.inlierRatio < $1.inlierRatio }!
            recent.removeAll()
            return .locked(best)
        }
        return .searching("Court found — verifying (\(recent.count)/3)…")
    }

    /// Max reprojection distance of court corners between two calibrations, modulo court symmetry.
    static func cornerDistance(_ a: Homography, _ b: Homography) -> Double {
        let flips: [Homography] = [
            .identity,
            Homography(rows: [[-1, 0, Court.width], [0, 1, 0], [0, 0, 1]]),
            Homography(rows: [[1, 0, 0], [0, -1, Court.length], [0, 0, 1]]),
            Homography(rows: [[-1, 0, Court.width], [0, -1, Court.length], [0, 0, 1]]),
        ]
        let pts = Court.corners + [Vec2(Court.width / 2, Court.net)]
        return flips.map { f in
            let bb = f.then(b)
            return pts.map { (a.apply($0) - bb.apply($0)).length }.max() ?? .infinity
        }.min() ?? .infinity
    }

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
