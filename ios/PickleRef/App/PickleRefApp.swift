import AVFoundation
import SwiftUI

@main
struct PickleRefApp: App {
    @StateObject private var model = AppModel()

    var body: some Scene {
        WindowGroup {
            LiveView()
                .environmentObject(model)
                .preferredColorScheme(.dark)
                .statusBar(hidden: true)
                .onAppear { model.start() }
        }
    }
}

struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession

    /// The preview connection only exists once the session has its input (configured asynchronously),
    /// so the landscape orientation is re-applied on layout and whenever the session starts running.
    final class PreviewView: UIView {
        override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
        var previewLayer: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }
        private var observer: NSObjectProtocol?

        func attach(_ session: AVCaptureSession) {
            previewLayer.session = session
            observer = NotificationCenter.default.addObserver(forName: .AVCaptureSessionDidStartRunning, object: session,
                                                              queue: .main) { [weak self] _ in self?.applyOrientation() }
        }

        func applyOrientation() {
            if let c = previewLayer.connection, c.isVideoOrientationSupported, c.videoOrientation != .landscapeRight {
                c.videoOrientation = .landscapeRight
            }
        }

        override func layoutSubviews() {
            super.layoutSubviews()
            applyOrientation()
        }

        deinit { if let o = observer { NotificationCenter.default.removeObserver(o) } }
    }

    func makeUIView(context: Context) -> PreviewView {
        let v = PreviewView()
        v.attach(session)
        v.previewLayer.videoGravity = .resizeAspect
        v.backgroundColor = .black
        return v
    }

    func updateUIView(_ v: PreviewView, context: Context) { v.applyOrientation() }
}

/// Maps capture pixels into an aspect-fit view rect.
struct ImageMapper {
    let image: Vec2
    let view: CGSize

    var scale: Double { min(Double(view.width) / image.x, Double(view.height) / image.y) }
    var origin: CGPoint {
        CGPoint(x: (Double(view.width) - image.x * scale) / 2, y: (Double(view.height) - image.y * scale) / 2)
    }

    func map(_ p: Vec2) -> CGPoint { CGPoint(x: Double(origin.x) + p.x * scale, y: Double(origin.y) + p.y * scale) }
}

extension RefCall.Verdict {
    var color: Color {
        switch self {
        case .inBounds, .clean: return .green
        case .out, .fault: return .red
        case .tooClose: return .yellow
        }
    }
}
