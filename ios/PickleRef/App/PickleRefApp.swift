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

    final class PreviewView: UIView {
        override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
        var previewLayer: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }
    }

    func makeUIView(context: Context) -> PreviewView {
        let v = PreviewView()
        v.previewLayer.session = session
        v.previewLayer.videoGravity = .resizeAspect
        v.backgroundColor = .black
        return v
    }

    func updateUIView(_ v: PreviewView, context: Context) {
        if let c = v.previewLayer.connection, c.isVideoOrientationSupported { c.videoOrientation = .landscapeRight }
    }
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
