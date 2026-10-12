import Accelerate
import CoreImage
import CoreVideo
import Foundation

enum ImageUtils {
    /// Downscales the luma (Y) plane of a bi-planar YUV buffer to `width`×`height` with vImage.
    static func luma(_ pb: CVPixelBuffer, width: Int, height: Int) -> GrayImage? {
        CVPixelBufferLockBaseAddress(pb, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
        guard CVPixelBufferGetPlaneCount(pb) >= 1, let base = CVPixelBufferGetBaseAddressOfPlane(pb, 0) else { return nil }
        var src = vImage_Buffer(data: base,
                                height: vImagePixelCount(CVPixelBufferGetHeightOfPlane(pb, 0)),
                                width: vImagePixelCount(CVPixelBufferGetWidthOfPlane(pb, 0)),
                                rowBytes: CVPixelBufferGetBytesPerRowOfPlane(pb, 0))
        var pixels = [UInt8](repeating: 0, count: width * height)
        let err: vImage_Error = pixels.withUnsafeMutableBytes { raw in
            var dst = vImage_Buffer(data: raw.baseAddress, height: vImagePixelCount(height),
                                    width: vImagePixelCount(width), rowBytes: width)
            return vImageScale_Planar8(&src, &dst, nil, vImage_Flags(kvImageNoFlags))
        }
        guard err == kvImageNoError else { return nil }
        return GrayImage(width: width, height: height, pixels: pixels)
    }
}

/// Small pool-backed BGRA renderer (GPU, Core Image) for model inputs, pose and replay thumbnails.
final class BufferRenderer {
    let width: Int
    let height: Int
    private var pool: CVPixelBufferPool?
    private let context: CIContext

    init(width: Int, height: Int, context: CIContext) {
        self.width = width; self.height = height; self.context = context
        let attrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: width,
            kCVPixelBufferHeightKey as String: height,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:],
        ]
        CVPixelBufferPoolCreate(nil, [kCVPixelBufferPoolMinimumBufferCountKey as String: 8] as CFDictionary, attrs as CFDictionary, &pool)
    }

    /// Renders `crop` (full-res pixel rect, top-left origin) of `source` scaled to this renderer's size.
    func render(_ source: CVPixelBuffer, crop: CGRect? = nil) -> CVPixelBuffer? {
        guard let pool = pool else { return nil }
        var out: CVPixelBuffer?
        CVPixelBufferPoolCreatePixelBuffer(nil, pool, &out)
        guard let dst = out else { return nil }
        let srcW = CGFloat(CVPixelBufferGetWidth(source)), srcH = CGFloat(CVPixelBufferGetHeight(source))
        var img = CIImage(cvPixelBuffer: source)
        let r = crop ?? CGRect(x: 0, y: 0, width: srcW, height: srcH)
        // Core Image uses a lower-left origin.
        let ciCrop = CGRect(x: r.minX, y: srcH - r.maxY, width: r.width, height: r.height)
        img = img.cropped(to: ciCrop)
            .transformed(by: CGAffineTransform(translationX: -ciCrop.minX, y: -ciCrop.minY))
            .transformed(by: CGAffineTransform(scaleX: CGFloat(width) / ciCrop.width, y: CGFloat(height) / ciCrop.height))
        context.render(img, to: dst, bounds: CGRect(x: 0, y: 0, width: width, height: height), colorSpace: CGColorSpaceCreateDeviceRGB())
        return dst
    }
}

/// Rolling buffer of JPEG thumbnails for slow-motion evidence replay.
final class FrameRing {
    struct Frame { let t: Double; let jpeg: Data }
    private var frames: [Frame] = []
    private let lock = NSLock()
    var seconds = 12.0

    func append(_ f: Frame) {
        lock.lock(); defer { lock.unlock() }
        frames.append(f)
        let cutoff = f.t - seconds
        if let i = frames.firstIndex(where: { $0.t >= cutoff }), i > 0 { frames.removeFirst(i) }
    }

    func frames(from a: Double, to b: Double) -> [Frame] {
        lock.lock(); defer { lock.unlock() }
        return frames.filter { $0.t >= a && $0.t <= b }
    }

    var latestTime: Double? { lock.lock(); defer { lock.unlock() }; return frames.last?.t }
}
