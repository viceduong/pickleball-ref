import SwiftUI

/// Evidence screen: slow-motion replay with the tracked ball, verdict with measured margin and
/// uncertainty, a true-scale close-up of ball footprint vs line, and the court map.
struct ChallengeView: View {
    @EnvironmentObject var model: AppModel
    @Environment(\.dismiss) private var dismiss
    let call: RefCall

    @State private var frames: [FrameRing.Frame] = []
    @State private var index = 0
    @State private var playing = true
    @State private var speed = 0.25
    private let tick = Timer.publish(every: 1.0 / 60, on: .main, in: .common).autoconnect()
    @State private var accumulator = 0.0

    var body: some View {
        GeometryReader { geo in
            HStack(spacing: 12) {
                replay.frame(width: geo.size.width * 0.58)
                ScrollView {
                    VStack(alignment: .leading, spacing: 12) {
                        verdictHeader
                        if call.kind != .doubleBounce {
                            FootprintView(call: call).frame(height: 150)
                        }
                        CourtMapView(highlight: call, calls: model.calls.filter { $0.rallyId == call.rallyId })
                            .frame(height: 140)
                        Text("Rule \(call.rule) · \(call.detail)").font(.caption).foregroundColor(.secondary)
                        if call.verdict == .tooClose {
                            Text("Measurement cannot separate the ball from the line with 95% certainty. Rule 8.D: a ball that cannot be called out is IN.")
                                .font(.caption2).foregroundColor(.yellow)
                        }
                        if call.fusedPeers > 0 {
                            Text("Fused with \(call.fusedPeers) other phone\(call.fusedPeers > 1 ? "s" : "")").font(.caption2).foregroundColor(.mint)
                        }
                    }
                }
            }
            .padding(12)
        }
        .background(Color.black.ignoresSafeArea())
        .onAppear(perform: load)
        .onReceive(tick) { _ in advance() }
        .overlay(alignment: .topTrailing) {
            Button { dismiss() } label: { Image(systemName: "xmark.circle.fill").font(.title) }.padding(8)
        }
    }

    private var verdictHeader: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(call.headline).font(.system(size: 40, weight: .black)).foregroundColor(call.verdict.color)
            Text(call.kind.rawValue + " call").font(.headline)
            if call.kind != .doubleBounce {
                Text(String(format: "%@ %.1f cm  (± %.1f cm, 1σ) · confidence %.0f%%",
                            call.marginCm >= 0 ? "Inside by" : "Outside by", abs(call.marginCm), call.sigmaCm, call.confidence * 100))
                    .font(.subheadline.monospacedDigit())
            }
        }
    }

    private var replay: some View {
        VStack(spacing: 8) {
            ZStack {
                Color.black
                if frames.indices.contains(index), let img = UIImage(data: frames[index].jpeg) {
                    GeometryReader { g in
                        let mapper = ImageMapper(image: Vec2(Double(img.size.width), Double(img.size.height)), view: g.size)
                        let k = Double(img.size.width) / model.imageSize.x
                        ZStack {
                            Image(uiImage: img).resizable().aspectRatio(contentMode: .fit)
                                .frame(width: g.size.width, height: g.size.height)
                            Canvas { ctx, _ in
                                let t = frames[index].t
                                let c = mapper.map(Vec2(call.imageX, call.imageY) * k)
                                if t >= call.t - 0.02 {
                                    ctx.stroke(Path(ellipseIn: CGRect(x: c.x - 12, y: c.y - 12, width: 24, height: 24)),
                                               with: .color(call.verdict.color), lineWidth: 3)
                                }
                                if let geo = model.geometry {
                                    var p = Path()
                                    for s in Court.segments {
                                        p.move(to: mapper.map(geo.toImage(s.a) * k)); p.addLine(to: mapper.map(geo.toImage(s.b) * k))
                                    }
                                    ctx.stroke(p, with: .color(.green.opacity(0.35)), lineWidth: 1)
                                }
                            }
                        }
                    }
                } else {
                    Text("No replay frames").foregroundColor(.secondary)
                }
            }
            .clipShape(RoundedRectangle(cornerRadius: 10))
            HStack {
                Button { playing.toggle() } label: { Image(systemName: playing ? "pause.fill" : "play.fill") }
                Slider(value: Binding(get: { Double(index) }, set: { index = Int($0); playing = false }),
                       in: 0...Double(max(1, frames.count - 1)))
                Picker("", selection: $speed) {
                    Text("¼×").tag(0.25); Text("½×").tag(0.5); Text("1×").tag(1.0)
                }.pickerStyle(.segmented).frame(width: 140)
            }
            if frames.indices.contains(index) {
                Text(String(format: "t = %+.3f s", frames[index].t - call.t)).font(.caption.monospacedDigit()).foregroundColor(.secondary)
            }
        }
    }

    private func load() {
        frames = model.capture.ring.frames(from: call.t - 1.0, to: call.t + 0.6)
        index = 0
    }

    private func advance() {
        guard playing, frames.count > 1 else { return }
        accumulator += speed / 60
        while index + 1 < frames.count, frames[index + 1].t - frames[index].t <= accumulator {
            accumulator -= frames[index + 1].t - frames[index].t
            index += 1
        }
        if index + 1 >= frames.count { index = 0; accumulator = 0 }
    }
}

/// True-scale top-down close-up: line (2 in) vs ball footprint, with the 2σ uncertainty band.
struct FootprintView: View {
    let call: RefCall

    var body: some View {
        Canvas { ctx, size in
            let p = call.courtPoint
            let region = call.kind == .serve ? Court.serviceCourt(half: Court.half(of: p), side: Court.side(of: p)) : Court.fullCourt
            let (_, normal) = region.signedMargin(p)
            // Window: 1.2 ft around the nearest boundary point, scaled to fit.
            let span = 1.2
            let s = Double(min(size.width, size.height)) / span
            let center = CGPoint(x: size.width / 2, y: size.height / 2)
            let n = normal == .zero ? Vec2(0, 1) : normal
            // Draw in a frame where +x points outward across the boundary.
            func toView(_ d: Double, _ along: Double) -> CGPoint { CGPoint(x: Double(center.x) + d * s, y: Double(center.y) - along * s) }
            let margin = call.marginCm / Court.feetToCm - LineJudge.contactPatchRadius
            ctx.fill(Path(CGRect(origin: .zero, size: size)), with: .color(Color(red: 0.12, green: 0.35, blue: 0.6)))
            // Outside area.
            ctx.fill(Path(CGRect(x: center.x, y: 0, width: size.width / 2, height: size.height)), with: .color(Color(red: 0.1, green: 0.4, blue: 0.25)))
            // Line: occupies [-lineWidth, 0] measured inward from the outer edge.
            let l0 = toView(-Court.lineWidth, span), l1 = toView(0, -span)
            ctx.fill(Path(CGRect(x: l0.x, y: 0, width: l1.x - l0.x, height: size.height)), with: .color(.white))
            // Ball footprint (contact patch) and full ball outline.
            let bx = -margin
            let sig = call.sigmaCm / Court.feetToCm
            let band = toView(bx - 2 * sig, 0), band2 = toView(bx + 2 * sig, 0)
            ctx.fill(Path(CGRect(x: band.x, y: center.y - 30, width: band2.x - band.x, height: 60)), with: .color(.yellow.opacity(0.25)))
            let c = toView(bx, 0)
            let rb = Court.ballRadius * s, rc = LineJudge.contactPatchRadius * s
            ctx.stroke(Path(ellipseIn: CGRect(x: c.x - rb, y: c.y - rb, width: 2 * rb, height: 2 * rb)), with: .color(.yellow.opacity(0.6)), lineWidth: 1)
            ctx.fill(Path(ellipseIn: CGRect(x: c.x - rc, y: c.y - rc, width: 2 * rc, height: 2 * rc)), with: .color(call.verdict.color))
            ctx.draw(Text("OUT →").font(.caption2).foregroundColor(.white.opacity(0.7)), at: CGPoint(x: size.width - 30, y: 12))
            ctx.draw(Text("← IN").font(.caption2).foregroundColor(.white.opacity(0.7)), at: CGPoint(x: 28, y: 12))
            _ = n
        }
        .clipShape(RoundedRectangle(cornerRadius: 10))
    }
}

/// Top-down court with this rally's calls.
struct CourtMapView: View {
    let highlight: RefCall
    let calls: [RefCall]

    var body: some View {
        Canvas { ctx, size in
            // Court drawn horizontally: court y → view x.
            let s = min(Double(size.width) / (Court.length + 6), Double(size.height) / (Court.width + 4))
            let ox = (Double(size.width) - Court.length * s) / 2, oy = (Double(size.height) - Court.width * s) / 2
            func v(_ p: Vec2) -> CGPoint { CGPoint(x: ox + p.y * s, y: oy + (Court.width - p.x) * s) }
            ctx.fill(Path(CGRect(origin: .zero, size: size)), with: .color(Color(red: 0.1, green: 0.4, blue: 0.25)))
            var court = Path()
            court.addRect(CGRect(x: ox, y: oy, width: Court.length * s, height: Court.width * s))
            ctx.fill(court, with: .color(Color(red: 0.12, green: 0.35, blue: 0.6)))
            var lines = Path()
            for seg in Court.segments { lines.move(to: v(seg.a)); lines.addLine(to: v(seg.b)) }
            ctx.stroke(lines, with: .color(.white), lineWidth: 1.5)
            var net = Path(); net.move(to: v(Vec2(-1, Court.net))); net.addLine(to: v(Vec2(Court.width + 1, Court.net)))
            ctx.stroke(net, with: .color(.gray), style: StrokeStyle(lineWidth: 2, dash: [3, 2]))
            for c in calls {
                let p = v(c.courtPoint)
                let r: CGFloat = c.id == highlight.id ? 6 : 3.5
                ctx.fill(Path(ellipseIn: CGRect(x: p.x - r, y: p.y - r, width: 2 * r, height: 2 * r)), with: .color(c.verdict.color))
            }
        }
        .clipShape(RoundedRectangle(cornerRadius: 10))
    }
}
