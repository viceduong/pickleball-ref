import SwiftUI

struct LiveView: View {
    @EnvironmentObject var model: AppModel
    @State private var showSettings = false
    @State private var challenged: RefCall?
    @State private var showLog = false

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            CameraPreview(session: model.capture.session).ignoresSafeArea()
            GeometryReader { geo in
                OverlayCanvas(mapper: ImageMapper(image: model.imageSize, view: geo.size))
            }
            .ignoresSafeArea()
            .allowsHitTesting(false)

            VStack(spacing: 0) {
                topBar
                Spacer()
                bottomBar
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 10)

            if model.cameraDenied {
                Text("Camera access is required. Enable it in Settings → PickleRef.")
                    .padding().background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
            }
        }
        .sheet(isPresented: $showSettings) { SettingsView().environmentObject(model) }
        .sheet(item: $challenged) { call in ChallengeView(call: call).environmentObject(model) }
        .sheet(isPresented: $showLog) { CallLogView(onSelect: { c in showLog = false; DispatchQueue.main.asyncAfter(deadline: .now() + 0.4) { challenged = c } }).environmentObject(model) }
    }

    private var topBar: some View {
        HStack(alignment: .top, spacing: 8) {
            HStack(spacing: 6) {
                Circle().fill(model.geometry == nil ? Color.orange : Color.green).frame(width: 9, height: 9)
                Text(model.status).font(.footnote.weight(.semibold)).lineLimit(2)
            }
            .padding(.horizontal, 10).padding(.vertical, 6)
            .background(.ultraThinMaterial, in: Capsule())
            Spacer()
            VStack(alignment: .trailing, spacing: 4) {
                Text(String(format: "%.0f fps", model.fps)).font(.caption.monospacedDigit())
                if model.linkPhones {
                    Text(model.peers.peers.isEmpty ? "Linking…" : "Linked: \(model.peers.peers.count + 1) phones")
                        .font(.caption)
                }
                if model.settings.useNeural && model.neuralAvailable { Text("WASB NN").font(.caption2).foregroundColor(.mint) }
            }
            .padding(6).background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 8))
            Button { showSettings = true } label: {
                Image(systemName: "gearshape.fill").font(.title3).padding(8).background(.ultraThinMaterial, in: Circle())
            }
        }
    }

    private var bottomBar: some View {
        HStack(alignment: .bottom, spacing: 12) {
            if let c = model.latestCall {
                Button { challenged = c } label: { CallBadge(call: c) }.buttonStyle(.plain)
            } else {
                Text(model.geometry == nil ? "Aim at the court — calibration is automatic" : "Play! Calls appear here")
                    .font(.callout).padding(10).background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 10))
            }
            Spacer()
            Button { showLog = true } label: {
                Image(systemName: "list.bullet").font(.title3).padding(12).background(.ultraThinMaterial, in: Circle())
            }
            Button {
                if let c = model.latestCall { challenged = c }
            } label: {
                Text("CHALLENGE").font(.headline.weight(.heavy)).padding(.horizontal, 22).padding(.vertical, 14)
                    .background(model.latestCall == nil ? Color.gray : Color.blue, in: Capsule())
                    .foregroundColor(.white)
            }
            .disabled(model.latestCall == nil)
        }
    }
}

struct CallBadge: View {
    let call: RefCall
    var body: some View {
        HStack(spacing: 10) {
            Text(call.headline).font(.title2.weight(.black)).foregroundColor(call.verdict.color)
            VStack(alignment: .leading, spacing: 2) {
                Text(call.kind.rawValue).font(.caption.weight(.bold))
                if call.kind != .doubleBounce {
                    Text(String(format: "%+.1f cm ± %.1f · %.0f%%", call.marginCm, call.sigmaCm, call.confidence * 100))
                        .font(.caption.monospacedDigit())
                }
            }
        }
        .padding(.horizontal, 12).padding(.vertical, 8)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
    }
}

/// Court overlay, live ball trail and recent events drawn over the camera preview.
struct OverlayCanvas: View {
    @EnvironmentObject var model: AppModel
    let mapper: ImageMapper

    var body: some View {
        Canvas { ctx, _ in
            if let g = model.geometry {
                var path = Path()
                for s in Court.segments {
                    path.move(to: mapper.map(g.toImage(s.a)))
                    path.addLine(to: mapper.map(g.toImage(s.b)))
                }
                ctx.stroke(path, with: .color(.green.opacity(0.75)), lineWidth: 1.5)
                // Kitchen tint.
                for h in [Court.Half.near, .far] {
                    let k = Court.kitchen(h)
                    var kp = Path()
                    let pts = [Vec2(k.x0, k.y0), Vec2(k.x1, k.y0), Vec2(k.x1, k.y1), Vec2(k.x0, k.y1)].map { mapper.map(g.toImage($0)) }
                    kp.addLines(pts); kp.closeSubpath()
                    ctx.fill(kp, with: .color(.orange.opacity(0.12)))
                }
            }
            let trail = model.snapshot.trail
            for (i, p) in trail.enumerated() {
                let a = Double(i + 1) / Double(max(1, trail.count))
                let c = mapper.map(p)
                ctx.fill(Path(ellipseIn: CGRect(x: c.x - 3, y: c.y - 3, width: 6, height: 6)), with: .color(.yellow.opacity(a)))
            }
            for e in model.snapshot.lastEvents {
                let c = mapper.map(e.imagePoint)
                let r: CGFloat = 9
                let rect = CGRect(x: c.x - r, y: c.y - r, width: 2 * r, height: 2 * r)
                ctx.stroke(e.kind == .bounce ? Path(ellipseIn: rect) : Path(rect), with: .color(e.kind == .bounce ? .cyan : .pink), lineWidth: 2)
            }
            if let call = model.latestCall, call.kind != .kitchen {
                let c = mapper.map(Vec2(call.imageX, call.imageY))
                ctx.stroke(Path(ellipseIn: CGRect(x: c.x - 14, y: c.y - 14, width: 28, height: 28)),
                           with: .color(call.verdict.color), lineWidth: 3)
            }
        }
    }
}

struct CallLogView: View {
    @EnvironmentObject var model: AppModel
    @Environment(\.dismiss) private var dismiss
    var onSelect: (RefCall) -> Void

    var body: some View {
        NavigationView {
            List(model.calls.reversed()) { c in
                Button { onSelect(c) } label: {
                    HStack {
                        Text(c.headline).font(.headline).foregroundColor(c.verdict.color).frame(width: 150, alignment: .leading)
                        Text(c.kind.rawValue)
                        Spacer()
                        Text(c.detail).font(.caption).foregroundColor(.secondary).lineLimit(1)
                    }
                }
            }
            .navigationTitle("Calls")
            .toolbar { ToolbarItem(placement: .confirmationAction) { Button("Done") { dismiss() } } }
        }
        .navigationViewStyle(.stack)
    }
}
