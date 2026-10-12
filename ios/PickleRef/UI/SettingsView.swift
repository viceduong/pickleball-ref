import SwiftUI

struct SettingsView: View {
    @EnvironmentObject var model: AppModel
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationView {
            Form {
                Section("Camera") {
                    Picker("Frame rate", selection: $model.settings.targetFps) {
                        Text("60 fps").tag(60.0); Text("120 fps").tag(120.0); Text("240 fps").tag(240.0)
                    }
                    Toggle("Fast shutter (1/1000 s)", isOn: $model.settings.fastShutter)
                    Button("Recalibrate court") { model.recalibrate(); dismiss() }
                }
                Section {
                    Toggle("WASB neural ball detector", isOn: $model.settings.useNeural).disabled(!model.neuralAvailable)
                    Toggle("Apple Vision trajectories", isOn: $model.settings.useVisionTrajectories)
                    Toggle("Body pose (kitchen faults)", isOn: $model.settings.usePose)
                } header: { Text("Detection") } footer: {
                    Text(model.neuralAvailable ? "WASB (HRNet, BMVC'23) runs on the Neural Engine; motion + Vision trackers are fused with it."
                                               : "Neural model not bundled in this build; using motion + Vision trackers.")
                }
                Section("Calls") {
                    Toggle("Announce OUT / faults", isOn: $model.announce)
                    Toggle("Kitchen (NVZ) volley faults", isOn: $model.kitchenCalls)
                }
                Section {
                    Toggle("Link nearby phones", isOn: $model.linkPhones)
                    if !model.peers.syncInfo.isEmpty { Text("Clock sync: \(model.peers.syncInfo)").font(.caption) }
                } header: { Text("Multi-phone") } footer: {
                    Text("Place a second phone on the other baseline or beside the kitchen. Phones discover each other over Wi-Fi/Bluetooth, sync clocks, and fuse bounce measurements (inverse-variance).")
                }
                Section("Setup tips") {
                    Text("• Tripod 6–10 ft high, behind a baseline (line calls) or beside the kitchen line (NVZ faults).")
                    Text("• Keep all court lines in frame; calibration is automatic and re-checks every few seconds.")
                    Text("• Use 120 fps in daylight; 60 fps indoors if the image is too dark.")
                    Text("• TOO CLOSE means the measurement can't separate ball and line — rule 8.D says IN.")
                }
                .font(.footnote)
            }
            .navigationTitle("PickleRef")
            .toolbar { ToolbarItem(placement: .confirmationAction) { Button("Done") { dismiss() } } }
        }
        .navigationViewStyle(.stack)
    }
}
