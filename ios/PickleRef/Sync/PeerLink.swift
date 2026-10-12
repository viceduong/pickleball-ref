import Foundation
import MultipeerConnectivity
import UIKit

/// Wire messages between linked phones.
enum PeerMessage: Codable {
    case ping(t0: Double)
    case pong(t0: Double, t1: Double, t2: Double)
    case contact(t: Double, x: Double, y: Double, sx: Double, sy: Double, kind: String)
    case status(calibrated: Bool, view: String)
}

/// Auto-discovering multi-phone link (MultipeerConnectivity: Wi-Fi / peer-to-peer Wi-Fi / Bluetooth),
/// with continuous clock synchronisation so contact events can be fused across cameras.
final class PeerLink: NSObject, ObservableObject {
    static let serviceType = "pkref"

    @Published private(set) var peers: [String] = []
    @Published private(set) var syncInfo: String = ""

    var onContact: ((_ localTime: Double, _ m: ContactMeasurement, _ kind: String, _ peer: String) -> Void)?

    private let myPeer = MCPeerID(displayName: UIDevice.current.name + "-" + String(UUID().uuidString.prefix(4)))
    private lazy var session: MCSession = {
        let s = MCSession(peer: myPeer, securityIdentity: nil, encryptionPreference: .required)
        s.delegate = self
        return s
    }()
    private var advertiser: MCNearbyServiceAdvertiser?
    private var browser: MCNearbyServiceBrowser?
    private var clocks: [MCPeerID: ClockSync] = [:]
    private var timer: Timer?
    private let queue = DispatchQueue(label: "peerlink")

    static func now() -> Double { CACurrentMediaTime() }

    func start() {
        guard advertiser == nil else { return }
        let adv = MCNearbyServiceAdvertiser(peer: myPeer, discoveryInfo: nil, serviceType: Self.serviceType)
        adv.delegate = self
        adv.startAdvertisingPeer()
        advertiser = adv
        let br = MCNearbyServiceBrowser(peer: myPeer, serviceType: Self.serviceType)
        br.delegate = self
        br.startBrowsingForPeers()
        browser = br
        timer = Timer.scheduledTimer(withTimeInterval: 0.25, repeats: true) { [weak self] _ in self?.pingAll() }
    }

    func stop() {
        timer?.invalidate(); timer = nil
        advertiser?.stopAdvertisingPeer(); advertiser = nil
        browser?.stopBrowsingForPeers(); browser = nil
        session.disconnect()
        DispatchQueue.main.async { self.peers = [] }
    }

    func broadcast(contact m: ContactMeasurement, kind: String) {
        send(.contact(t: m.t, x: m.point.x, y: m.point.y, sx: m.sigma.x, sy: m.sigma.y, kind: kind), to: session.connectedPeers)
    }

    private func pingAll() {
        guard !session.connectedPeers.isEmpty else { return }
        send(.ping(t0: Self.now()), to: session.connectedPeers, reliable: false)
        let info = session.connectedPeers.compactMap { p -> String? in
            guard let c = queue.sync(execute: { clocks[p] }), let u = c.uncertainty else { return nil }
            return String(format: "%@ ±%.1f ms", p.displayName.components(separatedBy: "-").first ?? "", u * 1000)
        }.joined(separator: ", ")
        DispatchQueue.main.async { self.syncInfo = info }
    }

    private func send(_ msg: PeerMessage, to peers: [MCPeerID], reliable: Bool = true) {
        guard !peers.isEmpty, let data = try? JSONEncoder().encode(msg) else { return }
        try? session.send(data, toPeers: peers, with: reliable ? .reliable : .unreliable)
    }
}

extension PeerLink: MCSessionDelegate {
    func session(_ session: MCSession, peer peerID: MCPeerID, didChange state: MCSessionState) {
        let names = session.connectedPeers.map { $0.displayName.components(separatedBy: "-").first ?? $0.displayName }
        DispatchQueue.main.async { self.peers = names }
        if state == .notConnected { queue.sync { _ = clocks.removeValue(forKey: peerID) } }
    }

    func session(_ session: MCSession, didReceive data: Data, fromPeer peerID: MCPeerID) {
        let t = Self.now()
        guard let msg = try? JSONDecoder().decode(PeerMessage.self, from: data) else { return }
        switch msg {
        case .ping(let t0):
            send(.pong(t0: t0, t1: t, t2: Self.now()), to: [peerID], reliable: false)
        case .pong(let t0, let t1, let t2):
            queue.sync {
                var c = clocks[peerID] ?? ClockSync()
                c.add(.init(t0: t0, t1: t1, t2: t2, t3: t))
                clocks[peerID] = c
            }
        case .contact(let rt, let x, let y, let sx, let sy, let kind):
            guard let local = queue.sync(execute: { clocks[peerID]?.toLocal(remoteTime: rt) }) else { return }
            let m = ContactMeasurement(point: Vec2(x, y), sigma: Vec2(sx, sy), t: local, imagePoint: .zero)
            onContact?(local, m, kind, peerID.displayName)
        case .status:
            break
        }
    }

    func session(_ session: MCSession, didReceive stream: InputStream, withName streamName: String, fromPeer peerID: MCPeerID) {}
    func session(_ session: MCSession, didStartReceivingResourceWithName resourceName: String, fromPeer peerID: MCPeerID, with progress: Progress) {}
    func session(_ session: MCSession, didFinishReceivingResourceWithName resourceName: String, fromPeer peerID: MCPeerID, at localURL: URL?, withError error: Error?) {}
}

extension PeerLink: MCNearbyServiceAdvertiserDelegate, MCNearbyServiceBrowserDelegate {
    func advertiser(_ advertiser: MCNearbyServiceAdvertiser, didReceiveInvitationFromPeer peerID: MCPeerID,
                    withContext context: Data?, invitationHandler: @escaping (Bool, MCSession?) -> Void) {
        invitationHandler(true, session)
    }

    func browser(_ browser: MCNearbyServiceBrowser, foundPeer peerID: MCPeerID, withDiscoveryInfo info: [String: String]?) {
        // Deterministic inviter avoids double invitations.
        if myPeer.displayName < peerID.displayName {
            browser.invitePeer(peerID, to: session, withContext: nil, timeout: 10)
        }
    }

    func browser(_ browser: MCNearbyServiceBrowser, lostPeer peerID: MCPeerID) {}
}
