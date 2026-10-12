import Foundation

/// NTP/Cristian-style offset estimation between two host clocks.
/// Each sample: t0 local send, t1 remote receive, t2 remote send, t3 local receive.
/// Only the lowest-RTT samples are trusted (queueing delay is one-sided noise).
struct ClockSync {
    struct Sample { var t0, t1, t2, t3: Double
        var rtt: Double { (t3 - t0) - (t2 - t1) }
        var offset: Double { ((t1 - t0) + (t2 - t3)) / 2 }   // remote − local
    }

    private(set) var samples: [Sample] = []
    var capacity = 64

    mutating func add(_ s: Sample) {
        guard s.rtt >= 0 else { return }
        samples.append(s)
        if samples.count > capacity { samples.removeFirst(samples.count - capacity) }
    }

    /// Estimated remote − local offset (s), from the best quartile by RTT.
    var offset: Double? {
        guard samples.count >= 3 else { return nil }
        let best = samples.sorted { $0.rtt < $1.rtt }.prefix(max(3, samples.count / 4))
        return Stats.median(best.map(\.offset))
    }

    /// Uncertainty bound: half the best round-trip time.
    var uncertainty: Double? {
        guard let m = samples.map(\.rtt).min() else { return nil }
        return m / 2
    }

    func toLocal(remoteTime t: Double) -> Double? { offset.map { t - $0 } }
}
