# PickleRef — phone-based pickleball referee (iOS 15+ POC)

Put a phone on a tripod beside the court and open the app. It finds the court lines by itself; then you play.
Every first bounce gets an IN / OUT / TOO CLOSE call with a measured margin in cm, an uncertainty, a confidence,
and a slow-motion evidence replay. Kitchen (NVZ) volley faults and double bounces are flagged too. Extra phones
link automatically and their measurements are fused.

## Pipeline (all on-device, offline)

| Stage | Technique |
| --- | --- |
| Capture | 1080p @ 120/240 fps, custom 1/1000 s shutter (≈1.3 cm blur at 30 mph), host-clock timestamps |
| Court calibration (automatic) | Line-pixel mask (Farin test) → Hough → two line families → exhaustive model fit over line↔court-line correspondences (≈60k homographies) → TLS line refinement + LS homography from all intersections → 3-frame agreement lock → periodic re-verification (auto-recalibrates if the tripod moves) |
| Camera pose | Homography + FOV → K[R\|t]; camera height/view type → setup advice |
| Ball detection | **WASB** (HRNet heatmap, BMVC'23, MIT) Core ML on the Neural Engine, fused with Apple Vision `VNDetectTrajectoriesRequest` and 3-frame-difference blobs |
| Tracking | Multi-hypothesis tentative tracks confirmed by ballistic fit, quadratic predictive gating, reorder buffer across async detectors |
| Bounce/hit | Independent arcs fit on each side of every split; sub-frame contact = arc intersection. Classified with **monocular physics-constrained 3D arcs** (gravity + drag, linear LS against the recovered camera) |
| Contact point | Ball-radius-corrected ground projection; σ from the homography Jacobian × (fit residual ⊕ calibration RMS ⊕ blur) + skid |
| Rules | 2026 USAP rulebook: lines IN (8.B), serve NVZ line OUT, centreline IN, TOO CLOSE → IN (8.D), double bounce (10.B.1), NVZ volley + momentum (11.A.1–2) via Vision body pose |
| Multi-phone | MultipeerConnectivity auto-link, NTP-style min-RTT clock sync, inverse-variance fusion of contacts |

## Build

CI (`.github/workflows/ios.yml`, macos-15): converts WASB to Core ML (`tools/convert_wasb.py`), generates the
project with XcodeGen, builds an unsigned Release IPA, runs unit tests (synthetic court renders for calibration,
simulated ball flights for bounce localisation, rules, tracking, clock sync). Install with Sideloadly/AltStore.

## Honest limits

- Single phone: ≈ ±3–10 cm (far baseline worst). Close calls are reported as TOO CLOSE, not guessed.
- WASB weights are tennis-trained; a pickleball fine-tune (10–20k labelled frames) is the next accuracy step.
- Kitchen faults need a side view along the NVZ line.

See `docs/RESEARCH.md` for the feasibility study, competitor analysis and sources.
