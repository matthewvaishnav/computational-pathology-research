# Short demo / figure thread

This is reusable copy for LinkedIn, X, BlueSky, email intros, or a GitHub release note.

## Short post

I built an independent computational pathology research project studying a failure mode in federated medical AI: FedAvg gives more aggregation authority to clients with more samples, but in pathology sample count is not always the same as task-specific site-signal alignment.

Using simulated federations over PANDA-derived Phikon prostate pathology features, I perturbed the largest simulated site's training signal while keeping validation labels clean. A detector-switch rule calibrated under dominant-site label-noise stress transferred to conservative ordinal threshold-shift stress: low clean-regime switching at 0%, then positive gains at 35% and 45% shift across global QWK, macro-F1, and worst-site QWK.

The result is research-only, not clinical software. The supported claim is narrower: sample-size dominance should be treated as an auditable modeling assumption, not an automatic guarantee of aggregation safety.

Paper PDF: https://matthewvaishnav.github.io/computational-pathology-research/

GitHub: https://github.com/matthewvaishnav/computational-pathology-research

## Thread version

**1/** I have been working on an independent computational pathology / federated learning research project. The core question: what happens when the biggest client in a federated pathology system has the most samples, but its training labels are less aligned with the validation objective?

**2/** Standard FedAvg weights clients by sample count. That is useful, but it silently assumes sample volume is a good proxy for aggregation authority. In pathology, that can fail because labels can differ by grading thresholds, staining/scanning workflow, case mix, annotation source, or reporting practice.

**3/** I simulated this failure mode using PANDA-derived Phikon prostate pathology feature representations. The largest simulated client was perturbed while validation labels stayed clean. The task was ISUP grade prediction, evaluated with ordinally relevant metrics like QWK.

**4/** The headline test: a fixed detector-switch rule calibrated under dominant-site label-noise stress was evaluated on conservative ordinal threshold-shift stress. In clean conditions, it switched rarely. Under stronger shift, it switched more and improved performance.

**5/** Conservative shift results: at 0% shift, trigger rate was 13.3% with near-zero global-QWK cost. At 35% shift, the detector-switch policy improved global QWK by +0.00542, macro-F1 by +0.00838, and worst-site QWK by +0.00991.

**6/** At 45% conservative shift, the detector-switch policy improved global QWK by +0.01053, macro-F1 by +0.01512, and worst-site QWK by +0.01290. These are not clinical claims; they are controlled simulated-federation stress results.

**7/** Diagnostic analysis suggested the detector was mainly triggered by ordinal-error increase and QWK degradation, not by one arbitrary site-spread heuristic. Removing the most frequent diagnostic reduced trigger rate but did not collapse the 35% / 45% transfer result.

**8/** Calibration sensitivity also looked encouraging: 29 of 36 nearby detector configurations preserved low clean-regime switching and positive 35% / 45% gains across global QWK, macro-F1, and worst-site QWK.

**9/** The takeaway: in federated computational pathology, more data does not automatically mean more trustworthy training influence. Sample-size dominance should be auditable, especially when site-specific label processes may diverge from the declared validation objective.

**10/** Research-only. Not clinical validation. Not diagnostic software. Not deployment-ready. The goal is to make a modeling assumption visible and testable.

PDF: https://matthewvaishnav.github.io/computational-pathology-research/

GitHub: https://github.com/matthewvaishnav/computational-pathology-research

## Figure captions for outreach

**Figure 1 - Problem schematic:** FedAvg gives sample-size authority to the largest client, but that client's training-label process can become less aligned with the validation objective.

**Figure 3 - Detector transfer:** The label-noise-calibrated detector transfers to conservative ordinal threshold shift, with low clean switching and stronger positive gains at 35% and 45% shift.

**Figure 4 - Ablation / calibration:** The detector result is not only a single-diagnostic or exact-threshold artifact; it persists across ablation and nearby calibration settings.
