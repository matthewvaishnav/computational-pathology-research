# Outreach list for federated computational pathology work

This list is for targeted research, employment, collaboration, and media outreach. It is not a claim that any target has endorsed this project.

## Priority outreach message

Short positioning:

> I independently built a computational pathology / federated learning research artifact showing a sample-volume / site-signal alignment failure mode in simulated federated pathology. A label-noise-calibrated detector-switch rule transfers to conservative ordinal threshold-shift stress with low clean-regime switching and positive 35% / 45% gains across global QWK, macro-F1, and worst-site QWK. I am looking for research software, applied ML, and medical AI opportunities where this work can be evaluated and extended.

## 1. Computational pathology and medical AI labs

### University Health Network / Princess Margaret / University of Toronto ecosystem

- Princess Margaret Cancer Centre / University Health Network - oncology research and clinical translation ecosystem.
- Ontario Cancer Institute / Princess Margaret research programs - cancer research setting connected to UHN and University of Toronto.
- University of Toronto computer science / medical biophysics / AI-for-health groups.
- Vector Institute affiliates working on health AI, robust ML, medical imaging, and trustworthy machine learning.

Suggested angle:

> This project sits at the intersection of computational pathology, federated learning, and validation reliability. I am looking for feedback, research-assistant opportunities, or collaboration paths to test the detector on naturally multi-center pathology data.

### Waterloo / Kitchener-Waterloo ecosystem

- University of Waterloo AI / machine learning researchers with health, medical imaging, privacy, or federated learning interests.
- Waterloo Data and Artificial Intelligence Institute / applied AI groups.
- Local health-tech or imaging-adjacent startups where a research-software portfolio may matter.

Suggested angle:

> I am local to Waterloo/Kitchener and have built a reproducible independent medical AI research artifact. I am looking for a research software / applied ML role or supervised extension of the work.

### International computational pathology labs to review

- Mahmood Lab / Harvard and Brigham ecosystem - computational pathology foundation models and WSI ML.
- Fuchs / computational pathology ecosystems - large-scale computational pathology and foundation-model work.
- MSK / Mount Sinai / major cancer-center computational pathology groups.
- Stanford, UCSF, Mayo, and other medical AI labs with pathology, federated learning, or health ML validation interest.

Suggested angle:

> I am not claiming clinical validation; I am asking whether the modeling failure mode and detector-switch framing are worth feedback or external validation.

## 2. Federated learning / privacy-preserving ML researchers

Target researchers and groups with interests in:

- federated optimization,
- client heterogeneity,
- robustness under non-IID data,
- validation under site shift,
- healthcare federated learning,
- privacy-preserving medical AI.

Suggested angle:

> Most federated learning discussions treat client-size weighting as a natural default. This project reframes sample-size dominance as an auditable assumption when the largest client's task-specific signal is misaligned with the validation objective.

## 3. Medical AI / computational pathology startups and companies

Target categories:

- digital pathology AI companies,
- medical imaging AI companies,
- pathology foundation-model companies,
- federated learning / privacy-preserving analytics companies,
- healthcare data infrastructure companies,
- clinical AI validation / MLOps companies.

Examples to research:

- Paige
- PathAI
- Owkin
- Aiforia
- Ibex Medical Analytics
- Proscia
- Tempus
- Imagia / health AI ecosystem
- Layer 6 / Toronto AI ecosystem
- BenchSci / biomedical AI ecosystem
- StackAdapt and other applied ML companies for non-health applied-science roles

Suggested angle:

> I am seeking applied ML / research software roles. My project demonstrates end-to-end experimental design, Python/PyTorch implementation, multi-seed evaluation, metric analysis, calibration sensitivity, CI/reproducibility work, and technical writing.

## 4. Local professors / institutional contacts

Contact types:

- Conestoga ACSIT program contacts for internal transfer and software/data alignment.
- University of Waterloo professors working in ML, health data, privacy, or medical imaging.
- University of Toronto / Vector health AI researchers.
- Local hospital research coordinators or lab managers, especially in data science or medical imaging.

Suggested angle:

> I have a nontraditional background but have produced a concrete research artifact. I am looking for advice on whether this can be shaped into a research-assistant, internship, supervised project, or degree/program pathway.

## 5. Journalists and communicators

Do not lead with hype. Lead with a careful claim boundary.

Target categories:

- AI and healthcare journalists.
- Canadian tech journalists.
- University / local innovation reporters.
- Medical AI newsletter writers.
- Open-source / independent researcher newsletters.

Suggested pitch:

> Independent researcher builds a reproducible computational pathology stress test showing why more data is not always more trustworthy in federated medical AI.

Important caveat to include in every media pitch:

> The work is research-only, not clinical validation, not diagnostic software, and not a claim about real hospital trustworthiness.

## 6. First-contact email template

Subject: Independent federated pathology research - site-signal alignment failure mode

Hi [Name],

I am an independent researcher/software developer working on computational pathology and federated medical AI. I recently built a reproducible research artifact studying a sample-volume / site-signal alignment failure mode in simulated federated pathology experiments over PANDA-derived Phikon features.

The core result: a detector-switch rule calibrated under dominant-site label-noise stress transfers to conservative ordinal threshold-shift stress, preserving low clean-regime switching while improving global QWK, macro-F1, and worst-site QWK at 35% and 45% shift. The project includes an arXiv-style PDF, reproducibility artifacts, diagnostic ablations, and calibration-sensitivity analysis.

I am not claiming clinical validation or deployment readiness. The narrower claim is that sample-size dominance in federated pathology should be treated as an auditable modeling assumption rather than an automatic guarantee of aggregation safety.

Paper PDF: https://matthewvaishnav.github.io/computational-pathology-research/
GitHub: https://github.com/matthewvaishnav/computational-pathology-research

I would be grateful for any feedback, advice, or suggestions about where this work might fit - especially research software, applied ML, computational pathology, federated learning, or medical AI validation opportunities.

Best,
Matthew Vaishnav

## 7. Outreach tracking table

| Target | Type | Contact found? | Message sent? | Reply | Follow-up date | Notes |
|---|---|---:|---:|---|---|---|
| UHN / Princess Margaret computational pathology contacts | Lab / hospital research | No | No | - | - | Start with lab pages and research coordinator contacts. |
| Vector Institute health AI affiliates | Research network | No | No | - | - | Look for health AI, robust ML, federated learning, medical imaging. |
| University of Waterloo health/medical AI researchers | Local professors | No | No | - | - | Prioritize Kitchener-Waterloo proximity. |
| PathAI / Paige / Proscia / Aiforia / Ibex / Owkin | Companies | No | No | - | - | Look for research software / ML internship / applied scientist roles. |
| Canadian AI / health-tech journalists | Media | No | No | - | - | Only after paper site and GitHub are clean. |
