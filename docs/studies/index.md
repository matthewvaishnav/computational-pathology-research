# Study package index

The main computational-pathology repository is the hub for the research program. Study-specific repositories hold detailed protocols, locked result tables, reproduction scripts, and focused paper drafts.

| Study | What it contains | Repository |
|---|---|---|
| Paired-Acquisition Neural Factorization SCORPION core study | Primary paired-scanner method study on 48 original human H&E slides, 480 aligned tissue regions, five scanners, and DINOv2/Phikon/ResNet50 transfer | Repository target: `paired-acquisition-factorization-scorpion` |
| Paired-Acquisition Neural Factorization external canine SCC validation | Independent five-scanner canine SCC paired-acquisition validation on 805 geometry-qualified five-view regions from 44 biological samples | [paired-acquisition-factorization-caninescc](https://github.com/matthewvaishnav/paired-acquisition-factorization-caninescc) |
| Paired-Acquisition Neural Factorization allocation study | Resource-allocation study testing biological pair diversity versus anchor repetition under matched pair-presentation budgets of 6,400 and 12,800 | [paired-acquisition-factorization-allocation](https://github.com/matthewvaishnav/paired-acquisition-factorization-allocation) |

## Linking policy

Public method names use logical titles such as **Paired-Acquisition Neural Factorization**. Repository URLs should use the current descriptive child-repository slugs. Historical script names, result directories, and archived paths may retain older internal identifiers only when changing them would break reproducibility.

Do not link to GitHub Pages PDFs until those pages are actually published. The repository link is the stable public entry point for each study package.
