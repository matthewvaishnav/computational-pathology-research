# Study package index

The main computational-pathology repository is the hub for the research program. Study-specific repositories hold detailed protocols, locked result tables, reproduction scripts, and focused paper drafts.

| Study | What it contains | Repository status |
|---|---|---|
| Paired-Acquisition Neural Factorization SCORPION core study | Primary paired-scanner method study on 48 original human H&E slides, 480 aligned tissue regions, five scanners, and DINOv2/Phikon/ResNet50 transfer | Repository target: `paired-acquisition-factorization-scorpion` |
| Paired-Acquisition Neural Factorization external canine SCC validation | Independent five-scanner canine SCC paired-acquisition validation on 805 geometry-qualified five-view regions from 44 biological samples | Repository target: `paired-acquisition-factorization-caninescc` |
| Paired-Acquisition Neural Factorization allocation study | Resource-allocation study testing biological pair diversity versus anchor repetition under matched pair-presentation budgets of 6,400 and 12,800 | Repository target: `paired-acquisition-factorization-allocation` |

## Linking policy

Public method names use logical titles such as **Paired-Acquisition Neural Factorization**. Child-package repository links should only be made clickable after the package URL is verified from the public view. Historical script names, result directories, and archived paths may retain older internal identifiers only when changing them would break reproducibility.

Do not link to GitHub Pages PDFs until those pages are actually published. The main repository remains the stable public entry point until each child package is verified.
