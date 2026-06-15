#!/usr/bin/env python3
from pathlib import Path
import re

path = Path("paper/arxiv/build/main.tex")
text = path.read_text(encoding="utf-8")

text = re.sub(
    r"\\documentclass\[[^\]]*\]\{article\}",
    r"\\documentclass[10pt,twocolumn]{article}",
    text,
    count=1,
)
text = re.sub(
    r"\\usepackage\[[^\]]*\]\{geometry\}",
    r"\\usepackage[letterpaper,top=0.70in,bottom=0.75in,left=0.68in,right=0.68in,columnsep=0.24in]{geometry}",
    text,
    count=1,
)
text = re.sub(
    r"\\date\{[^}]*\}",
    r"\\date{June 15, 2026}",
    text,
    count=1,
)

anchor = r"\usepackage[numbers,sort&compress]{natbib}"
style = "\n".join([
    anchor,
    r"\usepackage{indentfirst}",
    r"\setlength{\columnsep}{0.24in}",
    r"\setlength{\parindent}{1em}",
    r"\setlength{\parskip}{0pt}",
    r"\setlength{\abovedisplayskip}{4pt plus 1pt minus 1pt}",
    r"\setlength{\belowdisplayskip}{4pt plus 1pt minus 1pt}",
    r"\setlength{\textfloatsep}{7pt plus 2pt minus 2pt}",
    r"\setlength{\floatsep}{6pt plus 2pt minus 2pt}",
    r"\setlength{\intextsep}{6pt plus 2pt minus 2pt}",
    r"\emergencystretch=1em",
])
if r"\usepackage{indentfirst}" not in text:
    text = text.replace(anchor, style, 1)

text = text.replace(
    r"\setlength{\parindent}{0pt}%",
    r"\setlength{\parindent}{1em}%",
)
text = text.replace(
    r"\setlength{\parskip}{0.35em}%",
    r"\setlength{\parskip}{0pt}%",
)

calc = r"\input{identifiability_calculations.tex}"
if calc not in text:
    text = text.replace(
        r"\section{Limitations}",
        calc + "\n\n" + r"\section{Limitations}",
        1,
    )

if r"\setlength{\parindent}{0pt}" in text:
    raise RuntimeError("Block-style indentation remains")
if calc not in text:
    raise RuntimeError("Calculations section missing")

path.write_text(text, encoding="utf-8")
print(f"Prepared {path}")
