#!/usr/bin/env bash
# Portable local build; Overleaf builds main.tex automatically.
set -euo pipefail
cd "$(dirname "$0")"
for tool in pdflatex; do command -v "$tool" >/dev/null || { echo "Missing $tool" >&2; exit 1; }; done
if [[ ! -f figures/fig1_observation.pdf ]]; then
  python3 plotting/make_figures.py
fi
BIB=""
for candidate in bibtex bibtex.original bibtex8; do
  if command -v "$candidate" >/dev/null 2>&1; then BIB="$candidate"; break; fi
done
[[ -n "$BIB" ]] || { echo "Missing BibTeX executable" >&2; exit 1; }
pdflatex -interaction=nonstopmode -halt-on-error main.tex
"$BIB" main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
printf '\nCompiled main.pdf. Confirm maintextend in main.aux before submission.\n'
