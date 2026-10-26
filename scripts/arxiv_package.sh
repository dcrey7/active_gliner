#!/usr/bin/env bash
# Build dist/arxiv.tar.gz: the paper sources arXiv needs, with the compiled bibliography.
# arXiv compiles with pdflatex and uses main.bbl as shipped, so the .bib files stay out.
set -euo pipefail

out=dist/arxiv
rm -rf "$out" dist/arxiv.tar.gz build/paper
mkdir -p "$out" build/paper

# Compile once to get main.bbl.
tectonic -X compile --keep-intermediates --outdir build/paper paper/main.tex

cp paper/main.tex paper/numbers.tex paper/acl.sty paper/acl_natbib.bst "$out"/
cp build/paper/main.bbl "$out"/
cp -r paper/tables paper/figures "$out"/

# The package must build on its own, without the .bib files. Tectonic always reruns
# BibTeX, so the check copy reads main.bbl directly, as arXiv does.
check=build/paper/check
mkdir -p "$check"
cp -r "$out"/. "$check"/
sed -i 's/^\\bibliography{.*}$/\\input{main.bbl}/' "$check"/main.tex
tectonic -X compile "$check"/main.tex

tar -czf dist/arxiv.tar.gz -C "$out" .
echo "dist/arxiv.tar.gz: $(du -h dist/arxiv.tar.gz | cut -f1)"
