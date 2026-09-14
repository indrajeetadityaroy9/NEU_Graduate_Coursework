# Makefile for LaTeX paper compilation

MAIN = main
LATEX = pdflatex
BIBTEX = bibtex

.PHONY: all clean view

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex
	$(LATEX) $(MAIN)
	$(LATEX) $(MAIN)  # Run twice for references
	$(LATEX) $(MAIN)  # Run thrice for cross-references

clean:
	rm -f *.aux *.log *.bbl *.blg *.out *.toc *.lof *.lot *.fls *.fdb_latexmk *.synctex.gz

distclean: clean
	rm -f $(MAIN).pdf

view: $(MAIN).pdf
	open $(MAIN).pdf 2>/dev/null || xdg-open $(MAIN).pdf 2>/dev/null || echo "PDF viewer not found"

# Quick compile (single pass)
quick:
	$(LATEX) $(MAIN)
