PDF_NAME := cloudcropper-algorithm-notes
TEX := main.tex
OUT_DIR := build
LATEXMK ?= latexmk

LATEXMK_FLAGS := -xelatex -interaction=nonstopmode -file-line-error -halt-on-error
LATEXMK_FLAGS += -synctex=1 -outdir=$(OUT_DIR) -jobname=$(PDF_NAME)

.PHONY: all pdf watch clean distclean

all: pdf

pdf:
	mkdir -p $(OUT_DIR)
	$(LATEXMK) $(LATEXMK_FLAGS) $(TEX)

watch:
	mkdir -p $(OUT_DIR)
	$(LATEXMK) -pvc $(LATEXMK_FLAGS) $(TEX)

clean:
	$(LATEXMK) -c -outdir=$(OUT_DIR) -jobname=$(PDF_NAME) $(TEX)

distclean:
	rm -rf $(OUT_DIR)

