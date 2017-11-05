all: build/optischesPumpen.pdf

TeXOptions = -lualatex \
			 -interaction=nonstopmode \
			 -halt-on-error \
			 -output-directory=build

optischesPumpen/img/plotLande.pdf: optischesPumpen/scripts/plotLande.py optischesPumpen/data/messwerteLandeFaktor.txt
	python /optischesPumpen/scripts/plotLande.py

build/optischesPumpen.pdf: /optischesPumpen/img/plotLande.pdf
	latexmk $(TeXOptions) ./optischesPumpen/optischesPumpen.tex

FORCE:

build:
	mkdir -p build/

clean:
	rm -rf build
