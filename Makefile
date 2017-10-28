all: build/optischesPumpen.pdf


TeXOptions = -lualatex \
			 -interaction=nonstopmode \
			 -halt-on-error \
			 -output-directory=build

build/optischesPumpen.pdf: FORCE | build
	latexmk $(TeXOptions) /optischesPumpen/optischesPumpen.tex

FORCE:

build:
	mkdir -p build/

clean:
	rm -rf build
