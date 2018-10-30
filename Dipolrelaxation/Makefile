all: build/Dipolrelaxation.pdf


TeXOptions = -lualatex \
			 -interaction=nonstopmode \
			 -halt-on-error \
			 -output-directory=build

build/Dipolrelaxation.pdf: FORCE | build
	latexmk $(TeXOptions) Dipolrelaxation.tex

FORCE:

build:
	mkdir -p build/

clean:
	rm -rf build
