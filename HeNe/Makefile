all: build/HeNe.pdf


TeXOptions = -lualatex \
			 -interaction=nonstopmode \
			 -halt-on-error \
			 -output-directory=build

build/HeNe.pdf: FORCE | build
	latexmk $(TeXOptions) HeNe.tex

FORCE:

build:
	mkdir -p build/

clean:
	rm -rf build
