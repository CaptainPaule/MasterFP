all: build/Lehrstuhlversuch.pdf


TeXOptions = -lualatex \
			 -interaction=nonstopmode \
			 -halt-on-error \
			 -output-directory=build

build/Lehrstuhlversuch.pdf: FORCE | build
	latexmk $(TeXOptions) Lehrstuhlversuch.tex

FORCE:

build:
	mkdir -p build/

clean:
	rm -rf build
