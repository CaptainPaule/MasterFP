all: build/Operationsverstaerker.pdf


TeXOptions = -lualatex \
			 -interaction=nonstopmode \
			 -halt-on-error \
			 -output-directory=build

build/Operationsverstaerker.pdf: FORCE | build
	latexmk $(TeXOptions) Operationsverstaerker.tex

FORCE:

build:
	mkdir -p build/

clean:
	rm -rf build
