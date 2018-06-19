all: build/KosmMyonen.pdf


TeXOptions = -lualatex \
			 -interaction=nonstopmode \
			 -halt-on-error \
			 -output-directory=build
                                                                                
build/KosmMyonen.pdf: FORCE | build
	latexmk $(TeXOptions) KosmMyonen.tex
	
FORCE:

build:
	mkdir -p build/

clean:
	rm -rf build