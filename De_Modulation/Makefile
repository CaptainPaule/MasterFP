all: build/De_Modulation.pdf


TeXOptions = -lualatex \
			 -interaction=nonstopmode \
			 -halt-on-error \
			 -output-directory=build
                                                                                
build/De_Modulation.pdf: FORCE | build
	latexmk $(TeXOptions) De_Modulation.tex
	
FORCE:

build:
	mkdir -p build/

clean:
	rm -rf build