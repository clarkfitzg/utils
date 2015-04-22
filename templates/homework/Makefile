# Name of the report
TEXFILE = homework

# variable containing figures to be included in report
FIGURES = normal.pdf

# Run pdflatex twice so that the figure references are correct.
# $< references the first prerequisite
$(TEXFILE).pdf : $(TEXFILE).tex $(FIGURES)
	pdflatex $<
	pdflatex $<

$(FIGURES) : normal.R
	R CMD BATCH --vanilla $<

view :
	open $(TEXFILE).pdf

clean :
	rm *.aux
	rm *.log
	rm *.out
	rm *.Rout
