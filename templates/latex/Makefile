# Name of the report
TEXFILE = report

# Figures to be included in report
FIGURES = normal.pdf

# Report depends on this data
DATAFILE = normal.csv

# Summary tables
TABLES = 

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
