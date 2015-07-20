# Name of the report
TEXFILE = report

# Figures to be included in report
FIGURES = normal_hist.pdf

# Report depends on this data
DATAFILE = normal.csv

# Summary tables
TABLES = 

# Run pdflatex twice so that the figure references are correct.
# $< references the first prerequisite
$(TEXFILE).pdf : $(TEXFILE).tex $(FIGURES)
	pdflatex $<
	pdflatex $<

$(DATAFILE) : normal.py
	python $<

$(FIGURES) $(TABLES) : analysis.py
	python $<

view :
	open $(TEXFILE).pdf

clean :
	rm *.aux
	rm *.log
	rm *.out
	rm *.Rout

#$(FIGURES) : normal.R
#R CMD BATCH --vanilla $<
