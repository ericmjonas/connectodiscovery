


all: getbib techreport.pdf

clean:
	rm techreport.pdf

BIBHOME=${HOME}/Documents/
getbib:
	cp "${BIBHOME}/connectomics motif.bib" . 
	cp "${BIBHOME}/connectomics motif-relational models.bib" . 

techreport.pdf: techreport.tex 	
	pdflatex techreport.tex
	biber techreport
	pdflatex techreport.tex

dropbox:
	cp -r * ~/Dropbox/jonaskording-connectomics 
