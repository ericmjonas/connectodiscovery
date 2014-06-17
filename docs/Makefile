
OLDTEX=/usr/local/texlive/2011/bin/universal-darwin/

all: getbib paper.pdf

clean:
	rm paper.pdf paper.bcf paper.blg paper.aux paper.bbl paper.log paper.out paper.run.xml arxiv.pdf arxiv.bcf arxiv.blg arxiv.aux arxiv.bbl arxiv.log


BIBHOME=${HOME}/Documents/
getbib:
	cp "${BIBHOME}/connectomics motif.bib" . 
	cp "${BIBHOME}/connectomics motif-relational models.bib" . 

techreport.pdf: techreport.tex 	
	pdflatex techreport.tex
	bibtex8 techreport
	pdflatex techreport.tex

paper.pdf: paper.tex 	
	pdflatex paper.tex
	biber paper
	pdflatex paper.tex

arxiv.pdf: paper.tex  
	rm -f paper.bbl
	cp paper.tex arxiv.tex
	export PATH=${OLDTEX}:$$PATH; pdflatex arxiv.tex
	export PATH=${OLDTEX}:$$PATH; biber arxiv
	export PATH=${OLDTEX}:$$PATH; pdflatex arxiv.tex

arxiv.tar.gz: arxiv.pdf arxiv.tex
	tar czvf arxiv.tar.gz arxiv.tex arxiv.bbl 00README.XXX `python getgraphics.py paper.tex` mos6502.clusters.pdf

MOUSERETINA_TGT=../experiments/mouseretina/data/retina.1.srm_clist_xsoma.3.1.xyz.1.2.data-fixed_20_100-anneal_slow_1000
MOUSERETINA_OUT=mouseretina/mouseretina_tgt
update_mouseretina:
	cp ${MOUSERETINA_TGT}.latent.pdf ${MOUSERETINA_OUT}.latent.pdf
	cp ${MOUSERETINA_TGT}.somapos.pdf ${MOUSERETINA_OUT}.somapos.pdf
	python svg2ai.py ${MOUSERETINA_TGT}.circos.01.svg ${MOUSERETINA_OUT}.circos.ai ${MOUSERETINA_TGT}.circos.00.small.svg ${MOUSERETINA_OUT}.circos.00.small.ai ${MOUSERETINA_TGT}.circos.01.small.svg ${MOUSERETINA_OUT}.circos.01.small.ai ${MOUSERETINA_TGT}.circos.02.small.svg ${MOUSERETINA_OUT}.circos.02.small.ai 
	cp ${MOUSERETINA_TGT}.z.pdf ${MOUSERETINA_OUT}.z.pdf
dropbox:
	cp -r * ~/Dropbox/jonaskording-connectomics 

debug: paper.pdf
	biber paper | grep "didn't" 
	grep "LaTeX Warning" paper.log

