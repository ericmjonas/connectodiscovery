Title: Code
Tags: pelican, publishing
Date: 10/23/80
Slug: code
Authors: Eric Jonas, Konrad Kording
Summary: Short version for index and feeds

We wrote our core inference engine in C++-11. We made heavy use of [Boost Python](http://www.boost.org/doc/libs/1_58_0/libs/python/doc/index.html) to then perform all experiments
and analysis from within Python, via [Continuum's Anaconda Python Distribution](https://store.continuum.io/cshop/anaconda/) on OSX. All large-scale compute jobs were run via [pySpark](https://spark.apache.org/) on [Amazon Web Services](http://aws.amazon.com/). 

There are three repositories associated with this paper:

one with the canonical datasets
<div class="github-card" data-github="ericmjonas/circuitdata" data-width="400" data-height="150" data-theme="default"></div>
<script src="//cdn.jsdelivr.net/github-cards/latest/widget.js"></script>

one with the code of the core inference engine

<div class="github-card" data-github="ericmjonas/netmotifs" data-width="400" data-height="150" data-theme="default"></div>
<script src="//cdn.jsdelivr.net/github-cards/latest/widget.js"></script>

and one with the actual paper text and a lot of the experiment-running infrastructure

<div class="github-card" data-github="ericmjonas/connectodiscovery" data-width="400" data-height="150" data-theme="default"></div>
<script src="//cdn.jsdelivr.net/github-cards/latest/widget.js"></script>

This grew out of us wanting a cleaner, more separate repository for
the code and wanting to be able to reuse the inference engine without
having to download all of the figs, etc. for the paper.

