# Pain_discrimination
Pain Learning Lab Pain_discrimination Project 

To run analysis, go to VENV folder and run Analyses_USE THIS after changing BIDSROOT

By default, the analysis will run with an exclude_placebo = 0, meaning that every participant who reports any % of diminishing pain perception during TENS active trials will be included.
The main goals and research hypothesis can be verified using this: exclude_placebo = 0. However, to run a complete check on these hypotheses, run the code twice, once with = 0 and once with = 10.

As of now, the results do not differ between these analysis choices; we therefore keep the = 0 to include a higher number of participants and up our statistical power 
