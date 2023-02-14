
# A Scalable Method to Exploit Screening in Gaussian Process Models with Noise

This folder is a collection of the exact code used to generate the results for
this paper. As of this version (Jan 4, 2023), the code actually uses only
released versions of all packages except for my thin KNITRO wrapper that seems
pointless to register. 

To run this code and recreate the results from the paper, all you need to do is
run `bash run.sh`, although you might want to modify things like the number of
threads being used for the compute-intensive Julia commands. Here are the
requirements to reproduce:

- Julia 1.7+, and define your `JULIA_HOME` variable, that points to your Julia
binary. I would highly suggest getting the latest Julia you can, though, and
if you already have it but don't use it often please actually check that you
have the latest version.

- R, and within R the package `GPvecchia`, which is available in CRAN.

- gnuplot

- latex + tikz, if you want to compile the figures into a document and view
them with labeled ticks and stuff. The figures for this paper are made with
gnuplot on the epslatex terminal. Which is very cool, because it makes just the
figure in eps and then uses tikz to put all the marks in, so you can use latex
(and even your own commands!) for writing labels and stuff. But on the flip
side, that of course comes at the cost of a learning curve. 

- **(Optional)** Artelys KNITRO, which is the best nonlinear optimizer money can
buy. Emphasis on buy, though, because you do have to buy it and it is
expensive. There is a six month trial version, though, that you can get, and
you're restricted to 300 unknowns and constraints. So that version works fine
for recreating this code.  But it isn't trivial to get it working, and so if you
want to try to do that and aren't familiar with this process of connecting
software packages maybe open up an issue or email me or something. *I have
confirmed that the generic trust region optimizer that I have baked in
`Vecchia.jl` gives identical answers in this code, though, so you don't have to
deal with this unless you really really want to.*


As a general word of warning, **This code suite in total takes at least 6 hours
on my desktop workstation with an i5-11600K CPU (that can chug power from a wall
outlet!). And so if you're running this on a laptop, you should expect it to
take WAY longer than that.** If you just want to "spot check" the code, so to
speak, shoot me an email/open an issue/call me/whatever. I've designed the code
in such a way that it isn't too much of a pain to just estimate parameters for
one of the 50 trials, for example, just to get a sense of how things work and
stuff.

CG (Jan 2023)

