
set terminal epslatex color blacktext size 18cm,13cm
set datafile separator ","
set autoscale xfix
set autoscale yfix

set format x '\footnotesize %g'
set format y '\footnotesize %g'

set output "centerinterp_poster.tex"

set xtics  offset 0.0,0.35
set ylabel offset 2.0,0.0
set xlabel offset 0.0,0.35
set title  offset 0,-0.75

means="./data/plotdata_centerinterp_means.csv"

set multiplot layout 1,1 margins 0.1,0.9,0.1,0.9 spacing 0,0

set ylabel '\footnotesize SGV'
set xlabel '\footnotesize EM'
border=0.075
set xrange[0.0:border]
set yrange[0.0:border]
set xtics 0.01,0.02,0.07
set ytics 0.01,0.02,0.07
set arrow from 0.0,0.0 to 0.0,border nohead lc 'black' lt 2 dt 3
set arrow from 0.0,0.0 to border,0.0 nohead lc 'black' lt 2 dt 3
set arrow from 0.0,0.0 to border,border nohead lc 'black' lt 2 dt 3
plot means using 1:2 with points pointtype 7 lc 'black' notitle
unset arrow

unset multiplot

unset output
