
#set terminal pngcairo size 1200,550 enhanced font 'Verdana,10'
set terminal epslatex color blacktext size 17cm,7cm
set datafile separator ","
set autoscale xfix
set autoscale yfix

set format x '\footnotesize %g'
set format y '\footnotesize %g'

#set xtics x_0, dx, x_n
#set ytics y_0, dy, y_n

# place arbitrary text with:
# set label ... 

set output "centerinterp.tex"

set xtics  offset 0.0,0.35
set ylabel offset 2.0,0.0
set xlabel offset 0.0,0.35
set title  offset 0,-0.75

means="./data/centerinterp_meandifs.csv"
vars="./data/centerinterp_pluginmses.csv"

set multiplot layout 1,2 margins .1,0.9,0.1,0.9 spacing .08,0.1

set title '$|\hat{z}_c(\hat{\bth}) - \hat{z}_c(\bth_0)|$'
set ylabel "SGV"
set xlabel "EM"
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

set title 'plug-in prediction error (\ref{eq:pluginerr})'
set logscale x
set logscale y
unset ylabel
border=10.0
set xrange[0.285:1.5]
set yrange[0.285:border]
set ytics (0.3, 1.0, 2.0, 3.0, 5.0, 7.0)
set xtics (0.3, 0.5, 0.7, 1.1)
set arrow from 0.285,0.285 to 1.5,1.5 nohead lc 'black' lt 2 dt 3
plot vars  using 3:4 with points pointtype 7 lc 'black' notitle
unset arrow

unset multiplot

unset output

