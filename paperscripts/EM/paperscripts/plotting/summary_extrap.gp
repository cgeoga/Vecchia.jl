
set terminal pngcairo size 1200,550 enhanced font 'Verdana,10'
#set terminal epslatex color blacktext size 17cm,12cm
set datafile separator ","
set autoscale xfix
set autoscale yfix

#set format x '\footnotesize %g'
#set format y '\footnotesize %g'

#set xtics x_0, dx, x_n
#set ytics y_0, dy, y_n

# place arbitrary text with:
# set label ... 

set output "extrap.png"

#set xtics  offset 0.0,0.35
#set ylabel offset 2.0,0.0
#set title  offset 0,-0.75

means="./data/extrapolation_means.csv"
vars="./data/extrapolation_vars.csv"

set multiplot layout 1,2 margins .1,0.9,0.1,0.9 spacing .08,0.1

set title "plug-in mean - true mean      at (-0.0075, -0.0075)"
set ylabel "SGV"
set xlabel "EM"
border=0.15
set xrange[-border:border]
set yrange[-border:border]
set arrow from 0.0,-border to 0.0,border nohead lc 'black' lt 2 dt 3
set arrow from -border,0.0 to border,0.0 nohead lc 'black' lt 2 dt 3
plot means using ($3-$2):($4-$2) with points pointtype 7 lc 'black' notitle
unset arrow

set title "plug-in MSE - true MSE       at (-0.0075, -0.0075)\n{/*0.9 (true MSE=0.88)}"
unset ylabel
border=0.1
set xrange[-border:border]
set yrange[-border:border]
set arrow from 0.0,-border to 0.0,border nohead lc 'black' lt 2 dt 3
set arrow from -border,0.0 to border,0.0 nohead lc 'black' lt 2 dt 3
set arrow from -border,-border to border,border nohead lc 'black' lt 2 dt 3
plot vars  using ($3-$2):($4-$2) with points pointtype 7 lc 'black' notitle
unset arrow

unset multiplot

unset output

