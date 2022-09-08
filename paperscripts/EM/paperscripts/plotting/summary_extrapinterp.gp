
set terminal pngcairo size 1200,1200 enhanced font 'Verdana,10'
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

set output "extrapinterp.png"


set multiplot layout 2,2 margins .1,0.9,0.1,0.9 spacing .08,0.1

means="./data/extrapolation_means.csv"
vars="./data/extrapolation_vars.csv"

set title "(extrap) plug-in mean - true mean      at (-0.0075, -0.0075)"
set ylabel "SGV"
border=0.15
set xrange[-border:border]
set yrange[-border:border]
set arrow from 0.0,-border to 0.0,border nohead lc 'black' lt 2 dt 3
set arrow from -border,0.0 to border,0.0 nohead lc 'black' lt 2 dt 3
plot means using ($3-$2):($4-$2) with points pointtype 7 lc 'black' notitle
unset arrow

set title "(extrap) plug-in MSE - true MSE       at (-0.0075, -0.0075)\n{/*0.9 (true MSE=0.88)}"
unset ylabel
border=0.1
set xrange[-0.02:border]
set yrange[-0.02:border]
set arrow from 0.0,-0.02 to 0.0,border nohead lc 'black' lt 2 dt 3
set arrow from -0.02,0.0 to border,0.0 nohead lc 'black' lt 2 dt 3
set arrow from -0.02,-0.02 to border,border nohead lc 'black' lt 2 dt 3
plot vars  using ($3-$2):($4-$2) with points pointtype 7 lc 'black' notitle
unset arrow

means="./data/interp_means.csv"
vars="./data/interp_vars.csv"

set title "||mean_{plug-in} - mean_{true}||_{inf}  for dense center grid \n Dense grid is (0.5+-0.0075, 0.5+-0.0075)"
set ylabel "SGV"
set xlabel "EM"
border=0.125
set xrange[0.0:border]
set yrange[0.0:border]
set arrow from 0,0 to border,border nohead lc 'black' lt 2 dt 3
plot means using 1:2 with points pointtype 7 lc 'black' notitle
unset arrow

set title "||Sig_{plug} - Sig_{true}||_{op}  for dense center grid \n Dense grid is (0.5+-0.0075, 0.5+-0.0075)"
unset ylabel
border=0.03
set xrange[0.0:border]
set yrange[0.0:border]
set arrow from 0,0 to border,border nohead lc 'black' lt 2 dt 3
plot vars using 1:2 with points pointtype 7 lc 'black' notitle
unset arrow

unset multiplot

unset output

