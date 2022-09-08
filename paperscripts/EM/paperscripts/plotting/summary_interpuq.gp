
set terminal pngcairo size 1200,550 enhanced font 'Verdana,10'
#set terminal epslatex color blacktext size 17cm,12cm
set datafile separator ","
set autoscale xfix
set autoscale yfix

#set format x '\footnotesize %g'
#set format y '\footnotesize %g'

#set xtics x_0, dx, x_n
#set ytics y_0, dy, y_n

set output "interp.png"

#set xtics  offset 0.0,0.35
#set ylabel offset 2.0,0.0
#set title  offset 0,-0.75

means="./data/interp_means.csv"
vars="./data/interp_vars.csv"

set multiplot layout 1,2 margins .1,0.9,0.1,0.9 spacing .08,0.1

set title "||mean_{plug-in} - mean_{true}||_{inf}  for dense center grid \n Dense grid is (0.5+-0.0075, 0.5+-0.0075)"
set ylabel "SGV"
set xlabel "EM"
border=0.1
set xrange[0.0:border]
set yrange[0.0:border]
set arrow from 0,0 to border,border nohead lc 'black' lt 2 dt 3
plot means using 1:2 with points pointtype 7 lc 'black' notitle
unset arrow

set title "||Sig_{plug} - Sig_{true}||_{op}  for dense center grid \n Dense grid is (0.5+-0.0075, 0.5+-0.0075)"
unset ylabel
border=0.025
set xrange[0.0:border]
set yrange[0.0:border]
set arrow from 0,0 to border,border nohead lc 'black' lt 2 dt 3
plot vars using 1:2 with points pointtype 7 lc 'black' notitle
unset arrow

unset multiplot

unset output

