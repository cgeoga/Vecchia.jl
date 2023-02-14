
set terminal pdf color size 16cm,8cm enhanced font 'Verdana,9'
#set terminal epslatex color blacktext size 16cm,8cm
#set terminal pngcairo size 1200,800 enhanced font 'Verdana,10'
set key off
set datafile separator ","
#load 'cmocean_deep.pal'
set autoscale xfix
set autoscale yfix

set xlabel 'EM iteration number'

unset xlabel

PSZ=0.375

set title offset 0.0,-0.5

set output "saasummary_endsonly.pdf"

stem="./data/saatrial_end_"

set multiplot layout 2,2 margins .1,0.9,0.15,0.9 spacing .1,.15

set xrange [0.25:10.75]
set xtics 1,1,10

tail="1.csv"
set title "Scale (variance)"
YLOW=7.5
YHIH=12
set yrange [YLOW:YHIH]
do for [j=1:9] {
  set arrow from (j+.5),YLOW to (j+.5),YHIH nohead lc rgb 'black' lt 2 dt 3
}
set arrow from 0.25,10 to 10.75,10 nohead lc rgb 'red' lt 2 dt 1
plot stem.tail using 1:2 lc rgb "black" pointtype 7 pointsize PSZ
unset arrow
unset yrange


tail="2.csv"
set title "Range"
YLOW=0.016
YHIH=0.03
set yrange [YLOW:YHIH]
do for [j=1:9] {
  set arrow from (j+.5),YLOW to (j+.5),YHIH nohead lc rgb 'black' lt 2 dt 3
}
set arrow from 0.25,0.025 to 10.75,0.025 nohead lc rgb 'red' lt 2 dt 1
plot stem.tail using 1:2 lc rgb "black" pointtype 7 pointsize PSZ
unset arrow

set xlabel 'Dataset index'

tail="3.csv"
set title "Smoothness"
YLOW=1.9
YHIH=3.2
set yrange [YLOW:YHIH]
do for [j=1:9] {
  set arrow from (j+.5),YLOW to (j+.5),YHIH nohead lc rgb 'black' lt 2 dt 3
}
set arrow from 0.25,2.25 to 10.75,2.25 nohead lc rgb 'red' lt 2 dt 1
plot stem.tail using 1:2 lc rgb "black" pointtype 7 pointsize PSZ
unset arrow


tail="4.csv"
set title "Nugget (variance)"
YLOW=0.242
YHIH=0.256
set yrange [YLOW:YHIH]
do for [j=1:9] {
  set arrow from (j+.5),YLOW to (j+.5),YHIH nohead lc rgb 'black' lt 2 dt 3
}
set arrow from 0.25,0.25 to 10.75,0.25 nohead lc rgb 'red' lt 2 dt 1
plot stem.tail using 1:2 lc rgb "black" pointtype 7 pointsize PSZ

unset multiplot

unset output

