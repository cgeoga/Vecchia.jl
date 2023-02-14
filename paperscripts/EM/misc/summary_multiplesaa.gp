
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

PSZ=0.35

set title offset 0.0,-0.5

set output "saasummary.pdf"

stem="./data/saatrial_"

set multiplot layout 2,2 margins .1,0.9,0.15,0.9 spacing .1,.15

tail="_param_1.csv"
set title "Scale (variance)"
plot for [k=1:10] stem.k.tail using 1:2 with linespoints pointsize PSZ pointtype 1 lc rgb "black", \
  for [j=2:6] for [k=1:10] stem.k.tail using 1:j with linespoints pointsize PSZ pointtype j lc rgb "black"


tail="_param_2.csv"
set title "Range"
plot for [k=1:10] stem.k.tail using 1:2 with linespoints pointsize PSZ pointtype 1 lc rgb "black", \
  for [j=2:6] for [k=1:10] stem.k.tail using 1:j with linespoints pointsize PSZ pointtype j lc rgb "black"

set xlabel 'EM iteration number'

tail="_param_3.csv"
set title "Smoothness"
plot for [k=1:10] stem.k.tail using 1:2 with linespoints pointsize PSZ pointtype 1 lc rgb "black", \
  for [j=2:6] for [k=1:10] stem.k.tail using 1:j with linespoints pointsize PSZ pointtype j lc rgb "black"


tail="_param_4.csv"
set title "Nugget (variance)"
plot for [k=1:10] stem.k.tail using 1:2 with linespoints pointsize PSZ pointtype 1 lc rgb "black", \
  for [j=2:6] for [k=1:10] stem.k.tail using 1:j with linespoints pointsize PSZ pointtype j lc rgb "black"

unset multiplot

unset output

