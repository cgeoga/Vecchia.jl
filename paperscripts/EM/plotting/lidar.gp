set terminal epslatex color blacktext size 15cm,4cm
set key off
set datafile separator ","
load 'cmocean_balance.pal'
set autoscale xfix
set autoscale yfix

set cbtics format '\footnotesize %g'
set cblabel '\footnotesize label'

set format x '\footnotesize %g'
set format y '\footnotesize %g'

set xlabel  '\footnotesize UTC Time (h)'
set ylabel  '\footnotesize Altitude (km)'
set cblabel '\footnotesize Velocity (m/s)'

set xtics   offset  0,   0.5
set xlabel  offset  0,   0.5
set ylabel  offset  3.2, 0
set ytics   offset  0.5, 0
set cbtics  offset -0.5, 0
set cblabel offset -2.0, 0

set cbrange [-4:4]

set output "lidar_figure.tex"

set multiplot layout 1,1 margins .05,0.89,0.2,0.95 spacing .025,0.025

plot "../data/lidar_highalt_plotting.csv" matrix nonuniform with image

unset multiplot


unset output
