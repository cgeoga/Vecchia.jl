
#set terminal pngcairo size 1200,800 enhanced font 'Verdana,10'
set terminal epslatex color blacktext size 17cm,12cm
set datafile separator ","
set autoscale xfix
set autoscale yfix

set format x '\footnotesize %g'
set format y '\footnotesize %g'

do for [stem in "m10 m30"] {

scale="./data/estimates_scale_".stem.".csv"
range="./data/estimates_range_".stem.".csv"
smooth="./data/estimates_smooth_".stem.".csv"
nug="./data/estimates_nug_".stem.".csv"

#set xtics x_0, dx, x_n
#set ytics y_0, dy, y_n

# place arbitrary text with:
# set label ... 

set output "./sgv_compare_".stem.".tex"

SCALERANGE=3.0
RANGERANGE=0.023
SMOOTHRANGE=0.9
NUGRANGE=0.031

set xtics offset 0.0,0.35

set title offset 0,-0.75

set multiplot layout 2,2 margins .1,0.9,0.1,0.9 spacing .08,0.1

set ylabel offset 2.0,0.0

set ylabel '\footnotesize SGV'
set xrange[-SCALERANGE:SCALERANGE]
set yrange[-SCALERANGE:SCALERANGE]
set key bottom right
set title '\footnotesize scale $\hat{\sigma}^2 - \sigma_{\text{true}}^2$'
set arrow from 0.0,-SCALERANGE to 0.0,SCALERANGE nohead lc 'black' lt 2 dt 3
set arrow from -SCALERANGE,0.0 to SCALERANGE,0.0 nohead lc 'black' lt 2 dt 3
plot scale  using 1:2 with points pointtype 7 lc rgb "black" notitle
unset arrow

unset ylabel
set xrange[-RANGERANGE:RANGERANGE]
set yrange[-RANGERANGE:RANGERANGE]
set xtics -0.02,0.01,0.02
set title '\footnotesize range $\hat{\rho} - \rho_{\text{true}}$'
set arrow from 0.0,-RANGERANGE to 0.0,RANGERANGE nohead lc 'black' lt 2 dt 3
set arrow from -RANGERANGE,0.0 to RANGERANGE,0.0 nohead lc 'black' lt 2 dt 3
plot range  using 1:2 with points pointtype 7 lc rgb "black" notitle
unset arrow

set xtics auto

set ylabel offset 4.0,0.0

set ylabel '\footnotesize SGV'
set xlabel '\footnotesize EM'
set xrange[-SMOOTHRANGE:SMOOTHRANGE]
set yrange[-SMOOTHRANGE:SMOOTHRANGE]
set key top left
set title '\footnotesize smoothness $\hat{\nu} - \nu_{\text{true}}$'
set arrow from 0.0,-SMOOTHRANGE to 0.0,SMOOTHRANGE nohead lc 'black' lt 2 dt 3
set arrow from -SMOOTHRANGE,0.0 to SMOOTHRANGE,0.0 nohead lc 'black' lt 2 dt 3
plot smooth using 1:2 with points pointtype 7 lc rgb "black" notitle
unset arrow

unset ylabel
set xrange[-NUGRANGE:NUGRANGE]
set yrange[-NUGRANGE:NUGRANGE]
set ytics -0.03, 0.01, 0.03
set xtics -0.03, 0.01, 0.03
set title '\footnotesize nugget $\hat{\eta}^2 - \eta_{\text{true}}^2$'
set arrow from 0.0,-NUGRANGE to 0.0,NUGRANGE nohead lc 'black' lt 2 dt 3
set arrow from -NUGRANGE,0.0 to NUGRANGE,0.0 nohead lc 'black' lt 2 dt 3
plot nug    using 1:2 with points pointtype 7 lc rgb "black" notitle
unset arrow

unset multiplot

unset output

}
