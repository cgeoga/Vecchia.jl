set terminal epslatex color blacktext size 12cm,6cm
#set terminal pngcairo size 1200,800 enhanced font 'Verdana,10'
set datafile separator ","

set format x '\footnotesize %g'
set format y '\footnotesize %g'

set xtics offset 0.0,0.35
set ylabel offset 2.0,0.0
set xlabel offset 0.0,0.5

set ylabel '\footnotesize Frequency (count)'
set xlabel '\footnotesize $\ell_{\bS(\hat{\bth}_{\text{EM}}) + \bR(\hat{\bth}_{\text{EM}})}(\by) - \ell_{\bS(\hat{\bth}_{\text{SGV}}) + \bR(\hat{\bth}_{\text{SGV}})}(\by)$'

#set xtics x_0, dx, x_n
#set ytics y_0, dy, y_n

#set cbrange [0:1]

#file="./data/sgv_nll_diffs_hist.csv"
file="./data/sgv_difs_sgvm30_hist.csv"

set output "sgvm30_nll_hist.tex"

set multiplot layout 1,1 margins 0.1,0.9,0.1,0.9 spacing 0,0

set style fill solid 0.5 border
plot file using 1:2 with boxes lc rgb "black" notitle

unset multiplot

unset output

