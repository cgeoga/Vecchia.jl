#!/bin/bash

echo -e "***********************************************"
echo -e "******** Geoga & Stein EM+Vecchia code ********"
echo -e "***********************************************\n"

if [ ${1:-"dirty"} = "clean" ]; then
  read -p "Are you sure you want to delete all of the files and completely re-generate them? " -r
  echo
  if [[ $REPLY =~ ^[Yy]$ ]]
  then
    echo "Cleaning all files in ./data/..."
    cd ./data/
    rm -f *.csv
    rm -f *.jls
    cd ./tmp/
    rm -f *.csv
    rm -f *.jls
    cd ../../
    echo "done."
  else
    echo "Okay, keeping the data/estimate/etc files."
  fi
fi

# make directories if they don't already exist, and they won't if you just used
# git clone to get this code.
mkdir -p ./data
mkdir -p ./data/tmp
mkdir -p ./plotting/data

echo -e "\nInstantiating Julia Project.toml environment and adding my unregistered KNITRO wrapper (which is not a problem if you don't have KNITRO)..."
$JULIA_HOME/julia --project=Project.toml -e 'using Pkg; Pkg.add(url="https://git.sr.ht/~cgeoga/StandaloneKNITRO.jl")'
$JULIA_HOME/julia --project=Project.toml -e 'using Pkg; Pkg.instantiate()'

echo -e "\nRunning setup script..."
$JULIA_HOME/julia --project=Project.toml -t${2:-6} setup.jl

echo -e "\nRunning R simulation study estimation..."
Rscript ./R/fit.R 2>&1 | tee ./logs/Rfitting.log

echo -e "\n\nRunning Julia simulation study estimation..."
$JULIA_HOME/julia --project=Project.toml -O3 -C"native" -t${2:-6} fit.jl 1 2>&1 | tee ./logs/juliafitting.log

echo -e "\nRunning simulation study interpolation code..."
$JULIA_HOME/julia --project=Project.toml -O3 -C"native" interpolation.jl 

echo -e "\nRunning simulation study exact nll code..."
$JULIA_HOME/julia --project=Project.toml -O3 -C"native" -t${2:-6} nlls.jl 

echo -e "\nRunning code to generate simulation study plots..."
# julia code to do basic manipulations for gnuplot:
$JULIA_HOME/julia ./plotting/prepare_summary.jl
$JULIA_HOME/julia ./plotting/prepare_centerinterp.jl
$JULIA_HOME/julia ./plotting/prepare_histogram.jl 
# gnuplot code to generate figures:
gnuplot summary_estimates.gp
gnuplot summary_interp.gp
gnuplot summary_nlls.gp 

