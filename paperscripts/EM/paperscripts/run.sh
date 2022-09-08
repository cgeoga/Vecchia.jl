#!/bin/bash

if [ ${1:-"false"} = "true" ]; then
  echo "Cleaning all files in ./data/..."
  cd ./data/
  rm -f *.csv
  rm -f *.jls
  cd ./tmp/
  rm -f *.csv
  rm -f *.jls
  cd ../../
  echo "done."
fi

#echo "Running setup script..."
$JULIA_HOME/julia --project=Project.toml -t${2:-5} setup.jl

#echo "Running R estimation..."
Rscript ./R/fit.R 2>&1 | tee ./logs/Rfitting_m30.log

#echo "Running Julia estimation..."
for ix in {10..50}
do
  $JULIA_HOME/julia --project=Project.toml -t${2:-5} fit.jl "$ix" 2>&1 | tee -a ./logs/juliafitting.log
  echo "Sleeping for a minute to rest..."
  sleep 60 # to let my pathetic CPU calm down....
done

