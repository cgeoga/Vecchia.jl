
library("GPvecchia")

# Read in the data:
locs = as.matrix(read.csv("tmp.csv", header=FALSE))

# Create the Vecchia config with vecchia_specify, then write out the maximin
# ordering so that I can be sure I'm using the exact same one in the Julia
# implementation.
spec = vecchia_specify(locs, m=5, cond.yz='SGV', 
                       ordering='maxmin', conditioning='NN')

if(!file.exists("tmp_maximin_order.csv")){
  write.table(spec$ord, "tmp_maximin_order.csv", sep=",",
              row.names=FALSE, col.names=FALSE)
}

