
library("GPvecchia")

# Read in the data:
locs_data = read.csv("./data/data.csv", header=FALSE)
locs = as.matrix(locs_data[,1:2])
data = locs_data[,3:dim(locs_data)[2]]

# Read in the inits:
ini  = c(read.csv("./data/initparameters.csv", header=FALSE))$V1
ini[3] = 2.25

# Create the Vecchia config with vecchia_specify, then write out the maximin
# ordering so that I can be sure I'm using the exact same one in the Julia
# implementation.
#  if(!file.exists("./data/maximin_permutation_R.csv")){
#    spec = vecchia_specify(locs, m=10, cond.yz='SGV', 
#                           ordering='maxmin', conditioning='NN')
#    write.table(spec$ord, "./data/maximin_permutation_R_m10.csv", sep=",",
#                row.names=FALSE, col.names=FALSE)
#  }

# Allocate a matrix for the estimates and a vector for the times:
estimates_m10 = matrix(nrow=4, ncol=dim(data)[2])
#estimates_m30 = matrix(nrow=4, ncol=dim(data)[2])
#times_m30     = rep(NA, dim(data)[2])

# Now do the estimation loop if the estimate files don't already exist:
if(!file.exists("./data/estimates_R.csv")){
  for (j in 1:dim(data)[2]){
    cat("Fitting sample ", j, "/", dim(data)[2], "...\n")
    dataj = data[,j]

    tryCatch(expr = {
      res_m10 = vecchia_estimate(dataj, locs, NULL, m=10, theta.ini=ini, 
                                 cond.yz='SGV', ordering='maxmin', conditioning='NN')
      estimates_m10[,j] = res_m10$theta.hat
      # Skipping the m=30 guys for now because they are slow.
      #t1  = Sys.time()
      #res_m30 = vecchia_estimate(dataj, locs, NULL, m=30, theta.ini=ini, 
      #                           cond.yz='SGV', ordering='maxmin', conditioning='NN')
      #t2  = Sys.time()
      #estimates_m30[,j] = res_m30$theta.hat
      #times_m30[j] = as.numeric(t2-t1)
    })
    #cat("Sleeping for a minute...\n")
    #Sys.sleep(60)
  }
  write.table(estimates_m10, "./data/estimates_R_m10_truevinit.csv", sep=",", 
              row.names=FALSE, col.names=FALSE)
  #write.table(estimates_m30, "./data/estimates_R_m30.csv", sep=",", 
  #            row.names=FALSE, col.names=FALSE)
  #write.table(times, "./data/times_R_m10.csv", sep=",", 
  #            row.names=FALSE, col.names=FALSE)
}else{
  print("Estimate file already exists, so skipping the R estimation.")
}

