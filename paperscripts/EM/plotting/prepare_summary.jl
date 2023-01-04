
using DelimitedFiles

const TRUP  = vec(readdlm("./data/trueparameters.csv", ','))
const EM    = readdlm("./data/estimates_em.csv",    ',')
const SGV10 = readdlm("./data/estimates_R_m10.csv", ',')
const SGV30 = readdlm("./data/estimates_R_m30.csv", ',')

for (j, pname) in ((1, "scale"),  (2, "range"), (3, "smooth"), (4, "nug"))
  # get that row of parameters and subtract off the true value:
  EMj    = EM[j,:]    .- TRUP[j]
  SGV10j = SGV10[j,:] .- TRUP[j]
  SGV30j = SGV30[j,:] .- TRUP[j]
  # create the appropriate file names and write each to a file:
  m10name = "./plotting/data/estimates_"*pname*"_m10.csv"
  m30name = "./plotting/data/estimates_"*pname*"_m30.csv"
  writedlm(m10name, hcat(EMj, SGV10j), ',')
  writedlm(m30name, hcat(EMj, SGV30j), ',')
end

