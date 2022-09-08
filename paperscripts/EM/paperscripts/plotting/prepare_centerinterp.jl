
const CENTER_MEANS = readdlm("./data/centerinterp_means.csv", ',')
const CENTER_VARS  = readdlm("./data/centerinterp_vars.csv", ',')

const meandifs = map(1:50) do j
  (idx, true_mean, em_mean, sgv_mean) = CENTER_MEANS[j,:]
  (abs(true_mean-em_mean), abs(true_mean-sgv_mean))
end

const pluginuqvars = map(1:50) do j
  (idx, true_var, em_var, sgv_var) = CENTER_VARS[j,:]
  (meandifs[j][1]^2 + em_var, meandifs[j][2]^2 + em_var)
end

writedlm("./data/centerinterp_meandifs.csv",   meandifs,     ',')
writedlm("./data/centerinterp_pluginvars.csv", pluginuqvars, ',')

