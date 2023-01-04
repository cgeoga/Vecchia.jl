
using DelimitedFiles

const CENTER_MEANS = readdlm("./plotting/data/centerinterp_means.csv", ',')

const meandifs = map(1:50) do j
  (true_mean, em_mean, sgv_mean) = CENTER_MEANS[j,:]
  (abs(true_mean-em_mean), abs(true_mean-sgv_mean))
end

writedlm("./plotting/data/centerinterp_meandifs.csv", meandifs, ',')

