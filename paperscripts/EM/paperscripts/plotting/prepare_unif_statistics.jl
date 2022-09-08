
using StatsFuns

# Note that StatsFuns uses normcdf(mu, sigma, x) syntax.

if (!@isdefined MSES)
  const MEANS = readdlm("./data/centerinterp_means3.csv", ',')
  const MSES  = readdlm("./data/centerinterp_mses3.csv", ',')
end

const SGV_STATS = map(1:50) do j
  StatsFuns.normcdf(MEANS[j,3], MSES[j,3], MEANS[j,1])
end

const EM_STATS = map(1:50) do j
  StatsFuns.normcdf(MEANS[j,2], MSES[j,2], MEANS[j,1])
end

if !isinteractive()
  writedlm("./data/plotdata_centerinterp_means.csv", 
           hcat(abs.(MEANS[:,2] - MEANS[:,1]),
                abs.(MEANS[:,3] - MEANS[:,1])), ',')
  writedlm("./data/plotdata_centerinterp_unifstats.csv", 
           hcat(1:50, sort(EM_STATS), sort(SGV_STATS)), ',')

end
