
using Serialization, DelimitedFiles

if !(@isdefined RESULTS)
  const RESULTS   = deserialize("./data/saa_results_multipledata.jls")
  const saa_sizes = sort(collect(keys(RESULTS[1])))
  const param_ixs = Dict(:scale=>1, :range=>2, :smoothness=>3, :nugget=>4)
end

function extract_path(trial, nsaa, parameter)
  getindex.(RESULTS[trial][nsaa].path, param_ixs[parameter])
end

function all_ends(trial, parameter)
  [extract_path(trial, nsaa, parameter)[end] for nsaa in saa_sizes]
end

main_points = 1:10
sub_points  = range(-0.3, 0.3, length=length(saa_sizes))
full_points = reduce(vcat, [j .+ sub_points for j in main_points])

for (p_ix, param) in enumerate((:scale, :range, :smoothness, :nugget))
  fname = string("./data/saatrial_end_", p_ix, ".csv")
  tmp   = reduce(vcat, [all_ends(j, param) for j in 1:length(RESULTS)])
  writedlm(fname, hcat(full_points, tmp), ',')
end

