
using Serialization, DelimitedFiles

if !(@isdefined RESULTS)
  const RESULTS   = deserialize("./data/saa_results_multipledata.jls")
  const saa_sizes = sort(collect(keys(RESULTS[1])))
  const param_ixs = Dict(:scale=>1, :range=>2, :smoothness=>3, :nugget=>4)
end

function extract_path(trial, nsaa, parameter)
  getindex.(RESULTS[trial][nsaa].path, param_ixs[parameter])
end

function all_paths(trial, parameter)
  paths  = [extract_path(trial, nsaa, parameter) for nsaa in saa_sizes]
  maxlen = maximum(length, paths)
  for j in 1:length(saa_sizes)
    pathj = paths[j]
    if length(pathj) < maxlen
      append!(pathj, fill(NaN, maxlen - length(pathj)))
    end
  end
  reduce(hcat, paths)
end

#=
function all_paths_all_trials(parameter)
  trials  = [[extract_path(trial, nsaa, parameter) for nsaa in saa_sizes] 
             for trial in 1:length(RESULTS)]
  maxlens = [maximum(length, trial) for trial in trials]
  maxlen  = maximum(maxlens)
  for j in 1:length(saa_sizes)
    for k in 1:length(RESULTS)
      pathjk = trials[k][j]
      if length(pathjk) < maxlen
        append!(pathjk, fill(NaN, maxlen - length(pathjk)))
      end
    end
  end
  half_reduced = [reduce(hcat, path) for path in trials]
  reduce(hcat, half_reduced)
end
=#

if !isinteractive()
  for (j, param) in enumerate((:scale, :range, :smoothness, :nugget))
    for j in 1:length(RESULTS)
      trial = all_paths(j, param) 
      p_ix  = param_ixs[param]
      fname = string("./data/saatrial_", j, "_param_", p_ix, ".csv")
      writedlm(fname, hcat(1:size(trial, 1), trial), ',')
    end
  end
end

