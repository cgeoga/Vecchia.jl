
using LinearAlgebra, Serialization

const RESULTS = deserialize("./data/interp_uq_results.jls")

const means = map(RESULTS) do item
  truth = item.truth.mean
  EM    = item.EM.mean
  SGV   = item.SGV.mean
  #maximum(abs, truth-EM), maximum(abs, truth-SGV) 
  abs(truth[2]-EM[2]), abs(truth[2]-SGV[2])
end

const vars = map(RESULTS) do item
  truth = item.truth.var
  EM    = item.EM.var
  SGV   = item.SGV.var
  #opnorm(truth-EM), opnorm(truth-SGV) 
  maximum(abs, diag(truth-EM)), maximum(abs, diag(truth-SGV))
end

function write_results()
  writedlm("interp_vars.csv",  varsv,  ',')
end

if !isinteractive()
  writedlm("./data/interp_means.csv", hcat(getindex.(means, 1), getindex.(means,2)), ',')
  writedlm("./data/interp_vars.csv",  hcat(getindex.(vars, 1),  getindex.(vars,2)),  ',')
end


