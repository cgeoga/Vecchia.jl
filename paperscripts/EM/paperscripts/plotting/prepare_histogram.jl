
#=
const data  = vec(readdlm("./data/sgv_nll_diffs.csv", ','))
const boxes = range(-50.0, 6.0, length=28)
const out   = zeros(Int64, length(boxes)-1)
const tics  = map(x->(x[2]+x[1])/2, zip(boxes, Iterators.drop(boxes, 1)))

for j in 1:(length(boxes)-1)
  out[j] = length(findall(x->boxes[j] <= x < boxes[j+1], data))
end

writedlm("./data/sgv_nll_diffs_hist.csv", hcat(tics, out), ',')
=#

const data  = vec(readdlm("./data/nll_difs_sgvm30.csv", ','))
const boxes = range(-30.0, 10.0, length=20)
const out   = zeros(Int64, length(boxes)-1)
const tics  = map(x->(x[2]+x[1])/2, zip(boxes, Iterators.drop(boxes, 1)))

for j in 1:(length(boxes)-1)
  out[j] = length(findall(x->boxes[j] <= x < boxes[j+1], data))
end

writedlm("./data/sgv_difs_sgvm30_hist.csv", hcat(tics, out), ',')
