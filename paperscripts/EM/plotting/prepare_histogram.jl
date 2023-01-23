
for stem in ("m10", "m30")

  data  = vec(readdlm("./plotting/data/nll_difs_"*stem*".csv", ','))
  boxes = range(-35.0, 5.0, length=20)
  out   = zeros(Int64, length(boxes)-1)
  tics  = map(x->(x[2]+x[1])/2, zip(boxes, Iterators.drop(boxes, 1)))

  for j in 1:(length(boxes)-1)
    out[j] = length(findall(x->boxes[j] <= x < boxes[j+1], data))
  end

  writedlm("./plotting/data/nll_difs_"*stem*"_hist.csv", hcat(tics, out), ',')

end

