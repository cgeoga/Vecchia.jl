
include("shared.jl")

function simulate(init_seed, n, true_parms, nsamp)
  bthr    = BLAS.get_num_threads()
  rng     = StableRNG(init_seed)
  points  = rand(rng, SVector{2,Float64}, n)
  nugkern = (x,y) -> kernel_nonugget(x,y,true_parms) + Float64(x==y)*true_parms[end]
  K       = Symmetric([nugkern(x,y) for x in points, y in points])
  BLAS.set_num_threads(5)
  Kf      = cholesky!(K)
  data    = Kf.L*randn(rng, n, nsamp)
  BLAS.set_num_threads(bthr)
  writedlm("./data/data.csv", hcat(getindex.(points,1), getindex.(points,2), data), ',')
  (pts=points, data=data)
end

function generate_saa(init_seed, n, m, l)
  rng = StableRNG(init_seed)
  rand(rng, (-1.0, 1.0), n, m, l)
end

if !isinteractive()
  const _NDATA = 15_000
  const _NSAMP = 50
  const TRUE_PARAMS = [10.0, 0.025, 2.25, 0.25]
  const INIT_PARAMS = [05.0, 0.2,   1.00, 0.125]
  simulate(1234, _NDATA, TRUE_PARAMS, _NSAMP)
  writedlm("./data/trueparameters.csv", TRUE_PARAMS)
  writedlm("./data/initparameters.csv", INIT_PARAMS)
  serialize("./data/saa.jls", generate_saa(1234, _NDATA, 72, _NSAMP))
end

