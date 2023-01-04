
include("shared.jl")

function simulate(init_seed, n, true_parms, nsamp)
  bthr    = BLAS.get_num_threads()
  rng     = StableRNG(init_seed)
  points  = rand(rng, SVector{2,Float64}, n)
  nugkern = Vecchia.NuggetKernel(kernel_nonugget)
  K       = Symmetric([nugkern(x,y,true_parms) for x in points, y in points])
  BLAS.set_num_threads(5)
  Kf      = cholesky!(K)
  data    = Kf.L*randn(rng, n, nsamp)
  BLAS.set_num_threads(bthr)
  writedlm("./data/data.csv", hcat(getindex.(points,1), getindex.(points,2), data), ',')
  (pts=points, data=data)
end

# Note that this comes out as a BitArray now to save space. Since all we need is
# the sign (each entry is +1 or -1), we can do that with one bit. And so using a
# BitArray instead of storing them as floats turns 450 MiB into 5 MiB. 
function generate_saa(init_seed, n, m, l)
  rng = StableRNG(init_seed)
  BitArray(sgntobool.(rand(rng, (-1.0, 1.0), n, m, l)))
end

if !isinteractive()
  _NDATA = 15_000
  _NSAMP = 50
  TRUE_PARAMS = [10.0, 0.025, 2.25, 0.25]
  INIT_PARAMS = [05.0, 0.2,   1.00, 0.125]

  if isfile("./data/trueparameters.csv") 
    println("Found true parameter file, skipping writing...")
  else
    writedlm("./data/trueparameters.csv", TRUE_PARAMS)
  end

  if isfile("./data/initparameters.csv") 
    println("Found init parameter file, skipping writing...")
  else
    writedlm("./data/initparameters.csv", INIT_PARAMS)
  end

  if isfile("./data/data.csv") 
    println("Found simulated data file, skipping writing...")
  else
    simulate(1234, _NDATA, TRUE_PARAMS, _NSAMP)
  end

  if isfile("./data/saa.jls") 
    println("Found serialized SAA file, skipping writing...")
  else
    serialize("./data/saa.jls", generate_saa(1234, _NDATA, 72, _NSAMP))
  end

end

