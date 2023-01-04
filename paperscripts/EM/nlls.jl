
using NearestNeighbors, MLStyle
include("shared.jl")

# Not that these matrices are large enough to be annoying, but just for that
# extra boost I give BLAS more threads to work with.
BLAS.set_num_threads(6)

# Read in the points and data, already maximin-permuted:
if !(@isdefined NUGKERNEL)
  const (PTS, DATA_MATRIX) = prepare_dat()
  const TRUE_PARAMS     = vec(readdlm("./data/trueparameters.csv", ','))
  const EM_ESTIMATES    = readdlm("./data/estimates_em.csv", ',')
  const SGV10_ESTIMATES = readdlm("./data/estimates_R_m10.csv",  ',')
  const SGV30_ESTIMATES = readdlm("./data/estimates_R_m30.csv",  ',')
  const BUF             = Array{Float64}(undef, 15_000, 15_000)
  const NUGKERNEL       = Vecchia.NuggetKernel(kernel_nonugget)
end

function threaded_updatebuf!(params)
  Threads.@threads for k in eachindex(PTS)
    @inbounds ptk = PTS[k]
    @inbounds BUF[k,k] = NUGKERNEL(ptk, ptk, params)
    @inbounds for j in 1:(k-1)
      BUF[j,k] = NUGKERNEL(PTS[j], ptk, params)
    end
  end
  nothing
end

# Using all six threads, this takes about ten seconds for each invocation on my
# computer. So three cases * 50 trials gives about 25 minutes total runtime.
function exact_full_nll(j, case)
  # pick out the parameters:
  params = @match case begin
    :EM    => EM_ESTIMATES[:,j]
    :SGV10 => SGV10_ESTIMATES[:,j]
    :SGV30 => SGV30_ESTIMATES[:,j]
    _      => throw(error("Options are :EM, :SGV10, or :SGV30"))
  end
  # pick out the data, being very sure to copy since I'm going to mutate:
  data = hcat(DATA_MATRIX[:,j])
  # update the covariance buffer with the right parameters:
  threaded_updatebuf!(params)
  # factorize the matrix and get the likelihood pieces:
  buf_fact = cholesky!(Symmetric(BUF))
  (ldet, qform) = Vecchia.negloglik(buf_fact.U, data)
  (ldet + qform)/2
end

function generate_nlls()
  diffs_m10 = Array{Float64}(undef, 50)
  diffs_m30 = Array{Float64}(undef, 50)
  for j in 1:50
    println("Evaluating likelihoods for trial $j/50...")
    # get the EM likelihood once:
    em_nll_j    = exact_full_nll(j, :EM)
    sgv10_nll_j = exact_full_nll(j, :SGV10)
    sgv30_nll_j = exact_full_nll(j, :SGV30)
    # save the differences:
    diffs_m10[j] = em_nll_j - sgv10_nll_j
    diffs_m30[j] = em_nll_j - sgv30_nll_j
  end
  #write them to files:
  writedlm("./plotting/data/nll_difs_m10.csv", diffs_m10, ',')
  writedlm("./plotting/data/nll_difs_m30.csv", diffs_m30, ',')
  nothing
end

if !isinteractive()
  if all(isfile, ("./plotting/data/nll_difs_m10.csv", "./plotting/data/nll_difs_m30.csv"))
    println("nll difference files already exist, exiting this script.")
    exit(0)
  end
  generate_nlls()
end

