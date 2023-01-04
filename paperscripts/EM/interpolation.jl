
using NearestNeighbors
include("shared.jl")

# Not that these matrices are large enough to be annoying, but just for that
# extra boost I give BLAS more threads to work with.
BLAS.set_num_threads(4)

function prepare_nn(epv, pts, k=5_000)
  tre = KDTree(pts)
  (idxs, dists) = knn(tre, epv, k)
  (idxs, pts[idxs])
end

# Read in the points and data, already maximin-permuted:
if !(@isdefined PTS_SUB)
  const (PTS, DATA_MATRIX) = prepare_dat()
  const TRUE_PARAMS  = vec(readdlm("./data/trueparameters.csv", ','))
  const R_ESTIMATES  = readdlm("./data/estimates_R_m10.csv",  ',')
  const EM_ESTIMATES = readdlm("./data/estimates_em.csv", ',')
  const _EPV = @SVector [0.5, 0.5]
  const (IDXS_SUB, PTS_SUB) = prepare_nn(_EPV, PTS)
end

# Written manually here to try and squeeze out some extra performance.
# sub for testing for now.
function assemble_covmat_pieces(params)
  # cov mat for the conditioning points:
  buf = Matrix{Float64}(undef, length(PTS_SUB), length(PTS_SUB))
  @inbounds for k in 1:length(PTS_SUB)
    ptk = PTS_SUB[k]
    buf[k,k] = kernel_nonugget(ptk, ptk, params) + params[end]
    # fill in the triangle column:
    @inbounds for j in 1:(k-1)
      ptj = PTS_SUB[j]
      buf[j,k] = kernel_nonugget(ptj, ptk, params)
    end
  end
  # cross-cov vector:
  cross_buf = Vector{Float64}(undef, length(PTS_SUB))
  @inbounds for j in 1:length(PTS_SUB)
    cross_buf[j] = kernel_nonugget(_EPV, PTS_SUB[j], params)
  end
  # marginal var for the pred point:
  marg_var = kernel_nonugget(_EPV, _EPV, params)
  (cond=Symmetric(buf), cross=cross_buf, marg=marg_var)
end

function center_results_for_output()
  means = Matrix{Float64}(undef, 50, 3)
  vars  = Matrix{Float64}(undef, 50, 3)
  # Compute the pieces for the true law once:
  true_pieces = assemble_covmat_pieces(TRUE_PARAMS)
  S0f  = cholesky(true_pieces.cond)
  slv0 = S0f\true_pieces.cross
  mse0 = true_pieces.marg - dot(true_pieces.cross, slv0)
  for j in 1:50
    println("Running for trial $j/50...")
    em_j   = assemble_covmat_pieces(EM_ESTIMATES[:,j])
    sgv_j  = assemble_covmat_pieces(R_ESTIMATES[:,j])
    Semf   = cholesky(em_j.cond)
    Ssgvf  = cholesky(sgv_j.cond)
    means[j,1] = dot(true_pieces.cross, S0f\DATA_MATRIX[IDXS_SUB,j])
    means[j,2] = dot(em_j.cross,        Semf\DATA_MATRIX[IDXS_SUB,j])
    means[j,3] = dot(sgv_j.cross,       Ssgvf\DATA_MATRIX[IDXS_SUB,j])
    vars[j,1]  = mse0
    vars[j,2]  = em_j.marg  - dot(em_j.cross,  Semf\em_j.cross)
    vars[j,3]  = sgv_j.marg - dot(sgv_j.cross, Ssgvf\sgv_j.cross)
  end
  writedlm("./plotting/data/centerinterp_means.csv", means, ',')
  writedlm("./plotting/data/centerinterp_mses.csv",  vars,  ',')
end

if !isinteractive()
  if isfile("./plotting/data/centerinterp_means.csv") && isfile("./plotting/data/centerinterp_mses.csv")
    println("Interpolation files already exist, exiting this script.")
    exit(0)
  end
  center_results_for_output()
end

