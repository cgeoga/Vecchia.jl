
using NearestNeighbors
include("shared.jl")

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
  const _CEN = @SVector [0.5, 0.5]
  const _EPV = [SVector{2,Float64}(_CEN[1]-0.0075, _CEN[2]-0.0075),
                SVector{2,Float64}(_CEN[1]+0.0075, _CEN[2]-0.0075),
                SVector{2,Float64}(_CEN[1]-0.0075, _CEN[2]+0.0075),
                SVector{2,Float64}(_CEN[1]+0.0075, _CEN[2]+0.0075)]
  const (IDXS_SUB, PTS_SUB) = prepare_nn(_CEN, PTS)
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
  cross_buf = Matrix{Float64}(undef, length(PTS_SUB), length(_EPV))
  @inbounds for j in 1:length(PTS_SUB)
    cross_buf[j,1] = kernel_nonugget(_EPV[1], PTS_SUB[j], params)
    cross_buf[j,2] = kernel_nonugget(_EPV[2], PTS_SUB[j], params)
    cross_buf[j,3] = kernel_nonugget(_EPV[3], PTS_SUB[j], params)
    cross_buf[j,4] = kernel_nonugget(_EPV[4], PTS_SUB[j], params)
  end
  # marginal var for the pred point:
  marg_var = Symmetric([kernel_nonugget(x, y, params) for x in _EPV, y in _EPV] + I*params[end])
  (cond=Symmetric(buf), cross=cross_buf, marg=marg_var)
end

function prediction_and_mse(j, params)
  bufs = assemble_covmat_pieces(params)
  cond_chol = cholesky!(bufs.cond)
  cond_pred = bufs.cross'*(cond_chol\DATA_MATRIX[IDXS_SUB, j])
  cond_var  = Symmetric(bufs.marg - bufs.cross'*(cond_chol\bufs.cross))
  (mean=cond_pred, var=cond_var)
end

# Each row here will be (true value, EM, SGV).
function itp_results_for_output()
  res = map(1:50) do j
    println("Running trial $j/50...")
    truth = prediction_and_mse(j, TRUE_PARAMS)
    EM    = prediction_and_mse(j, EM_ESTIMATES[:,j])
    SGV   = prediction_and_mse(j, R_ESTIMATES[:,j])
    (;truth, EM, SGV)
  end
  serialize("./data/interp_uq_results.jls", res)
end

if !isinteractive()
  itp_results_for_output()
end

