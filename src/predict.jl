
struct VecchiaPredictionDesign{M,P,F}
  n::Int64
  meanfun::M
  kernel::F
  data::Matrix{Float64} # of size n \times k!
  joint_pts::Vector{P}
  joint_condix::Vector{Vector{Int64}}
  pred_perm::Vector{Int64}
end

struct VecchiaPrediction
  invperm::Vector{Int64}
  cmean::Matrix{Float64}
  prediction_rchol::UpperTriangular{Float64, SparseMatrixCSC{Float64,Int64}}
end

default_pred_conditioning(pts) = KNNConditioning(60, Euclidean())

# >= v0.12: updating this internal routine to just have one conditioning argument. 
function VecchiaPredictionDesign(va::SingletonVecchiaApproximation{M,P,F},
                                 pred_pts::Vector{P};
                                 ordering=default_ordering(pred_pts),
                                 conditioning=default_pred_conditioning(pred_pts)) where{M,P,F}
  isnothing(va.data) && throw(error("For prediction problems, you must provide data."))
  # re-order the points according to the requested ordering.
  (pred_perm, pred_pts_perm, _) = permute_points_and_data(pred_pts, nothing, ordering)
  # join va.pts with the new prediction points and provide new prediction sets.
  joint_pts    = vcat(va.pts, pred_pts_perm)
  joint_condix = conditioningsets(joint_pts, conditioning)
  # return the final prediction object.
  VecchiaPredictionDesign(length(va.pts), va.meanfun, va.kernel, va.data, 
                          joint_pts, joint_condix, pred_perm)
end


"""
`predict(appx, pred_pts, params; ordering, conditioning)`

Using the approximate sparse precision matrix induced by `appx::VecchiaApproximation`, predict your process at unobserved locations `pred_pts` using parameters `params` for your covariance function.

Arguments are given as:
- `appx::VecchiaApproximation`: your `VecchiaApproximation` object specifying the data model.
- `pred_pts::Vector{SVector{D,Float64}}`: the points at which you want to predict/simulate.
- `params`: the parameters for the covariance function in `appx`.

Keyword arguments are given as:
- `ordering::PointEnumeration=default_ordering(pred_pts)`: the way you want the prediction points to be ordered. The default in 1D is `Sorted1D()`, and in 2+D is `RandomOrdering()`.
- `conditioning::ConditioningSetDesign=default_pred_data_conditioning(pred_pts)`: The size of the conditioning sets for the prediction problem. In general, this will be larger than what is used for parameter estimation, and will default to `k=60` nearest neighbors.
"""
function predict(va::SingletonVecchiaApproximation{M,P,F},
                 pred_pts::Vector{P}, 
                 params;
                 ordering=default_ordering(pred_pts),
                 conditioning=default_pred_conditioning(pred_pts)) where{M,P,F}
  vp = VecchiaPredictionDesign(va, pred_pts; ordering, conditioning)
  predict(vp, params)
end

function predict(va::ChunkedVecchiaApproximation{M,D,F},
                 args...; kwargs...) where{M,D,F}
  throw(error("Prediction with `ChunkedVecchiaApproximation`s is not implemented, sorry. It wouldn't be hard, though, so please open an issue if you need that functionality."))
end

function predict(vp::VecchiaPredictionDesign{M,P,F}, cov_params, mean_params) where{M,P,F}
  jva = SingletonVecchiaApproximation(vp.meanfun, vp.kernel, [NaN;;], 
                                      vp.joint_pts, vp.joint_condix, Int64[])
  ixs     = (vp.n+1):length(vp.joint_pts)
  U       = rchol(jva, cov_params).U.data
  U_cross = U[(1:vp.n),ixs]
  U_pred  = UpperTriangular(U[ixs, ixs])
  mu      = [jva.meanfun(x, mean_params) for x in jva.pts]
  z       = -(U_cross'*(vp.data - mu[1:vp.n]))
  cmean   = U_pred'\z + mu[(vp.n+1):end]
  VecchiaPrediction(invperm(vp.pred_perm), cmean, U_pred)
end

predict(vp::VecchiaPredictionDesign, params) = predict(vp, params, params)

function predict(vp::VecchiaPredictionDesign, params::Parameters)
  predict(vp, params.cov_params, params.mean_params)
end

"""
`conditional_mean(vp::VecchiaPrediction)`

Returns the conditional mean from your prediction problem, permuted back to the
order in which data points were provided.
"""
conditional_mean(vp::VecchiaPrediction) = vp.cmean[vp.invperm,:]

"""
`full_conditional_covariance(vp::VecchiaPrediction)`

Returns the full joint conditional covariance matrix for your prediction
problem, permuted back to the order in which data points were provide. Note that
this matrix will in general be dense, so if you have many prediction points this
routine will not be feasible and you will need to work with
`vp.prediction_rchol` and `vp.invperm` directly (please open an issue to discuss
interface design if you are in this situation).
"""
function full_conditional_covariance(vp::VecchiaPrediction)
  U = vp.prediction_rchol
  Symmetric(inv(Matrix(U*U'))[vp.invperm, vp.invperm])
end

"""
`conditional_variances(vp::VecchiaPrediction)`

Returns the conditional variances from your prediction problem, permuted back to
the order in which data points were provided.
"""
function conditional_variances(vp::VecchiaPrediction)
  U = vp.prediction_rchol
  takahashi_diagonal(U)[vp.invperm]
end

"""
`conditional_simulate(vp::VecchiaPrediction; z=randn(size(vp.cmean)))`

Returns conditional simulation(s) from the conditional distribution specified by
`vp`, permuted back to the order in which data points were provided.
"""
function conditional_simulate(vp::VecchiaPrediction;
                              z=randn(size(vp.cmean)))
  sim = vp.cmean .+ vp.prediction_rchol'\z
  sim[vp.invperm,:]
end

