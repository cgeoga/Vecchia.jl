
struct VecchiaPrediction{M,D,F}
  n::Int64
  meanfun::M
  kernel::F
  data::Matrix{Float64} # of size n \times k!
  joint_pts::Vector{SVector{D,Float64}}
  joint_condix::Vector{Vector{Int64}}
  pred_perm::Vector{Int64}
end

default_pred_conditioning(pts) = KNNConditioning(60, Euclidean())

# >= v0.12: updating this internal routine to just have one conditioning argument. 
function VecchiaPrediction(va::SingletonVecchiaApproximation{M,D,F},
                           pred_pts::Vector{SVector{D,Float64}};
                           ordering=default_ordering(pred_pts),
                           conditioning=default_pred_conditioning(pred_pts)) where{M,D,F}
  isnothing(va.data) && throw(error("For prediction problems, you must provide data."))
  # re-order the points according to the requested ordering.
  (pred_perm, pred_pts_perm, _) = permute_points_and_data(pred_pts, nothing, ordering)
  # join va.pts with the new prediction points and provide new prediction sets.
  joint_pts    = vcat(va.pts, pred_pts_perm)
  joint_condix = conditioningsets(joint_pts, conditioning)
  # return the final prediction object.
  VecchiaPrediction(length(va.pts), va.meanfun, va.kernel, va.data, 
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
- `conditional_simulate=false`: a flag to indicate whether what is returned should be the conditional mean, or a full conditional simulation.
- `ordering::PointEnumeration=default_ordering(pred_pts)`: the way you want the prediction points to be ordered. The default in 1D is `Sorted1D()`, and in 2+D is `RandomOrdering()`.
- `conditioning::ConditioningSetDesign=default_pred_data_conditioning(pred_pts)`: The size of the conditioning sets for the prediction problem. In general, this will be larger than what is used for parameter estimation, and will default to `k=60` nearest neighbors.
"""
function predict(va::SingletonVecchiaApproximation{M,D,F},
                 pred_pts::Vector{SVector{D,Float64}}, 
                 params;
                 conditional_simulate=false,
                 ordering=default_ordering(pred_pts),
                 conditioning=default_pred_conditioning(pred_pts)) where{M,D,F}
  vp = VecchiaPrediction(va, pred_pts; ordering, conditioning)
  predict(vp, params)
end

function predict(va::ChunkedVecchiaApproximation{M,D,F},
                 args...; kwargs...) where{M,D,F}
  throw(error("Prediction with `ChunkedVecchiaApproximation`s is not implemented, sorry. It wouldn't be hard, though, so please open an issue if you need that functionality."))
end

function predict(vp::VecchiaPrediction{M,D,F}, cov_params, mean_params; 
                 conditional_simulate=false) where{M,D,F}
  jva = SingletonVecchiaApproximation(vp.meanfun, vp.kernel, [NaN;;], 
                                      vp.joint_pts, vp.joint_condix, Int64[])
  ixs     = (vp.n+1):length(vp.joint_pts)
  U       = rchol(jva, cov_params).U
  U_cross = U[(1:vp.n),ixs]
  U_pred  = UpperTriangular(U[ixs, ixs])
  mu      = [jva.meanfun(x, mean_params) for x in jva.pts]
  z       = -(U_cross'*(vp.data - mu[1:vp.n]))
  conditional_simulate && (z .+= randn(length(z)))
  cmean   = U_pred'\z + mu[(vp.n+1):end]
  cmean[invperm(vp.pred_perm)] 
end

function predict(vp::VecchiaPrediction{M,D,F}, params; 
                 conditional_simulate=false) where{M,D,F}
  predict(vp, params, params; conditional_simulate=conditional_simulate)
end

function predict(vp::VecchiaPrediction{M,D,F}, params::Parameters; 
                 conditional_simulate=false) where{M,D,F}
  predict(vp, params.cov_params, params.mean_params; 
          conditional_simulate=conditional_simulate)
end

