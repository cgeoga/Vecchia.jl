
struct VecchiaPrediction{D,F}
  n::Int64
  kernel::F
  data::Matrix{Float64}
  joint_pts::Vector{SVector{D,Float64}}
  joint_condix::Vector{Vector{Int64}}
  pred_perm::Vector{Int64}
end

function default_pred_conditioning(pts)
  KNNConditioning(30, Euclidean())
end

function check_prediction_compatibility(va::VecchiaApproximation)
  if !all(x->isone(length(x)), va.pts)
    throw(errror("For now, prediction problems require singleton prediction sets."))
  end
  isnothing(va.data) && throw(error("For prediction problems, you must provide data."))
  nothing
end

function VecchiaPrediction(va::VecchiaApproximation{D,F},
                           pred_pts::Vector{SVector{D,Float64}};
                           ordering=default_ordering(pred_pts),
                           data_conditioning=default_pred_conditioning(pred_pts),
                           pred_conditioning=default_pred_conditioning(pred_pts)) where{D,F}
  # check compatibility of the VecchiaApproximation. Most of the restrictions
  # here are temporary and just need implementation to lift.
  check_prediction_compatibility(va)
  # re-make the tree for the given data, and prepare the data conditioning
  # points. This tree doesn't have to be dynamic, so this is fast.
  pts  = reduce(vcat, va.pts)
  tree = HierarchicalNSW(pts)
  add_to_graph!(tree)
  # re-order the points according to the requested ordering.
  (pred_perm, pred_pts_perm, _) = permute_points_and_data(pred_pts, nothing, ordering)
  # compute the cdata conditioning sets.
  data_condix = [knn_search(tree, x, data_conditioning.k)[1] for x in pred_pts_perm]
  # create the conditioning set for the prediction points (and then offset them).
  pred_condix = conditioningsets(pred_pts_perm, pred_conditioning)
  foreach(c -> c .+= length(pts), pred_condix)
  # create the (j)oint conditioning sets and data objects.
  jcondix = vcat(va.condix, 
                 map(c1c2 -> sort(reduce(vcat, c1c2)), zip(data_condix, pred_condix)))
  jdata   = vcat(reduce(vcat, va.data), zeros((length(pred_pts), size(va.data[1], 2))))
  # return the final prediction object.
  VecchiaPrediction(length(pts), va.kernel, jdata, vcat(pts, pred_pts_perm), 
                    jcondix, pred_perm)
end


"""
`predict(appx, pred_pts, params; ordering, data_conditioning, pred_conditioning)`

Using the approximate sparse precision matrix induced by `appx::VecchiaApproximation`, predict your process at unobserved locations `pred_pts` using parameters `params` for your covariance function.

Arguments are given as:
- `appx::VecchiaApproximation`: your `VecchiaApproximation` object specifying the data model.
- `pred_pts::Vector{SVector{D,Float64}}`: the points at which you want to predict/simulate.
- `params`: the parameters for the covariance function in `appx`.

Keyword arguments are given as:
- `conditional_simulate=false`: a flag to indicate whether what is returned should be the conditional mean, or a full conditional simulation.
- `ordering::PointEnumeration=default_ordering(pred_pts)`: the way you want the prediction points to be ordered. The default in 1D is `Sorted1D()`, and in 2+D is `RandomOrdering()`.
- `data_conditioning::ConditioningSetDesign=default_pred_data_conditioning(pred_pts)`: how many of the *observed* data points you want to conditioning on for each prediction point. The default is `KNNConditioning(30, Euclidean())`.
- `pred_conditioning::ConditioningSetDesign=default_pred_data_conditioning(pred_pts)`: how many of the *prediction* points you want to conditioning on for each prediction point. The default is `KNNConditioning(30, Euclidean())`.
"""
function predict(va::VecchiaApproximation{D,F},
                 pred_pts::Vector{SVector{D,Float64}}, 
                 params;
                 conditional_simulate=false,
                 ordering=default_ordering(pred_pts),
                 data_conditioning=default_pred_conditioning(pred_pts),
                 pred_conditioning=default_pred_conditioning(pred_pts)) where{D,F}
  vp = VecchiaPrediction(va, pred_pts; ordering=ordering, 
                         data_conditioning=data_conditioning,
                         pred_conditioning=pred_conditioning)
  predict(vp, params)
end

function predict(vp::VecchiaPrediction{D,F}, params; 
                 conditional_simulate=false) where{D,F}
  jva = ChunkedVecchiaApproximation(vp.kernel, [[NaN;;]], [[x] for x in vp.joint_pts],
                                    vp.joint_condix, Int64[])
  ixs     = (vp.n+1):length(vp.joint_pts)
  U       = rchol(jva, params; use_tiles=true).U
  U_cross = U[(1:vp.n),ixs]
  U_pred  = UpperTriangular(U[ixs, ixs])
  z       = -U_cross'*vp.data[1:vp.n, :]
  conditional_simulate && (z .+= randn(length(z)))
  cmean   = U_pred'\z
  cmean[invperm(vp.pred_perm)] 
end

