
function posterior_cov(vc::VecchiaConfig{H,D,F}, params,
                       pred_pts::Vector{SVector{D,Float64}};
                       ncondition=maximum(length, vc.condix)) where{H,D,F}
  # get the given data points and make a tree for fast conditioning set
  # collection.
  (n, m) = (sum(length, vc.pts), length(pred_pts))
  pts    = reduce(vcat, vc.pts)
  # create the new conditioning set elements for the joint configuration of
  # given and prediction points.
  jcondix = copy(vc.condix)
  sizehint!(jcondix, length(vc.condix) + length(pred_pts)) # could also pre-allocate
  for k in eachindex(pred_pts)
    tree       = KDTree(vcat(pts, pred_pts[1:(k-1)]))
    k_cond_ixs = NearestNeighbors.knn(tree, pred_pts[k], min(ncondition, n+(k-1)))[1]
    sort!(k_cond_ixs)
    push!(jcondix, k_cond_ixs)
  end
  # create the augmented/joint point list (for now, just singleton predictions):
  jpts  = vcat(vc.pts, [[x] for x in pred_pts])
  # create the final joint config object, not actually using any data.
  jcfg  = VecchiaConfig(vc.kernel, [fill(NaN, (1,1)) for _ in 1:length(jpts)], jpts, jcondix)
  # for getting the necessary blocks of the actual covariance matrix, consider
  # the following. If Σ^{-1} = U U^T, then Σ v = U^{-T} (U^{-1} v). And to get
  # the Vecchia-implied marginal covariance of the points we are predicting at
  # and the cross covariance between those points and the observed data, we just
  # need to compute Σ [e_{n+1}, ..., e_{n+m}], where e_j is the coordinate
  # vector of zeros except e_j[j] = 1.0, n is the number of observed points, and
  # m is the number of prediction points.
  Us     = UpperTriangular(sparse(rchol(jcfg, params)))
  em     = zeros(m+n, m)
  foreach(j->(em[n+j,j] = 1.0), 1:m)
  cols   = adjoint(Us)\(Us\em)
  pred_pts_marginal = cols[(n+1):(n+m), :]
  pred_pts_cross    = cols[1:n, :]
  # finally, to get the inverse of the covariance of the data we have observed,
  # that can be applied with just the [1:n, 1:n] block of Us.
  Usnn = UpperTriangular(Us[1:n, 1:n])
  solved_cross = Usnn*(Usnn'*pred_pts_cross) # Σ_{data}^{-1} Σ_{cross}
  cond_mean    = solved_cross'*reduce(vcat, vc.data)
  cond_var     = pred_pts_marginal - solved_cross'pred_pts_cross
  # return:
  (;cond_mean, cond_var)
end

