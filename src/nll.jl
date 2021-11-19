
# Non-conditional negative log-likelihood.
function negloglik(kfun, params, pts, vals, w1)
  updatebuf!(w1, pts, pts, kfun, params)
  K   = cholesky!(Symmetric(w1))
  tmp = K.L\vals # alloc 1
  0.5*(logdet(K) + sum(abs2, tmp))
end

# A negloglik where a mean and covariance matrix are provided.
function negloglik(K::Cholesky, mu, vals)
  mu .-= vals    # note now that mu is not the mean anymore.
  ldiv!(K.L, mu) # "mu" is now actually K.L\(vals - mu)
  0.5*(logdet(K) + sum(abs2, mu))
end

function negloglik_precision(Omega, mu, vals)
  Omegaf = cholesky(Omega)
  0.5*(-logdet(Omegaf) + dot(vals-mu, Omega*(vals-mu)))
end

# Conditional negative log-likelihood. Organized in such a way that you
# accurately compute the mean and covariance of the conditioned component and
# then pass those values to negloglik.
function cond_negloglik(kfun, params::Vector{T}, pts, vals, 
                        cond_pts, cond_vals, w1, w2, w3) where{T}
  # Update the buffers:
  updatebuf!(w1, cond_pts, cond_pts, kfun, params, skipltri=true) 
  updatebuf!(w2, cond_pts, pts, kfun, params)
  updatebuf!(w3, pts, pts, kfun, params, skipltri=false)
  # Rename the buffers/factorize the first one:
  K_cond_cond = cholesky!(Symmetric(w1)) # cov between cond-pts and cond-pts
  K_pts_condt = w2                       # cov between cond-pts and pts (transp)
  K_pts_pts   = w3                       # cov between pts and pts
  # conditional mean, the one remaining allocation in this function. 
  mu = K_pts_condt'*(K_cond_cond\cond_vals) 
  # Conditional covariance, reusing buffers liberally (see ?mul! in a REPL):
  ldiv!(K_cond_cond.L, K_pts_condt)
  mul!(K_pts_pts, adjoint(K_pts_condt), K_pts_condt, -one(T), one(T))
  sig = cholesky!(Symmetric(K_pts_pts))
  # Regular negloglik with computed conditional covariance and mean:
  negloglik(sig, mu, vals)
end

function nll(V::VecchiaConfig{D,F}, params::Vector{T};
             execmode=ThreadedEx(), 
             include_ixs::AbstractVector{Int64}=1:length(V.condix)) where{D,F,T}
  # Allocate all the matrix buffers, and as many as there are threads:
  chsz   = V.chunksize
  ccsz   = chsz*V.blockrank
  out    = zero(T)
  @floop execmode for j in 1:length(V.condix)
    @init workcc = Array{T}(undef, ccsz, ccsz)
    @init workcp = Array{T}(undef, ccsz, chsz)
    @init workpp = Array{T}(undef, chsz, chsz)
    # If this isn't one of the included indices, just skip it.
    !in(j, include_ixs) && return zero(T)
    # Get the likelihood points, data, and buffer for K(pts, pts) sorted out:
    pts  = V.pts[j]
    dat  = V.data[j]
    buf3 = view(workpp, 1:length(pts),  1:length(pts))
    # if j==1, return just the regular negloglik for those points:
    if isone(j) 
      termj = negloglik(V.kernel, params, pts, dat, buf3)
    else
      # Now that we know the conditioning set isn't empty, get the conditioning
      # points sorted out:
      idxs = V.condix[j]
      cpts = reduce(vcat, V.pts[idxs])
      cdat = reduce(vcat, V.data[idxs])
      # Get the buffers loaded up and properly shaped:
      buf1 = view(workcc, 1:length(cpts), 1:length(cpts))
      buf2 = view(workcp, 1:length(cpts), 1:length(pts))
      # Evaluate the conditional nll:
      termj = cond_negloglik(V.kernel, params, pts, dat, cpts, cdat, buf1, buf2, buf3)
    end
    @reduce(out  += termj)
  end
  out
end

# for simple debugging and testing.
function exact_nll(V::VecchiaConfig{D,F}, params::Vector{T}) where{D,T,F}
  pts = reduce(vcat, V.pts)
  dat = reduce(vcat, V.dat)
  buf = Array{T}(undef, length(pts), length(pts))
  negloglik(V.kernel, params, pts, dat, buf)
end

