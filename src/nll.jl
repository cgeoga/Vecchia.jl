
# Non-conditional negative log-likelihood.
function negloglik(::Val{S}, kfun, params, pts, vals, w1) where{S}
  if !iszero(S)
    updatebuf_avx!(w1, Val(S), pts, pts, kfun, params)
  else
    updatebuf!(w1, pts, pts, kfun, params)
  end
  K   = cholesky!(Symmetric(w1))
  tmp = K.U'\vals # alloc 1
  0.5*(logdet(K) + sum(abs2, tmp))
end

# A negloglik where a mean and covariance matrix are provided.
function negloglik(K::Cholesky, mu, vals)
  mu .-= vals     # note now that mu is not the mean anymore.
  ldiv!(K.U', mu) # "mu" is now actually K.L\(vals - mu)
  0.5*(logdet(K) + sum(abs2, mu))
end

function negloglik_precision(Omega, mu, vals)
  Omegaf = cholesky(Omega)
  0.5*(-logdet(Omegaf) + dot(vals-mu, Omega*(vals-mu)))
end

# Conditional negative log-likelihood. Organized in such a way that you
# accurately compute the mean and covariance of the conditioned component and
# then pass those values to negloglik.
function cond_negloglik(::Val{S}, kfun, params::AbstractVector{T}, pts, vals, 
                        cond_pts, cond_vals, w1, w2, w3) where{S,D,T}
  # Update the buffers:
  if !iszero(S)
    updatebuf_avx!(w1, Val(S), cond_pts, cond_pts, kfun, params, skipltri=true) 
    updatebuf_avx!(w2, Val(S), cond_pts, pts, kfun, params)
    updatebuf_avx!(w3, Val(S), pts, pts, kfun, params, skipltri=false)
  else
    updatebuf!(w1, cond_pts, cond_pts, kfun, params, skipltri=true) 
    updatebuf!(w2, cond_pts, pts, kfun, params)
    updatebuf!(w3, pts, pts, kfun, params, skipltri=false)
  end
  # Rename the buffers/factorize the first one:
  K_cond_cond = cholesky!(Symmetric(w1)) # cov between cond-pts and cond-pts
  K_pts_condt = w2                       # cov between cond-pts and pts (transp)
  K_pts_pts   = w3                       # cov between pts and pts
  # conditional mean, the one remaining allocation in this function. 
  mu = K_pts_condt'*(K_cond_cond\cond_vals) 
  # Conditional covariance, reusing buffers liberally (see ?mul! in a REPL):
  ldiv!(K_cond_cond.U', K_pts_condt)
  mul!(K_pts_pts, adjoint(K_pts_condt), K_pts_condt, -one(T), one(T))
  sig = cholesky!(Symmetric(K_pts_pts))
  # Regular negloglik with computed conditional covariance and mean:
  negloglik(sig, mu, vals)
end

function nll(V::AbstractVecchiaConfig{D,F}, params::AbstractVector{T};
             execmode=ThreadedEx())::T where{D,F,T}
  chsz   = V.chunksize
  ccsz   = chsz*V.blockrank
  out    = zero(T)
  scalar = (V isa ScalarVecchiaConfig) ? Val(D) : Val(0)
  @floop execmode for j in 1:length(V.condix)
    # Allocate work buffers the correct way:
    @init workcc = Array{T}(undef, ccsz, ccsz)
    @init workcp = Array{T}(undef, ccsz, chsz)
    @init workpp = Array{T}(undef, chsz, chsz)
    # Get the likelihood points, data, and buffer for K(pts, pts) sorted out:
    pts  = V.pts[j]
    dat  = V.data[j]
    if scalar != Val(0)
      buf3 = view(workpp, 1:div(length(pts), D),  1:div(length(pts), D))
    else
      buf3 = view(workpp, 1:length(pts),  1:length(pts))
    end
    # if j==1, return just the regular negloglik for those points:
    if isone(j) 
      termj = negloglik(scalar, V.kernel, params, pts, dat, buf3)
    else
      # Now that we know the conditioning set isn't empty, get the conditioning
      # points sorted out:
      idxs = V.condix[j]
      cpts = reduce(vcat, V.pts[idxs])
      cdat = reduce(vcat, V.data[idxs])
      # Get the buffers loaded up and properly shaped:
      if scalar != Val(0)
        buf1 = view(workcc, 1:div(length(cpts), D), 1:div(length(cpts), D))
        buf2 = view(workcp, 1:div(length(cpts), D), 1:div(length(pts),  D))
      else
        buf1 = view(workcc, 1:length(cpts), 1:length(cpts))
        buf2 = view(workcp, 1:length(cpts), 1:length(pts))
      end
      # Evaluate the conditional nll:
      termj = cond_negloglik(scalar, V.kernel, params, pts, dat, 
                             cpts, cdat, buf1, buf2, buf3)
    end
    @reduce(out  += termj)
  end
  out
end

# for simple debugging and testing.
function exact_nll(V::VecchiaConfig{D,F}, params::Vector{T}) where{D,T,F}
  pts = reduce(vcat, V.pts)
  dat = reduce(vcat, V.data)
  buf = Array{T}(undef, length(pts), length(pts))
  negloglik(V.kernel, params, pts, dat, buf)
end

