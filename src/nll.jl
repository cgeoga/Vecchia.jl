
# Non-conditional negative log-likelihood.
function negloglik(kfun, params, pts, vals, w1)
  updatebuf!(w1, pts, pts, kfun, params)
  K   = cholesky!(Symmetric(w1))
  tmp = K.U'\vals # alloc 1, but this function only gets called once per nll.
  (logdet(K), sum(_square, tmp))
end

function negloglik(U::UpperTriangular, y_mut_allowed::Matrix{T}) where{T}
  ldiv!(adjoint(U), y_mut_allowed)
  (2*logdet(U), sum(_square, y_mut_allowed))
end

function nll(V::VecchiaConfig{H,D,F}, params::AbstractVector{T}) where{H,D,F,T}
  checkthreads()
  Z     = promote_type(H,T)
  nthr  = Threads.nthreads()
  ndata = size(V.data[1], 2)
  (logdets, qforms) = _nll(V, params, Val(nthr), Val(Z))
  (logdets*ndata + qforms)/2
end

function _nll(V::VecchiaConfig{H,D,F}, params::AbstractVector{T}, 
              ::Val{N}, ::Val{Z})::Tuple{Z,Z} where{H,D,F,T,N,Z}
  kernel  = V.kernel
  cpts_sz = V.chunksize*V.blockrank
  pts_sz  = V.chunksize
  ndata   = size(V.data[1], 2)
  # pre-allocate all buffers:
  bufs = ntuple(j->cnllbuf(Val(D), Val(Z), ndata, cpts_sz, pts_sz), N)
  # handle the first index base case:
  (ld0, qf0) = negloglik(kernel, params, V.pts[1], V.data[1], bufs[1].buf_pp) 
  # pre-allocate thread-arrays for the logdets and qforms:
  out_logdet  = zeros(Z, N)
  out_qforms  = zeros(Z, N)
  out_logdet[1] = ld0
  out_qforms[1] = qf0
  # Now do the main loop for the rest of the terms, which are all conditional nlls.
  # Note that I'm not just using Threads.@threads for [...] and then getting
  # buffers with bufs[Threads.threadid()], because this has the potential for
  # some soundness issues. Further reading:
  # https://discourse.julialang.org/t/behavior-of-threads-threads-for-loop/76042
  m = cld(length(V.condix)-1, Threads.nthreads())
  @sync for (i, chunk) in enumerate(Iterators.partition(2:length(V.condix), m))
    Threads.@spawn for j in chunk
      tbuf = bufs[i]
      pts  = V.pts[j]
      dat  = V.data[j]
      idxs = V.condix[j]
      (ldj, qfj) = cnll_str(V, idxs, tbuf, pts, dat, params)
      out_logdet[i] += ldj
      out_qforms[i] += qfj
    end
  end
  sum(out_logdet), sum(out_qforms)
end

function cnll_str(V, idxs, strbuf::CondLogLikBuf{D,T}, pts, dat, params) where{D,T}
  # prepare conditioning points:
  cpts  = updateptsbuf!(strbuf.buf_cpts, V.pts,  idxs)
  cdat  = updatedatbuf!(strbuf.buf_cdat, V.data, idxs)
  strbuf.buf_mdat .= dat
  # prepare and fill in the matrix buffers pertaining to the cond.  points:
  cov_pp = view(strbuf.buf_pp, 1:length(pts),  1:length(pts))
  cov_cp = view(strbuf.buf_cp, 1:length(cpts), 1:length(pts))
  cov_cc = view(strbuf.buf_cc, 1:length(cpts), 1:length(cpts))
  updatebuf!(cov_pp,  pts,  pts, V.kernel, params, skipltri=false)
  updatebuf!(cov_cc, cpts, cpts, V.kernel, params, skipltri=true)
  updatebuf!(cov_cp, cpts,  pts, V.kernel, params, skipltri=false)
  # Factorize the covariance matrix for the conditioning points:
  cov_cc_f = cholesky!(Symmetric(cov_cc))
  # Before mutating the cross-covariance buffer, compute y - hat{y}, where
  # hat{y} is the conditional expectation of y given the conditioning data.
  ldiv!(cov_cc_f, cdat)
  mul!(strbuf.buf_mdat, cov_cp', cdat, -one(T), one(T))
  # Now compute the conditional covariance matrix, reusing buffers to cut
  # out any unnecessary allocations.
  ldiv!(cov_cc_f.U', cov_cp)
  mul!(cov_pp, adjoint(cov_cp), cov_cp, -one(T), one(T))
  cov_pp_cond = cholesky!(Symmetric(cov_pp))
  # compute the log-likelihood:
  negloglik(cov_pp_cond.U, strbuf.buf_mdat)
end

function em_ejnll(V::VecchiaConfig{H,D,F}, params::AbstractVector{T},
                  y_minus_z0, presolved_saa_sumsq) where{H,D,F,T}
  # Like with the normal nll function, this section handles the things that
  # create type instability, and then passes them to _nll so that the function
  # barrier means that everything _inside_ _nll, which we want to be fast and
  # multithreaded, is stable and non-allocating.
  checkthreads()
  Z     = promote_type(H,T)
  nthr  = Threads.nthreads()
  ndata = size(y_minus_z0, 2)
  # compute the following terms:
  #   nll(V, z0)
  #   (2M)^{-1} sum_j \norm[2]{U(\params)^T v_j}^2, w/ v_j the pre-solved SAA.
  (logdets, qforms) = _nll(V, params, Val(nthr), Val(Z))
  out  = (logdets*ndata + qforms)/2
  # add on the generic nll for the measurement noise and the trace term. Note
  # that the sum of squares has already been divided by 2M.
  out + (generic_nll(I*params[end], y_minus_z0) + presolved_saa_sumsq/params[end])
end

