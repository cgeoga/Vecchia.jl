
# Non-conditional negative log-likelihood.
function negloglik(kfun, params, pts, vals, w1)
  updatebuf!(w1, pts, pts, kfun, params)
  K   = cholesky!(Symmetric(w1))
  tmp = K.U'\vals # alloc 1, but this function only gets called once per nll.
  (logdet(K), sum(x->x^2, tmp))
end

function negloglik(U::UpperTriangular, y_mut_allowed::Matrix{T}) where{T}
  ldiv!(adjoint(U), y_mut_allowed)
  (2*logdet(U), sum(x->x^2, y_mut_allowed))
end

function nll(V::VecchiaConfig{H,D,F}, params::AbstractVector{T}) where{H,D,F,T}
  checkthreads()
  Z = promote_type(H,T)
  nthr = Threads.nthreads()
  _nll(V, params, Val(nthr), Val(Z))
end

# TODO (cg 2022/10/28 14:30): This still makes sort of a lot of allocations
# (~850 with the example cfg used in ../examples/). The @code_warntype is all
# blue, though, so I'm not sure where it's coming from. I feel like the stuff
# inside the for loop should be entirely non-allocating.
function _nll(V::VecchiaConfig{H,D,F}, params::AbstractVector{T}, 
              ::Val{N}, ::Val{Z})::Z where{H,D,F,T,N,Z}
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
  # Now do the main loop for the rest of the terms, which are all conditional nlls:
  Threads.@threads for j in 2:length(V.condix)
    tbuf = bufs[Threads.threadid()]
    pts  = V.pts[j]
    dat  = V.data[j]
    idxs = V.condix[j]
    (ldj, qfj) = cnll_str(V, idxs, tbuf, pts, dat, params)
    out_logdet[Threads.threadid()] += ldj
    out_qforms[Threads.threadid()] += qfj
  end
  (sum(out_logdet)*ndata + sum(out_qforms))/2
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

function nll_floops(V::VecchiaConfig{H,D,F}, params::AbstractVector{T}) where{H,D,F,T}
  checkthreads()
  Z = promote_type(H,T)
  _nll_floops(V, params, Val(Z))
end

function _nll_floops(V::VecchiaConfig{H,D,F}, params::AbstractVector{T}, 
                     ::Val{Z})::Z where{H,D,F,T,Z}
  kernel  = V.kernel
  cpts_sz = V.chunksize*V.blockrank
  pts_sz  = V.chunksize
  ndata   = size(V.data[1], 2)
  # handle the first index base case:
  n0      = length(V.pts[1])
  (ld0, qf0) = negloglik(kernel, params, V.pts[1], V.data[1], zeros(Z, n0, n0))
  # Now do the main loop for the rest of the terms, which are all conditional nlls:
  @floop ThreadedEx() for j in 2:length(V.condix)
    @init tbuf = cnllbuf(Val(D), Val(Z), ndata, cpts_sz, pts_sz)
    pts  = V.pts[j]
    dat  = V.data[j]
    idxs = V.condix[j]
    (ldj, qfj) = cnll_str(V, idxs, tbuf, pts, dat, params)
    @reduce(out_logdet += ldj)
    @reduce(out_qforms += qfj)
  end
  ((out_logdet+ld0)*ndata + sum(out_qforms+qf0))/2
end

