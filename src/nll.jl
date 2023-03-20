
function negloglik(U::UpperTriangular, y_mut_allowed::AbstractMatrix{T}) where{T}
  ldiv!(adjoint(U), y_mut_allowed)
  (2*logdet(U), sum(_square, y_mut_allowed))
end

function nll(V::VecchiaConfig{H,D,F}, params::AbstractVector{T}) where{H,D,F,T}
  checkthreads()
  Z       = promote_type(H,T)
  ndata   = size(V.data[1], 2)
  cpts_sz = V.chunksize*V.blockrank
  pts_sz  = V.chunksize
  nthr    = Threads.nthreads()
  bufs    = allocate_cnll_bufs(nthr, Val(D), Val(Z), ndata, cpts_sz, pts_sz)
  (logdets, qforms) = _nll(V, params, bufs) 
  (logdets*ndata + qforms)/2
end

function _nll(V::VecchiaConfig{H,D,F}, params::AbstractVector{T}, 
              bufs::Vector{CondLogLikBuf{D,Z}})::Tuple{Z,Z} where{H,D,F,T,Z}
  kernel     = V.kernel
  out_logdet = zeros(Z, length(bufs))
  out_qforms = zeros(Z, length(bufs))
  # Note that I'm not just using Threads.@threads for [...] and then getting
  # buffers with bufs[Threads.threadid()], because this has the potential for
  # some soundness issues. Further reading:
  # https://discourse.julialang.org/t/behavior-of-threads-threads-for-loop/76042
  m = cld(length(V.condix), Threads.nthreads())
  @sync for (i, chunk) in enumerate(Iterators.partition(eachindex(V.condix), m))
    tbuf = bufs[i]
    Threads.@spawn for j in chunk
      (ldj, qfj) = cnll_str(V, j, tbuf, params)
      out_logdet[i] += ldj
      out_qforms[i] += qfj
    end
  end
  sum(out_logdet), sum(out_qforms)
end

function cnll_str(V::VecchiaConfig{H,D,F}, j::Int, 
                  strbuf::CondLogLikBuf{D,T}, params) where{H,D,F,T}
  # prepare the marginal points and buffer:
  pts    = V.pts[j]
  dat    = V.data[j]
  idxs   = V.condix[j]
  mdat   = view(strbuf.buf_mdat, 1:size(dat,1), :)
  mdat  .= dat
  cov_pp = view(strbuf.buf_pp, 1:length(pts),  1:length(pts))
  updatebuf!(cov_pp,  pts,  pts, V.kernel, params, skipltri=false)
  # if the conditioning set is empty, just return the marginal nll:
  if isempty(idxs)
    cov_pp_f = cholesky!(Symmetric(cov_pp))
    return negloglik(cov_pp_f.U, mdat)
  end
  # otherwise, proceed and prepare conditioning points:
  cpts  = updateptsbuf!(strbuf.buf_cpts, V.pts,  idxs)
  cdat  = updatedatbuf!(strbuf.buf_cdat, V.data, idxs)
  # prepare and fill in the matrix buffers pertaining to the cond.  points:
  cov_cp = view(strbuf.buf_cp, 1:length(cpts), 1:length(pts))
  cov_cc = view(strbuf.buf_cc, 1:length(cpts), 1:length(cpts))
  updatebuf!(cov_cc, cpts, cpts, V.kernel, params, skipltri=true)
  updatebuf!(cov_cp, cpts,  pts, V.kernel, params, skipltri=false)
  # Factorize the covariance matrix for the conditioning points:
  cov_cc_f = cholesky!(Symmetric(cov_cc))
  # Before mutating the cross-covariance buffer, compute y - hat{y}, where
  # hat{y} is the conditional expectation of y given the conditioning data.
  ldiv!(cov_cc_f, cdat)
  mul!(mdat, cov_cp', cdat, -one(T), one(T))
  # Now compute the conditional covariance matrix, reusing buffers to cut
  # out any unnecessary allocations.
  ldiv!(cov_cc_f.U', cov_cp)
  mul!(cov_pp, adjoint(cov_cp), cov_cp, -one(T), one(T))
  cov_pp_cond = cholesky!(Symmetric(cov_pp))
  # compute the log-likelihood:
  negloglik(cov_pp_cond.U, mdat)
end

