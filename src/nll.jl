
function negloglik(U::UpperTriangular, y_mut_allowed::AbstractMatrix{T}) where{T}
  ldiv!(adjoint(U), y_mut_allowed)
  (2*logdet(U), sum(_square, y_mut_allowed))
end

# see ./structstypes.jl for a def of the struct fields. But since this method is
# the core logic for the nll function, I think it should live here.
function (vp::VecchiaLikelihoodPiece{H,D,F,T})(p) where{H,D,F,T}
  out_logdet = zero(eltype(p))
  out_qforms = zero(eltype(p))
  for j in vp.ixrange
    (ldj, qfj)  = cnll_str(vp.cfg, j, vp.buf, p)
    out_logdet += ldj
    out_qforms += qfj
  end
  (out_logdet, out_qforms)
end

function nll(V::VecchiaConfig{H,D,F}, params::AbstractVector{T}) where{H,D,F,T}
  checkthreads()
  Z      = promote_type(H,T)
  ndata  = size(first(V.data), 2)
  pieces = split_nll_pieces(V, Val(Z), Threads.nthreads())
  (logdets, qforms) = _nll(pieces, params) 
  (logdets*ndata + qforms)/2
end

function _nll(pieces::Vector{VecchiaLikelihoodPiece{H,D,F,T}}, 
              params) where{H,D,F,T}
  logdets = zeros(eltype(params), length(pieces))
  qforms  = zeros(eltype(params), length(pieces))
  @sync for j in eachindex(pieces)
    Threads.@spawn begin
      pj = pieces[j]
      (ldj, qfj) = pj(params)
      logdets[j] = ldj
      qforms[j]  = qfj
    end
  end
  (sum(logdets), sum(qforms))
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

