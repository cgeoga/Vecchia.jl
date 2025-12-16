
struct CondLogLikBuf{D,T}
  buf_pp::Matrix{T}
  buf_cp::Matrix{T}
  buf_cc::Matrix{T}
  buf_cdat::Matrix{T}
  buf_mdat::Matrix{T}
  buf_cpts::Vector{SVector{D,Float64}}
end

function cnllbuf(V::VecchiaApproximation{D,F}, 
                 params::AbstractVector{T}) where{D,F,T}
  ndata    = size(V.data[1], 2)
  pts_sz   = maximum(length, V.pts)
  cpts_sz  = pts_sz*maximum(length, V.condix)
  buf_pp   = Array{T}(undef,  pts_sz,  pts_sz)
  buf_cp   = Array{T}(undef, cpts_sz,  pts_sz)
  buf_cc   = Array{T}(undef, cpts_sz, cpts_sz)
  buf_cdat = Array{T}(undef, cpts_sz, ndata)
  buf_mdat = Array{T}(undef,  pts_sz, ndata)
  buf_cpts = Array{SVector{D,Float64}}(undef, cpts_sz)
  CondLogLikBuf(buf_pp, buf_cp, buf_cc, buf_cdat, buf_mdat, buf_cpts)
end

function negloglik(U::UpperTriangular, y_mut_allowed::AbstractMatrix{T}) where{T}
  ldiv!(adjoint(U), y_mut_allowed)
  (2*logdet(U), sum(_square, y_mut_allowed))
end

function nll(V::VecchiaApproximation{D,F}, params::AbstractVector{T}) where{D,F,T}
  n         = length(V.pts)
  chunks    = collect(Iterators.partition(1:n, cld(n, Threads.nthreads())))
  works     = [cnllbuf(V, params) for _ in 1:length(chunks)]
  blas_nthr = BLAS.get_num_threads()
  BLAS.set_num_threads(1)
  out = _nll(V, params, works, chunks)
  BLAS.set_num_threads(blas_nthr)
  out
end

function _nll(V::VecchiaApproximation{D,F}, 
              params::AbstractVector{T},
              works, chunks) where{D,F,T}
  logdets = zeros(T, length(chunks))
  qforms  = zeros(T, length(chunks))
  @sync for (j, cj) in enumerate(chunks)
    Threads.@spawn begin
      wj = works[j]
      (_logdets, _qforms) = (0.0, 0.0)
      for k in cj
        (logdetk, qformk) = cnll_str(V, k, wj, params)
        _logdets += logdetk
        _qforms  += qformk
      end
      logdets[j] = _logdets
      qforms[j]  = _qforms
    end
  end
  total_logdets = sum(logdets)*size(first(V.data), 2)
  total_qforms  = sum(qforms)
  (total_logdets + total_qforms)/2
end

(V::VecchiaApproximation{D,F})(p) where{D,F} = nll(V, p)

function cnll_str(V::VecchiaApproximation{D,F}, j::Int, 
                  strbuf::CondLogLikBuf{D,T}, params) where{D,F,T}
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
    cov_pp_f = cholesky!(Hermitian(cov_pp))
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
  cov_cc_f = cholesky!(Hermitian(cov_cc))
  # Before mutating the cross-covariance buffer, compute y - hat{y}, where
  # hat{y} is the conditional expectation of y given the conditioning data.
  ldiv!(cov_cc_f, cdat)
  mul!(mdat, cov_cp', cdat, -one(T), one(T))
  # Now compute the conditional covariance matrix, reusing buffers to cut
  # out any unnecessary allocations.
  #
  # TODO (cg 2025/12/16 10:27): doing this solve column-by-column is necessary
  # to avoid a weird stray allocation here.
  for j in 1:size(cov_cp, 2)
    cj = view(cov_cp, :, j)
    ldiv!(cov_cc_f.U', cj)
  end
  mul!(cov_pp, cov_cp', cov_cp, -one(T), one(T))
  cov_pp_cond = cholesky!(Hermitian(cov_pp))
  # compute the log-likelihood:
  negloglik(cov_pp_cond.U, mdat)
end

