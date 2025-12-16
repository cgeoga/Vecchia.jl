
struct CondLogLikBuf{D,T}
  buf_pp::Matrix{T}
  buf_cp::Matrix{T}
  buf_cc::Matrix{T}
  buf_cdat::Matrix{T}
  buf_mdat::Matrix{T}
  buf_cpts::Vector{SVector{D,Float64}}
end

function cnllbuf(::Val{D}, ::Val{Z}, ndata, cpts_sz, pts_sz) where{D,Z}
  buf_pp = Array{Z}(undef,  pts_sz,  pts_sz)
  buf_cp = Array{Z}(undef, cpts_sz,  pts_sz)
  buf_cc = Array{Z}(undef, cpts_sz, cpts_sz)
  buf_cdat = Array{Z}(undef, cpts_sz, ndata)
  buf_mdat = Array{Z}(undef,  pts_sz, ndata)
  buf_cpts = Array{SVector{D,Float64}}(undef, cpts_sz)
  CondLogLikBuf{D,Z}(buf_pp, buf_cp, buf_cc, buf_cdat, buf_mdat, buf_cpts)
end

function negloglik(U::UpperTriangular, y_mut_allowed::AbstractMatrix{T}) where{T}
  ldiv!(adjoint(U), y_mut_allowed)
  (2*logdet(U), sum(_square, y_mut_allowed))
end

function nll(V::VecchiaApproximation{D,F}, params::AbstractVector{T}) where{D,F,T}
  n       = length(V.pts)
  chunks  = collect(Iterators.partition(1:n, cld(n, Threads.nthreads())))
  works   = [cnllbuf(Val(D), Val(T), size(first(V.data), 2),
                     chunksize(V)*blockrank(V), chunksize(V))
             for _ in 1:length(chunks)]
  logdets   = zeros(eltype(params), length(chunks))
  qforms    = zeros(eltype(params), length(chunks))
  blas_nthr = BLAS.get_num_threads()
  BLAS.set_num_threads(1)
  @sync for (j, cj) in enumerate(chunks)
    Threads.@spawn begin
      wj = works[j]
      for k in cj
        (logdetk, qformk) = cnll_str(V, k, wj, params)
        logdets[j] += logdetk
        qforms[j]  += qformk
      end
    end
  end
  BLAS.set_num_threads(blas_nthr)
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
  ldiv!(cov_cc_f.U', cov_cp)
  mul!(cov_pp, adjoint(cov_cp), cov_cp, -one(T), one(T))
  cov_pp_cond = cholesky!(Hermitian(cov_pp))
  # compute the log-likelihood:
  negloglik(cov_pp_cond.U, mdat)
end

