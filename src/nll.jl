
struct ChunkedCondLogLikBuf{D,T}
  buf_pp::Matrix{T}
  buf_cp::Matrix{T}
  buf_cc::Matrix{T}
  buf_cdat::Matrix{T}
  buf_mdat::Matrix{T}
  buf_cpts::Vector{SVector{D,Float64}}
  results::Vector{T} # [1] is logdets, [2] is qforms.
end

function cnllbuf(V::ChunkedVecchiaApproximation{M,D,F}, 
                 params::AbstractVector{T}) where{M,D,F,T}
  ndata    = size(V.data[1], 2)
  pts_sz   = maximum(length, V.pts)
  cpts_sz  = pts_sz*maximum(length, V.condix)
  buf_pp   = Array{T}(undef,  pts_sz,  pts_sz)
  buf_cp   = Array{T}(undef, cpts_sz,  pts_sz)
  buf_cc   = Array{T}(undef, cpts_sz, cpts_sz)
  buf_cdat = Array{T}(undef, cpts_sz, ndata)
  buf_mdat = Array{T}(undef,  pts_sz, ndata)
  buf_cpts = Array{SVector{D,Float64}}(undef, cpts_sz)
  ChunkedCondLogLikBuf(buf_pp, buf_cp, buf_cc, buf_cdat, 
                       buf_mdat, buf_cpts, zeros(T, 2))
end

# unlike in the chunked case, the pre-allocation here is _much_ simpler.
# Allocating for the Kriging weights isn't strictly necessary if you're clever,
# but I think allocating an extra 10 Float64s won't break the bank.
struct SingletonCondLogLikBuf{T}
  buf_cc::Matrix{T}
  buf_cp::Vector{T}
  buf_kwts::Vector{T}
  buf_cres::Vector{T}
  results::Vector{T} # [1] is logdets, [2] is qforms.
end

function cnllbuf(V::SingletonVecchiaApproximation{M,D,F},
                 params::AbstractVector{T}) where{M,D,F,T}
  cpts_sz  = maximum(length, V.condix)
  buf_cc   = Matrix{T}(undef, (cpts_sz, cpts_sz))
  buf_cp   = Vector{T}(undef, cpts_sz)
  buf_kwts = Vector{T}(undef, cpts_sz)
  buf_cres = Vector{T}(undef, cpts_sz)
  SingletonCondLogLikBuf(buf_cc, buf_cp, buf_kwts, buf_cres, zeros(T, 2))
end

function negloglik(U::UpperTriangular, y_mut_allowed::AbstractMatrix{T}) where{T}
  ldiv!(adjoint(U), y_mut_allowed)
  (2*logdet(U), sum(_square, y_mut_allowed))
end

function nll(V, params::AbstractVector{T};
             cov_param_ixs::UnitRange{Int64}=1:length(params),
             mean_param_ixs::UnitRange{Int64}=1:length(params)) where{T}
  n           = length(V.pts)
  chunks      = collect(Iterators.partition(1:n, cld(n, nthreads())))
  works       = [cnllbuf(V, params) for _ in 1:length(chunks)]
  blas_nthr   = BLAS.get_num_threads()
  # by default, the cov and mean params _do_ overlap and are just the whole
  # parameter vector, so that people who want to handle things on their own can
  # do that.
  cov_params  = params[cov_param_ixs]
  mean_params = params[mean_param_ixs]
  BLAS.set_num_threads(1)
  out = _nll(V, cov_params, mean_params, works, chunks)
  BLAS.set_num_threads(blas_nthr)
  out
end

function _nll(V, cov_params::AbstractVector{T},
              mean_params::AbstractVector{T},
              works, chunks) where{T}
  @sync for (j, cj) in enumerate(chunks)
    @spawn foreach(k->cnll_str!(works[j], V, k, cov_params, mean_params), cj)
  end
  total_logdets = sum(x->x.results[1], works)*n_data_samples(V)
  total_qforms  = sum(x->x.results[2], works)
  (total_logdets + total_qforms)/2
end

function (V::VecchiaApproximation{M,D,F})(p; cov_param_ixs=1:length(p),
                                          mean_param_ixs=1:length(p)) where{M,D,F}
  nll(V, p; cov_param_ixs, mean_param_ixs)
end

function cnll_str!(strbuf::ChunkedCondLogLikBuf{D,T}, 
                  V::ChunkedVecchiaApproximation{M,D,F}, j::Int, 
                  cov_params, mean_params)::Nothing where{M,D,F,T}
  # prepare the marginal points and buffer:
  pts    = V.pts[j]
  dat    = V.data[j]
  idxs   = V.condix[j]
  mdat   = view(strbuf.buf_mdat, 1:size(dat,1), :)
  mdat  .= dat
  cov_pp = view(strbuf.buf_pp, 1:length(pts),  1:length(pts))
  updatebuf!(cov_pp,  pts,  pts, V.kernel, cov_params, skipltri=false)
  # subtract off mean from the prediction set data.
  for (k, ptk) in enumerate(pts)
    view(mdat, k, :) .-= V.meanfun(ptk, mean_params)
  end
  # if the conditioning set is empty, just return the marginal nll:
  if isempty(idxs)
    cov_pp_f = cholesky!(Hermitian(cov_pp))
    (logdet, qforms)   = negloglik(cov_pp_f.U, mdat)
    strbuf.results[1] += logdet
    strbuf.results[2] += qforms
    return nothing
  end
  # otherwise, proceed and prepare conditioning points:
  cpts  = updateptsbuf!(strbuf.buf_cpts, V.pts,  idxs)
  cdat  = updatedatbuf!(strbuf.buf_cdat, V.data, idxs)
  # subtract off mean from the prediction set data.
  for (k, ptk) in enumerate(cpts)
    view(cdat, k, :) .-= V.meanfun(ptk, mean_params)
  end
  # prepare and fill in the matrix buffers pertaining to the cond.  points:
  cov_cp = view(strbuf.buf_cp, 1:length(cpts), 1:length(pts))
  cov_cc = view(strbuf.buf_cc, 1:length(cpts), 1:length(cpts))
  updatebuf!(cov_cc, cpts, cpts, V.kernel, cov_params, skipltri=false)
  updatebuf!(cov_cp, cpts,  pts, V.kernel, cov_params, skipltri=false)
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
  # compute the log-likelihood and update the buffer.
  (logdet, qform) = negloglik(cov_pp_cond.U, mdat)
  strbuf.results[1] += logdet
  strbuf.results[2] += qform
  nothing
end

# pulling this functionality out because it can be reused in the simpler rchol
# in the singleton case.
function prepare_conditional!(strbuf::SingletonCondLogLikBuf{T}, j::Int,
                              V::SingletonVecchiaApproximation{M,D,F},
                              cov_params::AbstractVector{T}) where{M,D,F,T}
  ptj   = V.pts[j]
  covjj = V.kernel(ptj, ptj, cov_params)
  idxs  = V.condix[j]
  isempty(idxs) && return covjj
  cpts   = view(V.pts, idxs)
  cov_cc = view(strbuf.buf_cc,   1:length(idxs), 1:length(idxs))
  cov_cp = view(strbuf.buf_cp,   1:length(idxs))
  kwts   = view(strbuf.buf_kwts, 1:length(idxs))
  updatebuf!(cov_cc, cpts, V.kernel, cov_params)
  updatebuf!(cov_cp, cpts, ptj, V.kernel, cov_params)
  cov_cc_f = cholesky!(Symmetric(cov_cc))
  ldiv!(kwts, cov_cc_f, cov_cp)
  covjj - dot(kwts, cov_cp)
end

function cnll_str!(strbuf::SingletonCondLogLikBuf{T},
                  V::SingletonVecchiaApproximation{M,D,F}, j::Int,
                  cov_params::AbstractVector{T},
                  mean_params::AbstractVector{T})::Nothing where{M,D,F,T}
  ndata = size(V.data, 2)
  ptj   = V.pts[j]
  meanj = V.meanfun(ptj, mean_params)
  idxs  = V.condix[j]
  cvar  = prepare_conditional!(strbuf, j, V, cov_params)
  icvar = inv(cvar)
  if isempty(idxs) 
    dataj  = view(V.data, j, :)
    qforms = sum(k->icvar*(V.data[j,k] - meanj)^2, 1:ndata)
    strbuf.results[1] += log(cvar)
    strbuf.results[2] += qforms
    return nothing
  end
  kwts = view(strbuf.buf_kwts, 1:length(idxs))
  qforms  = sum(1:ndata) do k
    cpts  = view(V.pts, idxs)
    cdata = view(V.data, idxs, k)
    cres  = view(strbuf.buf_cres, 1:length(idxs))
    copyto!(cres, cdata)
    for (j, cj) in enumerate(cpts)
      cres[j] -= V.meanfun(cj, mean_params) 
    end
    icvar*((V.data[j,k] - meanj) - dot(cres, kwts))^2
  end
  strbuf.results[1] += log(cvar)
  strbuf.results[2] += qforms
  nothing
end

