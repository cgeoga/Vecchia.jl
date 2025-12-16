
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

struct VecchiaLikelihoodPiece{D,F,T}
  cfg::VecchiaApproximation{D,F}
  buf::CondLogLikBuf{D,T}
  ixrange::UnitRange{Int64}
end

# TODO (cg 2025/12/14 15:26): the piece evaluation framework here was formulated
# to try and faciliate parallel reverse-mode differentiation because each of the
# pieces would yield a single-threaded routine to make a ReverseDiff tape for.
# But it's been years and I haven't done that yet, so it's probably time.
struct PieceEvaluation{D,F,T} <: Function
  piece::VecchiaLikelihoodPiece{D,F,T}
end

function (c::PieceEvaluation{D,F,T})(p) where{D,F,T}
  (logdets, qforms) = c.piece(p)
  ndata = size(first(c.piece.cfg.data), 2)
  (ndata*logdets + qforms)/2
end

function negloglik(U::UpperTriangular, y_mut_allowed::AbstractMatrix{T}) where{T}
  ldiv!(adjoint(U), y_mut_allowed)
  (2*logdet(U), sum(_square, y_mut_allowed))
end

# see ./structstypes.jl for a def of the struct fields. But since this method is
# the core logic for the nll function, I think it should live here.
function (vp::VecchiaLikelihoodPiece{D,F,T})(p) where{D,F,T}
  out_logdet = zero(eltype(p))
  out_qforms = zero(eltype(p))
  for j in vp.ixrange
    (ldj, qfj)  = cnll_str(vp.cfg, j, vp.buf, p)
    out_logdet += ldj
    out_qforms += qfj
  end
  (out_logdet, out_qforms)
end

# TODO (cg 2025/12/14 18:52): these type games can be simplified now that H is
# being removed.
function split_nll_pieces(V::VecchiaApproximation{D,F}, ::Val{Z}, m) where{D,F,Z}
  ndata   = size(first(V.data), 2)
  cpts_sz = chunksize(V)*blockrank(V)
  pts_sz  = chunksize(V)
  chunks  = Iterators.partition(eachindex(V.pts), cld(length(V.pts), m))
  map(chunks) do chunk
    local_buf = cnllbuf(Val(D), Val(Z), ndata, cpts_sz, pts_sz)
    VecchiaLikelihoodPiece(V, local_buf, chunk)
  end
end

function nll(V::VecchiaApproximation{D,F}, params::AbstractVector{T}) where{D,F,T}
  ndata  = size(first(V.data), 2)
  pieces = split_nll_pieces(V, Val(T), Threads.nthreads())
  (logdets, qforms) = _nll(pieces, params) 
  (logdets*ndata + qforms)/2
end

(V::VecchiaApproximation{D,F})(p) where{D,F} = nll(V, p)

function _nll(pieces::Vector{VecchiaLikelihoodPiece{D,F,T}}, 
              params) where{D,F,T}
  logdets   = zeros(eltype(params), length(pieces))
  qforms    = zeros(eltype(params), length(pieces))
  blas_nthr = BLAS.get_num_threads()
  BLAS.set_num_threads(1)
  @sync for j in eachindex(pieces)
    Threads.@spawn begin
      pj = pieces[j]
      (ldj, qfj) = pj(params)
      logdets[j] = ldj
      qforms[j]  = qfj
    end
  end
  BLAS.set_num_threads(blas_nthr)
  (sum(logdets), sum(qforms))
end

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

