
# TODO (cg 2025/12/15 14:17): 
#
# - generalize the extra methods for AbstractMatrix as well. Just need to do a
# little for loop for the permutations.
struct RCholesky <: AbstractMatrix{Float64}
  U::UpperTriangular{Float64, SparseMatrixCSC{Float64, Int64}}
  p::Vector{Int64}
  ip::Vector{Int64}
  buf::Vector{Float64}
end

function Base.display(rc::RCholesky)
  println("RCholesky factor with: ")
  println("  - size:         $(size(rc))")
  println("  - nzfraction:   $(nnz(rc.U)/prod(size(rc)))")
end

function Base.display(arc::Adjoint{Float64, RCholesky})
  rc = arc.parent
  println("Adjoint RCholesky factor with: ")
  println("  - size:         $(size(rc))")
  println("  - nzfraction:   $(nnz(rc.U)/prod(size(rc)))")
end

Base.size(rc::RCholesky)    = size(rc.U)
Base.size(rc::RCholesky, j) = size(rc.U, j)
Base.eltype(rc::RCholesky)  = Float64

Base.adjoint(rc::RCholesky) = Adjoint{Float64, RCholesky}(rc)

LinearAlgebra.issymmetric(rc::RCholesky) = false
LinearAlgebra.ishermitian(rc::RCholesky) = false

function LinearAlgebra.mul!(buf::AbstractVector, rc::RCholesky, v::AbstractVector)
  mul!(buf, rc.U, v)
  permute!(buf, rc.ip)
end

function Base.:*(rc::RCholesky, v::AbstractVector)
  buf = similar(v)
  mul!(buf, rc, v)
end

function LinearAlgebra.mul!(buf::AbstractVector, 
                            arc::Adjoint{Float64, RCholesky}, 
                            v::AbstractVector)
  rc = arc.parent
  copyto!(rc.buf, v)
  permute!(rc.buf, rc.p)
  mul!(buf, rc.U', rc.buf)
end

function Base.:*(arc::Adjoint{Float64, RCholesky}, v::AbstractVector)
  rc  = arc.parent
  buf = similar(v)
  mul!(buf, arc, v)
end

function LinearAlgebra.ldiv!(rc::RCholesky, v::AbstractVector)
  permute!(v, rc.p)
  ldiv!(rc.U, v)
end

function LinearAlgebra.ldiv!(buf::AbstractVector, rc::RCholesky, 
                             v::AbstractVector)
  copyto!(buf, v)
  ldiv!(rc, buf)
end

function Base.:\(rc::RCholesky, v::AbstractVector)
  buf = similar(v)
  ldiv!(buf, rc, v)
end

function LinearAlgebra.ldiv!(arc::Adjoint{Float64, RCholesky}, 
                             v::AbstractVector)
  rc = arc.parent
  ldiv!(rc.U', v)
  permute!(v, rc.ip)
end

function LinearAlgebra.ldiv!(buf::AbstractVector, 
                             arc::Adjoint{Float64, RCholesky}, 
                             v::AbstractVector)
  copyto!(buf, v)
  ldiv!(arc, buf)
end

function Base.:\(arc::Adjoint{Float64, RCholesky}, v::AbstractVector)
  buf = similar(v)
  ldiv!(buf, arc, v)
end

LinearAlgebra.logdet(rc::RCholesky) = logdet(rc.U)

struct RCholeskyPreconditioner <: AbstractMatrix{Float64}
  U::UpperTriangular{Float64, SparseMatrixCSC{Float64, Int64}}
  p::Vector{Int64}
  ip::Vector{Int64}
  buf::Vector{Float64}
end

function Base.display(rc::RCholeskyPreconditioner)
  println("RCholeskyPreconditioner with: ")
  println("  - size:                          $(size(rc))")
  println("  - (Cholesky factor) nzfraction:  $(nnz(rc.U)/prod(size(rc)))")
end

Base.size(rc::RCholeskyPreconditioner)    = size(rc.U)
Base.size(rc::RCholeskyPreconditioner, j) = size(rc.U, j)
Base.eltype(rc::RCholeskyPreconditioner)  = Float64

function Base.adjoint(rc::RCholeskyPreconditioner) 
  Adjoint{Float64, RCholeskyPreconditioner}(rc)
end

LinearAlgebra.issymmetric(rc::RCholeskyPreconditioner) = true
LinearAlgebra.ishermitian(rc::RCholeskyPreconditioner) = true

function LinearAlgebra.mul!(buf::AbstractVector, 
                            rc::RCholeskyPreconditioner, 
                            v::AbstractVector)
  copyto!(rc.buf, v)
  permute!(rc.buf, rc.p)
  mul!(buf, rc.U', rc.buf)
  mul!(rc.buf, rc.U, buf)
  permute!(rc.buf, rc.ip)
  copyto!(buf, rc.buf)
end

struct RCholeskyStorage
  diagonals::Vector{UpperTriangular{Float64,Matrix{Float64}}}
  odiagonals::Vector{Matrix{Float64}}
  condix::Vector{Vector{Int64}}
  idxs::Vector{UnitRange{Int64}} 
  is_instantiated::Vector{Bool}
end

function RCholApplicationBuffer(U::RCholeskyStorage, ndata::Int64)
  m    = length(U.condix)
  out  = Array{Float64}(undef, maximum(j->size(U.odiagonals[j], 2), 1:m), ndata)
  bufz = Array{Float64}(undef, maximum(j->size(U.odiagonals[j], 2), 1:m), ndata)
  bufv = Array{Float64}(undef, maximum(length, U.condix)*maximum(length, U.idxs), ndata)
  bufm = Array{Float64}(undef, maximum(length, U.idxs), ndata)
  RCholApplicationBuffer(bufv, bufm, bufz, out)
end

struct CondRCholBuf{D}
  buf_pp::Matrix{Float64}
  buf_cp::Matrix{Float64}
  buf_cc::Matrix{Float64}
  buf_cpts::Vector{SVector{D,Float64}}
end

function crcholbuf(::Val{D}, cpts_sz, pts_sz) where{D}
  buf_pp = Array{Float64}(undef,  pts_sz,  pts_sz)
  buf_cp = Array{Float64}(undef, cpts_sz,  pts_sz)
  buf_cc = Array{Float64}(undef, cpts_sz, cpts_sz)
  buf_cpts = Array{SVector{D,Float64}}(undef, cpts_sz)
  CondRCholBuf{D}(buf_pp, buf_cp, buf_cc, buf_cpts)
end

function prepare_diagonal_chunks(sizes)
  map(sizes) do sz
    UpperTriangular{Float64, Matrix{Float64}}(I(sz))
  end
end

function prepare_odiagonal_chunks(condix, sizes)
  map(enumerate(condix)) do (j,cj)
    isempty(cj) ? zeros(0, 0) : zeros(sum(view(sizes, cj)), sizes[j])
  end
end

# Just allocates all the memory for the struct. This does NOT fill in values.
#
# TODO (cg 2025/12/16 11:25): in the chunked case, this produces a huge number
# of allocations because it is a ton of small matrix allocations. For the
# SingletonVecchiaConfig, this can probably be brought down dramatically.
function RCholeskyStorage_alloc(V::ChunkedVecchiaApproximation{M,D,F}) where{M,D,F}
  szs    = map(length, V.pts)
  diags  = prepare_diagonal_chunks(szs)
  odiags = prepare_odiagonal_chunks(V.condix, szs)
  RCholeskyStorage(diags, odiags, V.condix, globalidxs(V.pts), [false]) 
end

function allocate_crchol_bufs(V::ChunkedVecchiaApproximation{M,D,F}, 
                              cpts_sz, pts_sz) where{M,D,F}
  [crcholbuf(Val(D), cpts_sz, pts_sz) for _ in 1:nthreads()]
end

function _rchol_instantiate_index!(strbuf::RCholeskyStorage, 
                                   V::ChunkedVecchiaApproximation{M,D,F},
                                   bufs, i::Int, j::Int, params, tiles) where{M,D,F}
  # get the buffer for this thread:
  tbuf = bufs[i]
  # get the data and points:
  idxs = V.condix[j]
  pts  = V.pts[j]
  cov_pp = view(tbuf.buf_pp, 1:length(pts), 1:length(pts))
  if isempty(idxs)
    # In this special case, I actually can skip the lower triangle. 
    if tiles isa Nothing
      updatebuf!(cov_pp, pts, pts, V.kernel, params, skipltri=true)
    else
      updatebuf_tiles!(cov_pp, tiles, j, j)
    end
    cov_pp_f = cholesky!(Hermitian(cov_pp))
    buf      = strbuf.diagonals[j]
    #ldiv!(cov_pp_f.U, buf)
    for j in 1:size(buf, 2)
      cj = view(buf, :, j)
      ldiv!(cov_pp_f.U, cj)
    end
  else
    # If the set of conditioning points is nonempty, then I'm going to use
    # an in-place mul! on this buffer, so I need to fill it all in.
    if tiles isa Nothing
      updatebuf!(cov_pp, pts, pts, V.kernel, params, skipltri=false)
    else
      updatebuf_tiles!(cov_pp, tiles, j, j)
    end
    # prepare conditioning points:
    cpts = updateptsbuf!(tbuf.buf_cpts, V.pts, idxs)
    # prepare and fill in the matrix buffers pertaining to the cond.  points:
    cov_cp = view(tbuf.buf_cp, 1:length(cpts), 1:length(pts))
    cov_cc = view(tbuf.buf_cc, 1:length(cpts), 1:length(cpts))
    if tiles isa Nothing
      updatebuf!(cov_cc, cpts, cpts, V.kernel, params, skipltri=false)
      updatebuf!(cov_cp, cpts,  pts, V.kernel, params, skipltri=false)
    else
      updatebuf_tiles!(cov_cc, tiles, idxs, idxs)
      updatebuf_tiles!(cov_cp, tiles, idxs, j)
    end
    # pre-factorize the cpts:cpts marginal matrix, then get the two remaining
    # pieces, like the conditional covaraince of the prediction points. I
    # acknowledge that this is a little hard to read, but it really nicely cuts
    # out all the unnecessary allocations. If you do a manual check, you can
    # confirm that cov_pp becomes the conditional covariance of pts | cpts, etc.
    cov_cc_f = cholesky!(Hermitian(cov_cc))
    #ldiv!(cov_cc_f.U', cov_cp)
    for j in 1:size(cov_cp, 2)
      cj = view(cov_cp, :, j)
      ldiv!(cov_cc_f.U', cj)
    end
    mul!(cov_pp, adjoint(cov_cp), cov_cp, -1.0, 1.0)
    #ldiv!(cov_cc_f.U, cov_cp)
    for j in 1:size(cov_cp, 2)
      cj = view(cov_cp, :, j)
      ldiv!(cov_cc_f.U, cj)
    end
    Djf = cholesky!(Hermitian(cov_pp))
    # Update the struct buffers. Note that the diagonal elements are actually
    # UpperTriangular, and I am not supposed to mutate those. But we do the ugly
    # hack of directly working with the data buffer backing the UpperTriangular
    # wrapper, which is probably not really recommended, but it works.
    #strbuf.odiagonals[j] .= Bt_chunks
    strbuf.odiagonals[j] .= cov_cp
    strbuf_Djf = strbuf.diagonals[j].data
    #ldiv!(Djf.U, strbuf_Djf)
    for j in 1:size(strbuf_Djf, 2)
      cj = view(strbuf_Djf, :, j)
      ldiv!(Djf.U, cj)
    end
    # now update the block with the rmul!, being careful to now use the
    # UpperTriangular matrix, NOT the raw buffer!
    strbuf_Bt = strbuf.odiagonals[j]
    strbuf_Bt .*= -1.0
    rmul!(strbuf_Bt, strbuf.diagonals[j])
  end
end

function rchol_instantiate!(strbuf::RCholeskyStorage, 
                            V::ChunkedVecchiaApproximation{M,D,F},
                            params, tiles) where{M,D,F}
  @assert !strbuf.is_instantiated[] "This routine assumes an un-instantiated RCholeskyStorage. Please allocate a new one and try again."
  strbuf.is_instantiated[] = true
  cpts_sz = chunksize(V)*blockrank(V)
  pts_sz  = chunksize(V)
  # allocate three buffers:
  nthr = nthreads()
  bufs = allocate_crchol_bufs(V, cpts_sz, pts_sz)
  # do the main loop:
  m = cld(length(V.condix), nthreads())
  blas_nthr = BLAS.get_num_threads()
  BLAS.set_num_threads(1)
  chunks = collect(Iterators.partition(1:length(V.condix), m))
  @sync for (i, chunk) in enumerate(chunks)
    @spawn for j in chunk
      # confirmed non-allocating.
      _rchol_instantiate_index!(strbuf, V, bufs, i, j, params, tiles)
    end
  end
  BLAS.set_num_threads(blas_nthr)
  strbuf
end

# TODO (cg 2025/12/15 13:16): some smarter way of indexing so that this can be
# parallelized.
function SparseArrays.sparse(U::RCholeskyStorage)
  master_len = _nnz(U.idxs, U.condix)
  Iv = Vector{Int64}(undef, master_len)
  Jv = Vector{Int64}(undef, master_len)
  Vv = Vector{Float64}(undef, master_len)
  master_ix = 1
  # Diagonal blocks. This section makes no allocations and probably can't really
  # be improved upon.
  for (j,ix) in enumerate(U.idxs)
    Dj = U.diagonals[j]
    sz = size(Dj, 1)
    @inbounds offset = ix[1]-1
    for col_ix in 1:sz
      for row_ix in 1:col_ix
        @inbounds Iv[master_ix] = row_ix+offset
        @inbounds Jv[master_ix] = col_ix+offset
        @inbounds Vv[master_ix] = Dj[row_ix, col_ix]
        master_ix += 1
      end
    end
  end
  # Off-diagonal blocks. There are a couple enumerates in here that will
  # inevitably make a few allocations, but I think for the moment this is about
  # as optimized as is sensible.
  for (j,ix_c) in enumerate(U.condix)
    isempty(ix_c) && continue
    Bj = U.odiagonals[j]
    c_offset = 0
    for ix_c_k in ix_c
      @inbounds idxs_c = U.idxs[ix_c_k]
      for (_i1, i1) in enumerate(idxs_c)
        for (_i2, i2) in enumerate(U.idxs[j])
          @inbounds Iv[master_ix] = i1
          @inbounds Jv[master_ix] = i2
          @inbounds Vv[master_ix] = Bj[_i1+c_offset,_i2]
          master_ix += 1
        end
      end
      c_offset += length(idxs_c)
    end
  end
  sparse(Iv, Jv, Vv)
end

function _rchol(V::ChunkedVecchiaApproximation{M,D,F}, params; 
                use_tiles=false) where{M,D,F}
  out   = RCholeskyStorage_alloc(V)
  tiles = use_tiles ? build_tiles(V, params) : nothing
  rchol_instantiate!(out, V, params, tiles)
end

function _rchol_singleton!(valbuf, V::SingletonVecchiaApproximation{M,D,F},
                           buffers, chunks, params, vixs) where{M,D,F}
  @sync for (i, chunk) in enumerate(chunks)
    @spawn begin
      bufi = buffers[i]
      for j in chunk
        valj = view(valbuf, vixs[j])
        cvar = Vecchia.prepare_conditional!(bufi, j, V, params)
        idxs = V.condix[j]
        if isempty(idxs)
          valj[1] = inv(sqrt(cvar))
        else
          kwts = view(bufi.buf_kwts, 1:length(idxs))
          isqrt_cvar = inv(sqrt(cvar))
          kwts .*= -isqrt_cvar
          view(valj, 1:(length(valj)-1)) .= kwts
          valj[end] = isqrt_cvar
        end
      end
    end
  end
  nothing
end

function _rchol(V::SingletonVecchiaApproximation{M,D,F}, params) where{M,D,F}
  n   = length(V.pts)
  # pre-allocate the COO format (I,J,V), like the output from findnz(sparse_matrix).
  nnz = sum(length, V.condix) + length(V.condix)
  IJ  = Vector{Tuple{Int64, Int64}}()
  sizehint!(IJ, nnz)
  for (j, cj) in enumerate(V.condix)
    for k in cj
      push!(IJ, (k, j))
    end
    push!(IJ, (j,j))
  end
  (I, J)   = (getindex.(IJ, 1), getindex.(IJ, 2))
  vals     = Vector{Float64}(undef, nnz)
  view_ixs = Vector{UnitRange{Int64}}()
  sizehint!(view_ixs, length(V.condix))
  current_ix = 1
  for (j, cj) in enumerate(V.condix)
    cix = current_ix:(current_ix + length(cj))
    push!(view_ixs, cix)
    current_ix = cix[end]+1
  end
  # split the work into chunks and pre-allocate work buffers.
  chunks  = collect(Iterators.partition(1:n, cld(n, nthreads())))
  buffers = [cnllbuf(V, params) for _ in 1:length(chunks)]
  # with everything allocated, hand to the non-allocating routine.
  blas_nthr = BLAS.get_num_threads()
  BLAS.set_num_threads(1)
  _rchol_singleton!(vals, V, buffers, chunks, params, view_ixs)
  BLAS.set_num_threads(blas_nthr)
  UpperTriangular(sparse(I, J, vals))
end

"""
`rchol(cfg::VecchiaApproximation, params; issue_warning=true, use_tiles=false)`

A method for assembling an upper-triangular matrix `U` that gives a sparse "reverse" Cholesky factor for your precision matrix. In particular, if Σ is the covariance matrix for your data, then Σ^{-1} ≈ U*U'. Permutations are handled internally and will depend on your `VecchiaApproximation`..

Optional keyword arguments are:
- `use_tiles` is an option to pre-compute block covariances and store them. This can potentially speed up assembly in stationary models or approximation configurations with many redundant kernel evaluations.
"""
function rchol end

function rchol(V::ChunkedVecchiaApproximation{M,D,F}, params; 
               use_tiles=false) where{M,D,F}
  out = _rchol(V, params; use_tiles=use_tiles)
  RCholesky(UpperTriangular(sparse(out)), V.perm, invperm(V.perm),
            Array{Float64}(undef, length(V.pts)))
end

function rchol(V::SingletonVecchiaApproximation{M,D,F}, params; 
               use_tiles=false) where{M,D,F}
  use_tiles && throw(error("Sorry, `use_tiles=true` is not currently implemented for singleton configurations."))
  out = _rchol(V, params)
  RCholesky(out, V.perm, invperm(V.perm),
            Array{Float64}(undef, length(V.pts)))
end

"""
`rchol_preconditioner(cfg::VecchiaApproximation, params; issue_warning=true, use_tiles=false)`

A method for assembling a preconditioner based on the sparse reverse inverse Cholesky factorization induced by the Vecchia approximation.

Optional keyword arguments are:
- `use_tiles` is an option to pre-compute block covariances and store them. This can potentially speed up assembly in stationary models or approximation configurations with many redundant kernel evaluations.
"""
function rchol_preconditioner end

function rchol_preconditioner(V::ChunkedVecchiaApproximation{M,D,F},
                              params; use_tiles=false) where{M,D,F}
  out = _rchol(V, params; use_tiles=use_tiles)
  RCholeskyPreconditioner(UpperTriangular(sparse(out)), V.perm, invperm(V.perm),
                          Array{Float64}(undef, length(V.pts)))
end

function rchol_preconditioner(V::SingletonVecchiaApproximation{M,D,F},
                              params; use_tiles=false) where{M,D,F}
  use_tiles && throw(error("Sorry, `use_tiles=true` is not currently implemented for singleton configurations."))
  out = _rchol(V, params)
  RCholeskyPreconditioner(out, V.perm, invperm(V.perm),
                          Array{Float64}(undef, length(V.pts)))
end

