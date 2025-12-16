
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
  println("  - (Cholesky factor) nzfraction:   $(nnz(rc.U)/prod(size(rc)))")
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

struct RCholeskyStorage{T}
  diagonals::Vector{UpperTriangular{T,Matrix{T}}}
  odiagonals::Vector{Matrix{T}}
  condix::Vector{Vector{Int64}}
  idxs::Vector{UnitRange{Int64}} 
  is_instantiated::Vector{Bool}
end

function RCholApplicationBuffer(U::RCholeskyStorage{T}, ndata::Int64, ::Val{V}) where{T,V}
  Z = promote_type(T, V)
  m = length(U.condix)
  out  = Array{Z}(undef, maximum(j->size(U.odiagonals[j], 2), 1:m), ndata)
  bufz = Array{Z}(undef, maximum(j->size(U.odiagonals[j], 2), 1:m), ndata)
  bufv = Array{Z}(undef, maximum(length, U.condix)*maximum(length, U.idxs), ndata)
  bufm = Array{Z}(undef, maximum(length, U.idxs), ndata)
  RCholApplicationBuffer{Z}(bufv, bufm, bufz, out)
end

struct CondRCholBuf{D,T}
  buf_pp::Matrix{T}
  buf_cp::Matrix{T}
  buf_cc::Matrix{T}
  buf_cpts::Vector{SVector{D,Float64}}
end

function crcholbuf(::Val{D}, ::Val{Z}, cpts_sz, pts_sz) where{D,Z}
  buf_pp = Array{Z}(undef,  pts_sz,  pts_sz)
  buf_cp = Array{Z}(undef, cpts_sz,  pts_sz)
  buf_cc = Array{Z}(undef, cpts_sz, cpts_sz)
  buf_cpts = Array{SVector{D,Float64}}(undef, cpts_sz)
  CondRCholBuf{D,Z}(buf_pp, buf_cp, buf_cc, buf_cpts)
end

function prepare_diagonal_chunks(::Val{T}, sizes) where{T}
  map(sizes) do sz
    UpperTriangular{T, Matrix{T}}(I(sz))
  end
end

function prepare_odiagonal_chunks(::Val{T}, condix, sizes) where{T}
  map(enumerate(condix)) do (j,cj)
    isempty(cj) ? zeros(T, 0, 0) : zeros(T, sum(view(sizes, cj)), sizes[j])
  end
end

# Just allocates all the memory for the struct. This does NOT fill in values.
function RCholeskyStorage_alloc(V::VecchiaApproximation{D,F}, ::Val{T}) where{D,F,T}
  szs    = map(length, V.pts)
  diags  = prepare_diagonal_chunks(Val(T), szs)
  odiags = prepare_odiagonal_chunks(Val(T), V.condix, szs)
  RCholeskyStorage{T}(diags, odiags, V.condix, globalidxs(V.pts), [false]) 
end

@generated function allocate_crchol_bufs(::Val{N}, ::Val{D}, ::Val{Z}, 
                                         cpts_sz, pts_sz) where{N,D,Z}
  quote
    Base.Cartesian.@ntuple $N j->crcholbuf(Val(D), Val(Z), cpts_sz, pts_sz)
  end
end

function allocate_crchol_bufs(n::Int64, ::Val{D}, ::Val{Z}, 
                              cpts_sz, pts_sz) where{D,Z}
  [crcholbuf(Val(D), Val(Z), cpts_sz, pts_sz) for _ in 1:n]
end

# TODO (cg 2022/05/30 11:05): Continually look to squeeze allocations out of
# here. Maybe I can pre-allocate things for the BLAS calls, even?
function rchol_instantiate!(strbuf::RCholeskyStorage, V::VecchiaApproximation{D,F},
                            params::AbstractVector{T}, ::Val{Z}, tiles) where{D,F,T,Z}
  @assert !strbuf.is_instantiated[] RCHOL_INSTANTIATE_ERROR
  strbuf.is_instantiated[] = true
  kernel  = V.kernel
  cpts_sz = chunksize(V)*blockrank(V)
  pts_sz  = chunksize(V)
  # allocate three buffers:
  nthr = Threads.nthreads()
  bufs = allocate_crchol_bufs(nthr, Val(D), Val(Z), cpts_sz, pts_sz)
  # do the main loop:
  m = cld(length(V.condix), Threads.nthreads())
  blas_nthr = BLAS.get_num_threads()
  BLAS.set_num_threads(1)
  @sync for (i, chunk) in enumerate(Iterators.partition(1:length(V.condix), m))
    Threads.@spawn for j in chunk
      # get the buffer for this thread:
      tbuf = bufs[i]
      # get the data and points:
      idxs = V.condix[j]
      pts  = V.pts[j]
      cov_pp = view(tbuf.buf_pp, 1:length(pts), 1:length(pts))
      if isempty(idxs)
        # In this special case, I actually can skip the lower triangle. 
        if tiles isa Nothing
          updatebuf!(cov_pp, pts, pts, kernel, params, skipltri=true)
        else
          updatebuf_tiles!(cov_pp, tiles, j, j)
        end
        cov_pp_f = cholesky!(Hermitian(cov_pp))
        buf      = strbuf.diagonals[j]
        ldiv!(cov_pp_f.U, buf)
      else
        # If the set of conditioning points is nonempty, then I'm going to use
        # an in-place mul! on this buffer, so I need to fill it all in.
        if tiles isa Nothing
          updatebuf!(cov_pp, pts, pts, kernel, params, skipltri=false)
        else
          updatebuf_tiles!(cov_pp, tiles, j, j)
        end
        # prepare conditioning points:
        cpts = updateptsbuf!(tbuf.buf_cpts, V.pts, idxs)
        # prepare and fill in the matrix buffers pertaining to the cond.  points:
        cov_cp = view(tbuf.buf_cp, 1:length(cpts), 1:length(pts))
        cov_cc = view(tbuf.buf_cc, 1:length(cpts), 1:length(cpts))
        if tiles isa Nothing
          updatebuf!(cov_cc, cpts, cpts, kernel, params, skipltri=false)
          updatebuf!(cov_cp, cpts,  pts, kernel, params, skipltri=false)
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
        ldiv!(cov_cc_f.U', cov_cp)
        mul!(cov_pp, adjoint(cov_cp), cov_cp, -one(Z), one(Z))
        ldiv!(cov_cc_f.U, cov_cp)
        Djf = cholesky!(Hermitian(cov_pp))
        # Update the struct buffers. Note that the diagonal elements are actually
        # UpperTriangular, and I am not supposed to mutate those. But we do the ugly
        # hack of directly working with the data buffer backing the UpperTriangular
        # wrapper, which is probably not really recommended, but it works.
        #strbuf.odiagonals[j] .= Bt_chunks
        strbuf.odiagonals[j] .= cov_cp
        strbuf_Djf = strbuf.diagonals[j].data
        ldiv!(Djf.U, strbuf_Djf)
        # now update the block with the rmul!, being careful to now use the
        # UpperTriangular matrix, NOT the raw buffer!
        strbuf_Bt = strbuf.odiagonals[j]
        strbuf_Bt .*= -one(Z)
        rmul!(strbuf_Bt, strbuf.diagonals[j])
      end
    end
  end
  BLAS.set_num_threads(blas_nthr)
  strbuf
end

# TODO (cg 2025/12/15 13:16): some smarter way of indexing so that this can be
# parallelized.
function SparseArrays.sparse(U::RCholeskyStorage{T}) where{T}
  master_len = _nnz(U.idxs, U.condix)
  Iv = Vector{Int64}(undef, master_len)
  Jv = Vector{Int64}(undef, master_len)
  Vv = Vector{T}(undef, master_len)
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

function _rchol(V::VecchiaApproximation{D,F}, params::AbstractVector{T}; 
                use_tiles=false) where{D,F,T}
  out   = RCholeskyStorage_alloc(V, Val(T))
  tiles = use_tiles ? build_tiles(V, params) : nothing
  rchol_instantiate!(out, V, params, Val(Float64), tiles)
end

"""
`rchol(cfg::VecchiaApproximation, params; issue_warning=true, use_tiles=false)`

A method for assembling an upper-triangular matrix `U` that gives a sparse "reverse" Cholesky factor for your precision matrix. In particular, if Σ is the covariance matrix for your data, then Σ^{-1} ≈ U*U'. Permutations are handled internally and will depend on your `VecchiaApproximation`..

Optional keyword arguments are:
- `use_tiles` is an option to pre-compute block covariances and store them. This can potentially speed up assembly in stationary models or approximation configurations with many redundant kernel evaluations.
"""
function rchol(V::VecchiaApproximation{D,F}, params::AbstractVector{T};
               use_tiles=false) where{D,F,T}
  out = _rchol(V, params; use_tiles=use_tiles)
  RCholesky(UpperTriangular(sparse(out)), V.perm, invperm(V.perm),
            Array{Float64}(undef, length(V.pts)))
end

"""
`rchol_preconditioner(cfg::VecchiaApproximation, params; issue_warning=true, use_tiles=false)`

A method for assembling a preconditioner based on the sparse reverse inverse Cholesky factorization induced by the Vecchia approximation.

Optional keyword arguments are:
- `use_tiles` is an option to pre-compute block covariances and store them. This can potentially speed up assembly in stationary models or approximation configurations with many redundant kernel evaluations.
"""
function rchol_preconditioner(V::VecchiaApproximation{D,F}, params::AbstractVector{T};
               use_tiles=false) where{D,F,T}
  out = _rchol(V, params; use_tiles=use_tiles)
  RCholeskyPreconditioner(UpperTriangular(sparse(out)), V.perm, invperm(V.perm),
                          Array{Float64}(undef, length(V.pts)))
end

