
struct RCholesky{T}
  diagonals::Vector{UpperTriangular{T,Matrix{T}}}
  odiagonals::Vector{Matrix{T}}
  condix::Vector{Vector{Int64}}
  idxs::Vector{UnitRange{Int64}} 
  is_instantiated::Vector{Bool}
end

# TODO (cg 2022/05/30 12:10): make this more information.
Base.display(U::RCholesky{T}) where{T} = println("RCholesky{$T}")
Base.display(Ut::Adjoint{T,RCholesky{T}}) where{T} = println("Adjoint{$T, RCholesky{$T}}")

# Pass this around instead?
struct RCholApplicationBuffer{T}
  bufv::Matrix{T}
  bufm::Matrix{T}
  bufz::Matrix{T}
  out::Matrix{T}
end

function RCholApplicationBuffer(U::RCholesky{T}, ndata::Int64, ::Val{V}) where{T,V}
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
function RCholesky_alloc(V::VecchiaApproximation{H,D,F}, ::Val{T}) where{H,D,F,T}
  szs    = map(length, V.pts)
  diags  = prepare_diagonal_chunks(Val(T), szs)
  odiags = prepare_odiagonal_chunks(Val(T), V.condix, szs)
  RCholesky{T}(diags, odiags, V.condix, globalidxs(V.pts), [false]) 
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
function rchol_instantiate!(strbuf::RCholesky, V::VecchiaApproximation{H,D,F},
                           params::AbstractVector{T}, ::Val{Z}, tiles) where{H,D,F,T,Z}
  checkthreads()
  @assert !strbuf.is_instantiated[] RCHOL_INSTANTIATE_ERROR
  strbuf.is_instantiated[] = true
  kernel  = V.kernel
  cpts_sz = chunksize(V)*blockrank(V)
  pts_sz  = chunksize(V)
  # allocate three buffers:
  bufs = allocate_crchol_bufs(Threads.nthreads(), Val(D), Val(Z), cpts_sz, pts_sz)
  # do the main loop:
  m = cld(length(V.condix), Threads.nthreads())
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
  nothing
end

"""
`rchol(cfg::VecchiaApproximation, params; issue_warning=true, use_tiles=false)`

A method for assembling an upper-triangular matrix `U` that gives a sparse "reverse" Cholesky factor for your precision matrix. In particular, if
```
data = reduce(vcat, cfg.data) # to be sure of the permutations matching
```
and `S =`Var(`data[:,1]`), then `inv(S) ≈ U*U'`, where
```
U = UpperTriangular(sparse(rchol(cfg, params))).
```

Optional keyword arguments are:
- `issue_warning`, where a value of `false` will silence the warning about using the correct permutation, and
- `use_tiles` is an option to pre-compute block covariances and store them. This can potentially speed up assembly in stationary models with many redundant kernel evaluations.
"""
function rchol(V::VecchiaApproximation{H,D,F}, params::AbstractVector{T}; 
               issue_warning=true, use_tiles=false) where{H,D,F,T}
  if issue_warning
    notify_disable("issue_warning=false")
    @warn RCHOL_WARN maxlog=1
  end
  # allocate:
  out = RCholesky_alloc(V, Val(promote_type(T,H)))
  # compute the out type and the number of threads to pass in as vals:
  Z   = promote_type(H, T)
  # create tiles if requested:
  tiles = use_tiles ? build_tiles(V, params, Val(Z)) : nothing
  rchol_instantiate!(out, V, params, Val(Z), tiles)
  out
end

function nll(U::RCholesky{T}, data::AbstractVecOrMat{V}) where{T,V}
  buffers = [RCholApplicationBuffer(U, size(data,2), Val(V)) 
             for _ in 1:Threads.nthreads()]
  (logdets, qforms) = _nll(U, buffers, data)
  (logdets + qforms)/2
end

# Note that if the RChol factorization gives \Sigma^{-1} = U*U^T, then this
# computes the nll with the logdet how you'd expect and the quadratic form
# computed as sum(abs2, U^T*y). But I've squeezed all the allocations out of the
# hot loop so that it can be parallelized, so each little piece of U^T*y is
# computed in parallel (see _rchol_nll_term below for the code for an individual
# term).
function _nll(U::RCholesky{T}, buffers::Vector{RCholApplicationBuffer{V}},
              data) where{T,V}
  logdets = zeros(T, length(buffers))
  qforms  = zeros(T, length(buffers))
  m       = cld(length(U.condix), length(buffers))
  chunks  = Iterators.partition(eachindex(U.condix), m)
  @sync for (j, chunk) in enumerate(chunks)
    buf = buffers[j]
    Threads.@spawn for k in chunk
      (ldk, qfk)  = _rchol_nll_term(U, buf, data, k)
      logdets[j] += ldk
      qforms[j]  += qfk
    end
  end
  (sum(logdets), sum(qforms))
end

function _rchol_nll_term(U, buf, data, k)
  ck   = U.condix[k]
  ixk  = U.idxs[k]
  out_mod_chunk = view(buf.out, 1:length(ixk), :)
  mul!(out_mod_chunk, U.diagonals[k]', view(data, ixk, :))
  isempty(ck) && return (-2*logdet(U.diagonals[k]), sum(_square, out_mod_chunk))
  ixck = view(U.idxs, ck)
  Bjt_chunk = U.odiagonals[k]
  bufv_v = prepare_v_buf!(buf.bufv, data, ixck)
  bufm_v = view(buf.bufm, 1:length(ixk), :)
  mul!(bufm_v, Bjt_chunk', bufv_v)
  out_mod_chunk .+= bufm_v
  (-2*logdet(U.diagonals[k]), sum(_square, out_mod_chunk))
end

# This is really just for debugging.
function nll_rchol(V::VecchiaApproximation{H,D,F}, params::AbstractVector{T};
                   issue_warning=true) where{H,D,F,T}
  U    = rchol(V, params; issue_warning=issue_warning)
  data = reduce(vcat, V.data)
  nll(U, data)
end

# So much manual indexing here. I'm sorry to anybody who tries to read this. I
# just really wanted to keep allocations down, so I really did some grinding on
# the logic and a lot of manual book-keeping of the indices so I could use
# setindex! instead of push!.
#
# TODO (cg 2022/05/30 15:51): This is mucho serial. Which is probably best
# because of how fast it should run for even enormous matrices. But at some
# point should consider a parallel constructor that uses the fancy tools in the
# Transducers ecosystem (@floop, append!!, push!!, etc) to see if a parallel
# constructor works faster, even though it will allocate much more.
function SparseArrays.sparse(U::RCholesky{T}) where{T}
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

