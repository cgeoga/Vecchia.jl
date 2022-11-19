
# TODO (cg 2022/05/29 14:42): 
#
# 5. Longer-term, this constructor occasionally redundantly computes the
#    cross-covariance for individual chunks (j,k), or at least has the potential
#    to. It would be nice to offer users the option to allocate even more, but in
#    doing so create a cache that computes all the pairwise cross-covariances once
#    and only once, and then uses that cache to fill in chunks of the potentially
#    blocked cross-covs that are used when your condix has more than a single past
#    chunk in the conditioning sets.
#
#    This will probably not be earth-shattering in terms of perf unless the
#    kernel is really expensive...but some kernels are pretty expensive. I
#    frequently end up working with expensive kernels, even. So worth doing.
#

function prepare_diagonal_chunks(::Val{T}, sizes) where{T}
  map(sizes) do sz
    UpperTriangular{T, Matrix{T}}(I(sz))
  end
end

function prepare_odiagonal_chunks(::Val{T}, condix, sizes) where{T}
  map(enumerate(condix)) do (j,cj)
    isempty(cj) ? zeros(T, 0, 0) : zeros(T, sum(k->sizes[k], cj), sizes[j])
  end
end

# Just allocates all the memory for the struct. This does NOT fill in values.
function RCholesky_alloc(V::AbstractVecchiaConfig{H,D,F}, ::Val{T}) where{H,D,F,T}
  szs    = map(length, V.pts)
  diags  = prepare_diagonal_chunks(Val(T), szs)
  odiags = prepare_odiagonal_chunks(Val(T), V.condix, szs)
  RCholesky{T}(diags, odiags, V.condix, globalidxs(V.pts), [false]) 
end

# TODO (cg 2022/05/30 11:05): Continually look to squeeze allocations out of
# here. Maybe I can pre-allocate things for the BLAS calls, even?
function rchol_instantiate!(strbuf::RCholesky{T}, V::VecchiaConfig{H,D,F},
                            params::AbstractVector{T}, 
                            ::Val{Z}, ::Val{N}) where{H,D,F,T,Z,N}
  checkthreads()
  @assert !strbuf.is_instantiated[] "This instantiation function makes extensive use of in-place algebraic operations and makes certain assumptions about the values of those buffers coming in. Please make a new struct to pass in here, or manually reset your current one."
  strbuf.is_instantiated[] = true
  kernel  = V.kernel
  cpts_sz = V.chunksize*V.blockrank
  pts_sz  = V.chunksize
  # allocate three buffers:
  bufs = ntuple(j->crcholbuf(Val(D), Val(Z), cpts_sz, pts_sz), N)
  # do the main loop:
  m = cld(length(V.condix), Threads.nthreads())
  @sync for (i, chunk) in enumerate(Iterators.partition(1:length(V.condix), m))
    Threads.@spawn for j in chunk
      # get the buffer for this thread:
      tbuf = bufs[i]
      # get the data and points:
      pts = V.pts[j]
      dat = V.data[j]
      cov_pp = view(tbuf.buf_pp, 1:length(pts), 1:length(pts))
      if isone(j)
        # In this special case, I actually can skip the lower triangle. 
        updatebuf!(cov_pp, pts, pts, kernel, params, skipltri=true)
        cov_pp_f = cholesky!(Symmetric(cov_pp))
        buf      = strbuf.diagonals[1]
        ldiv!(cov_pp_f.U, buf)
      else
        # If j != 1, then I'm going to use an in-place mul! on this buffer, so I
        # need to fill it all in.
        updatebuf!(cov_pp, pts, pts, kernel, params, skipltri=false)
        # prepare conditioning points:
        idxs = V.condix[j]
        cpts = updateptsbuf!(tbuf.buf_cpts, V.pts, idxs)
        # prepare and fill in the matrix buffers pertaining to the cond.  points:
        cov_cp = view(tbuf.buf_cp, 1:length(cpts), 1:length(pts))
        cov_cc = view(tbuf.buf_cc, 1:length(cpts), 1:length(cpts))
        updatebuf!(cov_cc, cpts, cpts, kernel, params, skipltri=false)
        updatebuf!(cov_cp, cpts,  pts, kernel, params, skipltri=false)
        # pre-factorize the cpts:cpts marginal matrix, then get the two remaining
        # pieces, like the conditional covaraince of the prediction points. I
        # acknowledge that this is a little hard to read, but it really nicely cuts
        # out all the unnecessary allocations. If you do a manual check, you can
        # confirm that cov_pp becomes the conditional covariance of pts | cpts, etc.
        cov_cc_f = cholesky!(Symmetric(cov_cc))
        ldiv!(cov_cc_f.U', cov_cp)
        mul!(cov_pp, adjoint(cov_cp), cov_cp, -one(Z), one(Z))
        ldiv!(cov_cc_f.U, cov_cp)
        Djf = cholesky!(Symmetric(cov_pp))
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

function rchol(V::VecchiaConfig{H,D,F}, params::AbstractVector{T}; 
               issue_warning=true) where{H,D,F,T}
  if issue_warning
    @warn "Note that this is the reverse Cholesky factor for your data enumerated according to the permutation of the VecchiaConfig structure, so if you plan to apply this to vectors be sure to be mindful of potentially re-permuting. The simplest way to permute your data correct is with reduce(vcat, my_config.data). You can turn this warning off with the issue_warning kwarg." maxlog=1
  end
  # allocate:
  out = RCholesky_alloc(V, Val(T))
  # compute the out type and the number of threads to pass in as vals:
  Z   = promote_type(H, T)
  N   = Threads.nthreads()
  rchol_instantiate!(out, V, params, Val(Z), Val(N))
  out
end

function nll(U::RCholesky{T}, data::Matrix{Float64}) where{T}
  -logdet(U) + sum(x->x^2, U'*data)/2
end

# This is really just for debugging.
function nll_rchol(V::VecchiaConfig{H,D,F}, params::AbstractVector{T}) where{H,D,F,T}
  U    = rchol(V, params)
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
  master_len = rchol_nnz(U)
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

