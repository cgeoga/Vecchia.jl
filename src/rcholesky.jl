
# TODO (cg 2022/05/29 14:42): 
#
# 2. Write a special constructor for that struct that gives the sparse matrix,
#    because I know people will want it. But that should be separate, and maybe
#    should really only serve as a debugging tool.
#
# 5. Longer-term, even the fancier RCholesky_Chunks occasionally redundantly
#    computes the cross-covariance for individual chunks (j,k), or at least has
#    the potential to. It would be nice to offer users the option to allocate
#    even more, but in doing so create a cache that computes all the pairwise
#    cross-covariances one and only once, and then uses that cache to fill in
#    chunks of the potentially blocked cross-covs that are used when your condix
#    has more than a single past chunk in the conditioning sets.

# Just allocates all the memory for the struct. This does NOT fill in values.
function RCholesky_alloc(V::AbstractVecchiaConfig{H,D,F}, T) where{H,D,F}
  szs    = map(length, V.pts)
  diags  = map(sz->UpperTriangular(Matrix{T}(I(sz))), szs)
  odiags = map(enumerate(V.condix)) do (j,cj)
    isempty(cj) && return zeros(T, 0, 0)
    c_size = sum(k->length(V.pts[k]), cj)
    zeros(T, c_size, szs[j])
  end
  RCholesky{T}(diags, odiags, V.condix, globalidxs(V.pts), false) 
end

# TODO (cg 2022/05/30 11:05): Continually look to squeeze allocations out of
# here. Maybe I can pre-allocate things for the BLAS calls, even?
function rchol_instantiate!(strbuf::RCholesky{T}, V::VecchiaConfig{H,D,F},
                            params::AbstractVector{T},
                            execmode=ThreadedEx()) where{H,D,F,T}
  @assert !strbuf.is_instantiated "This instantiation function makes extensive use of in-place algebraic operations and makes certain assumptions about the values of those buffers coming in. Please make a new struct to pass in here, or manually reset your current one."
  kernel  = V.kernel
  Z       = promote_type(H,T)
  cpts_sz = V.chunksize*V.blockrank
  pts_sz  = V.chunksize
  # allocate three buffers:
  @floop execmode for j in 1:length(V.condix)
    # allocate work buffers in the thread-safe way:
    @init buf_pp  = Array{Z}(undef,  pts_sz,  pts_sz)
    @init buf_cp  = Array{Z}(undef, cpts_sz,  pts_sz)
    @init buf_cc  = Array{Z}(undef, cpts_sz, cpts_sz)
    @init buf_pts = Array{SVector{D,Float64}}(undef, cpts_sz)
    pts = V.pts[j]
    dat = V.data[j]
    cov_pp = view(buf_pp, 1:length(pts), 1:length(pts))
    if isone(j)
      # In this special case, I actually can skip the lower triangle. 
      updatebuf!(cov_pp, pts, pts, kernel, params, skipltri=true)
      empty_dict = Dict{Tuple{Int64,Int64},Matrix{Z}}()
      cov_pp_f = cholesky!(Symmetric(cov_pp))
      buf      = strbuf.diagonals[1]
      ldiv!(cov_pp_f.U, buf)
      continue
    end
    # If j != 1, then I'm going to use an in-place mul! on this buffer, so I
    # need to fill it all in.
    updatebuf!(cov_pp, pts, pts, kernel, params, skipltri=false)
    # prepare conditioning points:
    idxs = V.condix[j]
    cpts = updateptsbuf!(buf_pts, V.pts, idxs)
    # prepare and fill in the matrix buffers pertaining to the cond.  points:
    cov_cp = view(buf_cp, 1:length(cpts), 1:length(pts))
    cov_cc = view(buf_cc, 1:length(cpts), 1:length(cpts))
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
  nothing
end

function rchol(V::VecchiaConfig{H,D,F}, params::AbstractVector{T}; 
               execmode=ThreadedEx()) where{H,D,F,T}
  out = RCholesky_alloc(V, T)
  rchol_instantiate!(out, V, params, execmode)
  out
end

function nll(U::RCholesky{T}, data::Matrix{Float64}) where{T}
  -logdet(U) + sum(x->x^2, U'*data)/2
end

# This is really just for debugging.
function nll_rchol(V::VecchiaConfig{H,D,F}, params::AbstractVector{T}; 
                   execmode=ThreadedEx()) where{H,D,F,T}
  U    = rchol(V, params; execmode=execmode)
  data = reduce(vcat, V.data)
  nll(U, data)
end

