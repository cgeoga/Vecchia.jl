
# From Sun and Stein 2016 JCGS, which is to my knowledge the first explicit
# method for computing precision matrices induced by Vecchia likelihoods.
#
# TODO (cg 2021/11/19 10:15): It would lead to even less readable code, but it
# would probably further help threading performance to squeeze out _all_
# allocations from the parallel loop, which means pre_allocating for _sI, _sJ,
# and _sV. Could potentially pre-allocate large enough buffers once, like with
# workcc/workpp/workcp, but I'm not sure how that will interact with append!!.
function precisionmatrix(V::VecchiaConfig{D,F}, params::Vector{T}) where{D,F,T}
  # Check that the conditioning set indices are sorted:
  checksorted(V)
  # Create master indices for sparse matrix stuff.
  gixs = globalidxs(V.data)
  # create thread buffers for the matrices like in the nll function.
  n    = sum(length, V.pts)
  chsz = V.chunksize        # this probably should be computed more carefully.
  ccsz = chsz*V.blockrank   # this probably should be computed more carefully. 
  cpsz = chsz + ccsz
  # Construct the small piece of the sparse matrix for each chunk.
  @floop ThreadedEx() for j in 1:length(V.condix)
    @init workcc = Array{T}(undef, cpsz, cpsz)
    @init workcp = Array{T}(undef, ccsz, chsz)
    @init workpp = Array{T}(undef, chsz, chsz)
    # Extract the relevant indices and the actual points. We delay preparing the
    # set of conditioning points in case it's empty.
    (idxs, pts) = (V.condix[j], V.pts[j])
    # Prepare the buffer for points[j], doing this first so that we can early
    # return for the edge case of j==1 and so cpts is empty.
    pbuf = view(workpp, 1:length(pts),  1:length(pts))
    updatebuf!(pbuf, pts, pts, V.kernel, params)
    if isone(j) 
      (_sI, _sJ, _sV) = sunsteinchunk1(T, n, pbuf)
    else
      # now that we know the set of conditioning points isn't empty, prepare it.
      cpts   = reduce(vcat, V.pts[idxs])
      # prepare the other buffers and update them.
      cbuf   = view(workcc, 1:length(cpts), 1:length(cpts))
      cpbuf  = view(workcp, 1:length(cpts), 1:length(pts))
      updatebuf!(cbuf,  cpts, cpts, V.kernel, params)
      updatebuf!(cpbuf, cpts,  pts, V.kernel, params)
      # compute the solve that gets used for b.
      fcbuf  = cholesky!(cbuf)
      solve  = fcbuf\cpbuf
      # compute the conditional covariance matrix and factorize it.
      mul!(pbuf, Adjoint(cpbuf), solve, -1, 1)
      fccov  = cholesky!(Symmetric(pbuf))
      # compute the additional nonzero indices and values for the sparse matrix.
      mulbuf = view(workcc, 1:(length(pts) + length(cpts)),
                            1:(length(pts) + length(cpts)))
      (_sI, _sJ, _sV) = sunsteinchunk(T, n, solve, fccov, mulbuf,
                                      gixs[j], reduce(vcat, gixs[idxs]))
    end
    # append (in a thread safe way) to the master (I,J,V) vectors.
    @reduce(sI = append!!(EmptyVector{Int64}(), _sI))
    @reduce(sJ = append!!(EmptyVector{Int64}(), _sJ))
    @reduce(sV = append!!(EmptyVector{T}(),     _sV))
  end
  Symmetric(sparse(sI, sJ, sV), :L)
end

