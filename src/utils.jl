
function checkthreads()
  nthr = Threads.nthreads()
  if nthr > 1 && BLAS.get_num_threads() > 1
    @warn "It looks like you started Julia with multiple threads but are also using \
    multiple BLAS threads. The Julia multithreading isn't composable with BLAS \
    multithreading, so you should probably choose one or the other. You can set \
    BLAS threading off with BLAS.set_num_threads(1). Alternatively, you can turn \
    off the Vecchia.jl multi-threading by passing FLoops.SequentialEx() as a kwarg \
    to Vecchia.nll and co. as execmode=Vecchia.FLoops.SequentialEx()." maxlog=1
  end
end

# A hacky function to return an empty Int64[] for the first conditioning set.
@inline cond_ixs(j, r) = j == 1 ? Int64[] : collect(max(1,j-r):max(1,j-1))

# number of elements in the lower triangle of an n x n matrix.
ltrisz(n) = div(n*(n+1), 2)

# Update kernel matrix buffer, exploiting redundancy for symmetric case. Not
# necessarily faster unless the kernel is very expensive to evaluate, but not
# slower in any case in my experimentation.
#
# Note that this does _NOT_ use threads, since I am already assuming that the
# nll function itself will be using threads, and in my benchmarking putting
# threaded constructors here slows things down a bit and increases allocations.
function updatebuf!(buf, pts1, pts2, kfun::F, params; skipltri=false) where{F}
  (F <: MemoizedKernel) && (@assert hash(params) === kfun.phash "params for memoized kernel don't agree with provided params. This shouldn't happen and is a bug.")
  if pts1 == pts2 && skipltri
    for k in eachindex(pts2)
      ptk = pts2[k]
      @inbounds buf[k,k] = kfun(ptk, ptk, params)
      @inbounds for j in 1:(k-1)
        buf[j,k] = kfun(pts1[j], ptk, params)
      end
    end
  elseif pts1 == pts2 && !skipltri
    for k in eachindex(pts2)
      ptk = pts2[k]
      @inbounds buf[k,k] = kfun(ptk, ptk, params)
      @inbounds for j in 1:(k-1)
        buf[j,k] = kfun(pts1[j], ptk, params)
        buf[k,j] = kfun(pts1[j], ptk, params)
      end
    end
  else
    @inbounds for k in eachindex(pts2), j in eachindex(pts1) 
      buf[j,k] = kfun(pts1[j], pts2[k], params)
    end
  end
  nothing
end

# This function works pretty differently: now we assume that the points have all
# been catted together and that the kernel function takes entirely scalar
# inputs. With this formatting, we can then actually use the SIMD tools of
# LoopVectorization.jl and get some serious speedup.
#
# Very grateful to Chris Elrod (@elrod on discourse, @chriselrod on Github) for
# the help in making this work.
@generated function updatebuf_avx!(buf, ::Val{D}, pts1, pts2, 
                                   kfun, params; skipltri=false) where{D}
  quote
    if pts1 == pts2 && skipltri
      for _k in 0:div(length(pts2)-1,$D)
        @turbo for _j in 0:_k # @turbo
        #@inbounds for _j in 0:_k 
          val = kfun($([:(pts1[_j*$D+$d]) for d in 1:D]...),
                     $([:(pts2[_k*$D+$d]) for d in 1:D]...),
                     params)
          buf[_j+1,_k+1] = val
        end
      end
    elseif pts1 == pts2 && !skipltri
      for _k in 0:div(length(pts2)-1,$D)
        @turbo for _j in 0:_k # @turbo
        #@inbounds for _j in 0:_k 
          val = kfun($([:(pts1[_j*$D+$d]) for d in 1:D]...),
                     $([:(pts2[_k*$D+$d]) for d in 1:D]...),
                      params)
          buf[_j+1,_k+1] = val
          buf[_k+1,_j+1] = val
        end
      end
    else
      @turbo for  _k in 0:div(length(pts2)-1,$D),  _j in 0:div(length(pts1)-1,$D) # @turbo
      #@inbounds for  _k in 0:div(length(pts2)-1,$D),  _j in 0:div(length(pts1)-1,$D) 
        val = kfun($([:(pts1[_j*$D+$d]) for d in 1:D]...),
                   $([:(pts2[_k*$D+$d]) for d in 1:D]...),
                    params)
        buf[_j+1,_k+1] = val
      end
    end
    nothing
  end
end

# Primarily kept because it's readable and helps to transition from math
# notation in a paper to code.
function sunsteinchunk_naive(T, n, cpbuf, solve, fccov, 
                             pts_ixs::AbstractVector{Int64}, 
                             cnd_ixs::AbstractVector{Int64})
  # allocate the sparse matrix for b_j^T. Eventually, a possible
  # micro-optimization would be to avoid this entirely and just build the final
  # one from the nzindices. But for now let's just do this.
  bt = spzeros(T, length(pts_ixs), n)
  # fill in the identity matrix part, doing the solves along the way.
  # about 50% of the time.
  for (j, ptixj) in enumerate(pts_ixs)
    ej    = zeros(T, length(pts_ixs))
    ej[j] = one(T)
    ldiv!(fccov.U', ej)
    bt[:,ptixj] .= ej
  end
  # solve the linear system, overwriting input "solve", and update the
  # corresponding columns of bt:
  ldiv!(fccov.U', Adjoint(solve))
  for (j, rowj) in enumerate(eachrow(solve))
    bt[:,cnd_ixs[j]] .= -rowj
  end
  # return the conjugation.
  return bt'bt # almost _all_ the allocs.
end

function sparseIJvecs_ltri(ixs)
  len = ltrisz(length(ixs))
  (Iv, Jv) = (Vector{Int64}(undef, len), Vector{Int64}(undef, len))
  current  = 1
  for j in eachindex(ixs)
    view(Iv, current:(current+j-1)) .= ixs[j]
    view(Jv, current:(current+j-1)) .= view(ixs, 1:j)
    current += j
  end
  (Iv, Jv)
end

function sparseIJvecs_full(ixs)
  ilen = length(ixs)
  len  = ilen^2
  (Iv, Jv) = (Vector{Int64}(undef, len), Vector{Int64}(undef, len))
  # fill in the I vector. Slightly more involved this time.
  current = 1
  for j in eachindex(ixs)
    view(Iv, current:(current+ilen-1)) .= ixs[j]
    current += ilen
  end
  # fill in the J vector. Again, slightly more involved.
  current = 1
  for j in eachindex(ixs)
    view(Jv, current:(current+ilen-1)) .= ixs
    current += ilen
  end
  (Iv, Jv)
end

function update_IJ!(ixs, I, J)
  @inbounds for l in eachindex(I,J)
    I[l] = ixs[I[l]]
    J[l] = ixs[J[l]]
  end
  nothing
end

function prepare_columns(T, solve, fccov)
  sz  = size(solve, 1) + size(fccov,1)
  out = Array{T}(undef, size(fccov,1), sz)
  for j in 1:size(solve,1)
    view(out, :, j) .= -view(solve, j, :)
  end
  for (row_counter, j) in enumerate((size(solve, 1)+1):sz)
    colj = view(out, :, j)
    fill!(colj, zero(T))
    colj[row_counter] = one(T)
  end
  lastchunk = view(out, :, (size(solve, 1)+1):sz)
  ldiv!(fccov.U', lastchunk)
  out
end

# TODO (cg 2021/04/24 18:26): this could always be optimized better. 
# I suppose I could get rid of the allocation in combined by making the cross
# buffer slightly larger than it needs to be and updating that object. One of
# the dimensions is right already. But it's hard to imagine that boosting
# performance.
function sunsteinchunk(T, n, solve, fccov, mulbuf,
                       pts_ixs::AbstractVector{Int64}, 
                       cnd_ixs::AbstractVector{Int64})::Tuple{Vector{Int64}, Vector{Int64}, Vector{T}}
  # Incredibly, this hcat code seems to be more efficient than prepare_columns
  # above. I'm really confused about how it has fewer allocations. 
  ldiv!(fccov.U', Adjoint(solve))
  combined     = hcat(-permutedims(solve), inv(fccov.U'))
  combined_ixs = vcat(cnd_ixs, pts_ixs)
  # index set:
  (Iv, Jv) = sparseIJvecs_ltri(1:length(combined_ixs))
  # fill in the matrix entries. The extra allocation isn't ideal, but BLAS-3!
  # having experimented with dropping the ltri-only approach so that I can just
  # return Vv = vec(mulbuf), this approach oddly has slightly more allocations
  # but was significantly faster---by about a factor of 2.
  mul!(mulbuf, Adjoint(combined), combined)
  Vv  = Vector{T}(undef, ltrisz(length(combined_ixs)))
  @turbo for l in eachindex(Iv, Jv)
  #@inbounds for l in eachindex(Iv, Jv)
    j     = Iv[l]
    k     = Jv[l]
    Vv[l] = mulbuf[j,k]
  end
  update_IJ!(combined_ixs, Iv, Jv)
  return (Iv, Jv, Vv)
end

# Primarily kept because it's readable and helps to transition from math
# notation in a paper to code.
function sunsteinchunk1_naive(T, n, buf)
  out = spzeros(T, n, n)
  len = size(buf, 1)
  out[1:len, 1:len] .= inv(buf)
  out
end

# something that instead returns nzindices instead of constructing the matrix.
function sunsteinchunk1(T, n, buf)::Tuple{Vector{Int64}, Vector{Int64}, Vector{T}}
  ibuf    = inv(buf)
  Vv      = Vector{T}(undef, ltrisz(size(buf, 1)))
  counter = 1
  @inbounds for j in 1:size(buf, 1)
    @inbounds for k in 1:j
      Vv[counter] = ibuf[j,k]
      counter += 1
    end
  end
  (Iv, Jv) = sparseIJvecs_ltri(1:size(buf, 1))
  return (Iv, Jv, Vv)
end

function globalidxs(datavv)
  (out, start) = (Vector{UnitRange{Int64}}(undef, length(datavv)), 1)
  for (j, datvj) in enumerate(datavv)
    len = size(datvj,1)
    out[j] = start:(start+len-1)
    start += len
  end
  out
end

function checksorted(V::VecchiaConfig{D,F}) where{D,F}
  all(issorted, V.condix) || throw(error("This function requires that every conditioning vector be sorted."))
  nothing
end

# TODO (cg 2022/04/21 16:16): This is totally not good.
function vec_of_vecs_to_matrows(vv)
  Matrix(reduce(hcat, vv)')
end

#= For debugging. This gives M = U*U'.
function rchol(M)
  tmp = cholesky(Symmetric(reverse(Matrix(M), dims=(1,2)))).L
  UpperTriangular(reverse(Matrix(tmp), dims=(1,2)))
end

# Again, for debugging. Don't use this.
irchol(M) = inv(cholesky(M).U)
=#

function prepare_v_buf!(buf, v::Matrix, idxv)
  _ix = 1
  for ixs in idxv
    for ix in ixs
      @inbounds view(buf, _ix, :) .= view(v, ix, :)
      _ix += 1
    end
  end
  view(buf, 1:(_ix-1), :)
end

function updateptsbuf!(ptbuf, ptvv, idxs)
  ix = 1
  for idx in idxs
    for pt in ptvv[idx]
      @inbounds ptbuf[ix] = pt
      ix += 1
    end
  end
  view(ptbuf, 1:(ix-1))
end

# Not a clever function at all,
function rchol_nnz(U::RCholesky{T}) where{T}
  # diagonal elements:
  out = sum(U.idxs) do ix
    n = length(ix)
    div(n*(n+1), 2)
  end
  # off-diagonal elements:
  out += sum(enumerate(U.condix)) do (j,ix_c)
    isempty(ix_c) && return 0
    tmp = 0
    len = length(U.idxs[j])
    for ix in ix_c
      @inbounds tmp += len*length(U.idxs[ix]) 
    end
    tmp
  end
  out
end

function debug_exactnll(cfg, params)
  pts = reduce(vcat, cfg.pts)
  dat = reduce(vcat, cfg.data)
  S   = Symmetric([cfg.kernel(x,y,params) for x in pts, y in pts])
  Sf  = cholesky!(S)
  (logdet(Sf), sum(z->z^2, Sf.U'\dat))
end

