
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
  #@turbo for l in eachindex(Iv, Jv)
  @simd for l in eachindex(Iv, Jv)
    @inbounds Vv[l] = mulbuf[Iv[l],Jv[l]]
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

function checksorted(V::VecchiaApproximation{D,F}) where{D,F}
  all(issorted, V.condix) || throw(error("This function requires that every conditioning vector be sorted."))
  nothing
end

