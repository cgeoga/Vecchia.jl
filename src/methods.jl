# A few small new methods:
Base.size(U::RCholesky{T}) where{T} = (U.idxs[end][end], U.idxs[end][end])
LinearAlgebra.adjoint(U::RCholesky{T}) where{T} = Adjoint{T,RCholesky{T}}(U)
LinearAlgebra.transpose(U::RCholesky{T}) where{T} = Adjoint{T,RCholesky{T}}(U)
LinearAlgebra.logdet(U::RCholesky{T}) where{T} = sum(logdet, U.diagonals)

# Pass this around instead?
struct RCholApplicationBuffer{T}
  bufv::Matrix{T}
  bufm::Matrix{T}
  bufz::Matrix{T}
  out::Matrix{T}
end

function RCholApplicationBuffer(U::RCholesky{T}, v::Matrix) where{T}
  Z = promote_type(T, eltype(v))
  out  = Array{Z}(undef, size(v))
  bufz = Array{Z}(undef, size(v))
  bufv = Array{Z}(undef, maximum(length, U.condix)*maximum(length, U.idxs), size(v,2))
  bufm = Array{Z}(undef, maximum(length, U.idxs), size(v,2))
  RCholApplicationBuffer{Z}(bufv, bufm, bufz, out)
end

# Transpose application U'*x. Formerly with LinearAlgebra.mul!, but this
# actually needs a few extra buffers, so I'm not doing that at the moment.
#
# TODO (cg 2022/05/30 12:15): Benchmark parallelizing and see if it helps.
#
#  TODO (cg 2022/05/31 11:36): Why are the so many allocations in this??
function apply!(out::Matrix{Z}, Ut::Adjoint{T,RCholesky{T}}, 
                v, bufv::Matrix{Z}, bufm::Matrix{Z}) where{Z,T}
  U = Ut.parent
  @assert size(bufv) == (maximum(length, U.condix)*maximum(length, U.idxs), size(v,2))
  @assert size(bufm) == (maximum(length, U.idxs), size(v,2))
  fill!(out, zero(eltype(out)))
  for j in eachindex(U.condix)
    cj   = U.condix[j]
    ixj  = U.idxs[j]
    ixcj = U.idxs[cj]
    out_mod_chunk = view(out, ixj, :)
    mul!(out_mod_chunk, U.diagonals[j]', view(v, ixj, :))
    isempty(cj) && continue
    Bjt_chunk = U.odiagonals[j]
    bufv_v = prepare_v_buf!(bufv, v, ixcj)
    bufm_v = view(bufm, 1:length(ixj), :)
    mul!(bufm_v, Bjt_chunk', bufv_v)
    out_mod_chunk .+= bufm_v
  end
  out
end


# Direct application U*x. See the above comments, which also apply here.
#
# For the moment while I work on the logic, just doing this in two passes.
#
# As written, this may be pretty hard to parallelize effectively.
function apply!(out::Matrix{Z}, U::RCholesky{T}, 
                v::Matrix, bufv::Matrix{Z}) where{Z,T}
  @assert size(bufv) == (maximum(length, U.condix)*maximum(length, U.idxs), size(v,2))
  fill!(out, zero(Z))
  # pass one: diagonal blocks.
  for j in eachindex(U.idxs)
    ixj = U.idxs[j]
    Dj  = U.diagonals[j]
    out_mod_chunk = view(out, ixj, :)
    mul!(out_mod_chunk, Dj, view(v, ixj, :))
  end
  # Second pass: off-diagonal blocks:
  for j in eachindex(U.condix)
    cixj = U.condix[j]
    isempty(cixj) && continue
    ixj  = U.idxs[j] 
    ixcj = U.idxs[cixj]
    Bjt_chunk = U.odiagonals[j]
    start_ix  = 1
    for k in eachindex(cixj)
      cixj_k  = cixj[k]
      ixk     = U.idxs[cixj_k]
      lenk    = length(ixk)
      stop_ix = start_ix+lenk-1
      Bjt_chunk_k = view(Bjt_chunk, start_ix:stop_ix, :)
      bufv_k  = view(bufv, 1:lenk, :)
      mul!(bufv_k, Bjt_chunk_k, view(v, ixj, :))
      view(out, ixk, :) .+= bufv_k
      start_ix += lenk
    end
  end
  out
end

function applyUUt!(bufs::RCholApplicationBuffer{Z}, 
                   U::RCholesky{T}, 
                   v::Matrix) where{Z,T}
  Ut = U'
  # first, adjoint application:
  apply!(bufs.bufz, Ut, v, bufs.bufv, bufs.bufm)
  # now direct application:
  apply!(bufs.out, U, bufs.bufz, bufs.bufv)
  bufs.out
end

function applyUt(U::RCholesky{T}, v) where{T}
  Z    = promote_type(T, eltype(v))
  out  = Array{Z}(undef, size(v))
  bufv = Array{Z}(undef, maximum(length, U.condix)*maximum(length, U.idxs), size(v,2))
  bufm = Array{Z}(undef, maximum(length, U.idxs), size(v,2))
  apply!(out, U', v, bufv, bufm)
end

function applyUUt(U::RCholesky{T}, v::Matrix) where{Z,T}
  bufs = RCholApplicationBuffer(U, v)
  applyUUt!(bufs, U, v)
end

function Base.:*(Ut::Adjoint{T,RCholesky{T}}, v::Matrix) where{T}
  Z    = promote_type(T, eltype(v))
  U    = Ut.parent
  out  = Array{Z}(undef, size(v))
  bufv = Array{Z}(undef, maximum(length, U.condix)*maximum(length, U.idxs), size(v,2))
  bufm = Array{Z}(undef, maximum(length, U.idxs), size(v,2))
  apply!(out, Ut, v, bufv, bufm)
end

function Base.:*(U::RCholesky{T}, v::Matrix) where{T}
  Z    = promote_type(T, eltype(v))
  out  = Array{Z}(undef, size(v))
  bufv = Array{Z}(undef, maximum(length, U.condix)*maximum(length, U.idxs), size(v,2))
  apply!(out, U, v, bufv)
end

function apply_parallel(Ut::Adjoint{T,RCholesky{T}}, v::AbstractMatrix) where{T}
  U    = Ut.parent
  Z    = promote_type(T, eltype(v))
  out  = Array{Z}(undef, size(v))
  fill!(out, zero(Z))
  _szv = (maximum(length, U.condix)*maximum(length, U.idxs), size(v,2))
  _szm = (maximum(length, U.idxs), size(v,2) )
  @floop ThreadedEx() for j in eachindex(U.condix)
    @init bufv = Array{Z}(undef, _szv)
    @init bufm = Array{Z}(undef, _szm)
    cj   = U.condix[j]
    ixj  = U.idxs[j]
    ixcj = U.idxs[cj]
    out_mod_chunk = view(out, ixj, :)
    mul!(out_mod_chunk, U.diagonals[j]', view(v, ixj, :))
    isempty(cj) && continue
    Bjt_chunk = U.odiagonals[j]
    bufv_v = prepare_v_buf!(bufv, v, ixcj)
    bufm_v = view(bufm, 1:length(ixj), :)
    mul!(bufm_v, Bjt_chunk', bufv_v)
    out_mod_chunk .+= bufm_v
  end
  out
end

