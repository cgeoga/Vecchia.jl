# A few small new methods:
Base.size(U::RCholesky{T}) where{T} = (U.idxs[end][end], U.idxs[end][end])
LinearAlgebra.adjoint(U::RCholesky{T}) where{T} = Adjoint{T,RCholesky{T}}(U)
LinearAlgebra.transpose(U::RCholesky{T}) where{T} = Adjoint{T,RCholesky{T}}(U)
LinearAlgebra.logdet(U::RCholesky{T}) where{T} = sum(logdet, U.diagonals)

# TODO (cg 2022/05/30 12:15): Benchmark parallelizing and see if it helps.
function LinearAlgebra.mul!(out::Matrix{Z}, Ut::Adjoint{T,RCholesky{T}}, 
                            v::Matrix) where{Z,T}
  U    = Ut.parent
  bufv = Array{Z}(undef, maximum(length, U.condix)*maximum(length, U.idxs), size(v,2))
  bufm = Array{Z}(undef, maximum(length, U.idxs), size(v,2))
  for (j, cj) in enumerate(U.condix)
    out_mod_chunk = view(out, U.idxs[j], :)
    #out_mod_chunk .+= U.diagonals[j]'*view(v, U.idxs[j], :)
    mul!(out_mod_chunk, U.diagonals[j]', view(v, U.idxs[j], :))
    isempty(cj) && continue
    Bjt_chunk = U.odiagonals[j]
    bufv_v = prepare_v_buf!(bufv, v, U.idxs[cj])
    bufm_v = view(bufm, 1:length(U.idxs[j]), :)
    mul!(bufm_v, Bjt_chunk', bufv_v)
    out_mod_chunk .+= bufm_v
  end
  out
end

function Base.:*(U::Adjoint{T,RCholesky{T}}, v::Matrix) where{T}
  Z    = promote_type(T, eltype(v))
  out  = zeros(Z, size(v))
  mul!(out, U, v)
end
