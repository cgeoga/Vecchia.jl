
#
# This code is a very slightly modified version of code graciously shared by
# Florian Schafer (f-t-s.github.io), and was used in the paper Schafer et al
# SISC 2021: Sparse Cholesky Factorization by Kullback-Leibler Minimization. 
#
# See the full repository for more information:
# https://github.com/f-t-scholesky_by_KL_minimization
#

function _innerprod(iter1, u1, iter2, u2, nzind, nzval::Vector{T}) where{T}
  @inbounds begin
    res = zero(T)
    while (iter1 <= u1) && (iter2 <= u2)
      while (nzind[iter1] == nzind[iter2])  &&(iter1 <= u1) && (iter2 <= u2)
        res += nzval[iter1] * nzval[iter2]
        iter1 += 1
        iter2 += 1
      end
      if nzind[iter1] < nzind[iter2]
        iter1 += 1
      else
        iter2 += 1
      end
    end
    return res
  end
end

function _innerprod(range1, range2, nzind, nzval::Vector{T}) where{T}
  iter1 = range1.start
  u1    = range1.stop
  iter2 = range2.start
  u2    = range2.stop
  _innerprod(iter1, u1, iter2, u2, nzind, nzval)
end

function _el_icholRWHL!(s::Vector{T}, ind, jnd) where{T}
  @inbounds begin
    #piv_i goes from 1 to N and designates the number, which is being treated 
    for piv_i = 1:(size(ind, 1) - 1) 
      #piv_j ranges over pointers and designates where to find the present column
      for piv_j = ind[piv_i]:(ind[piv_i+1]-1)
        #iter designates the pointer, at which the i-iterator starts
        iter = ind[piv_i]
        #jter designates the pointer, at which the j-iterator starts
        jter = ind[jnd[piv_j]]
        #The condition makes sure, that only the columns up to th pivot 
        #are being summed.
        s[piv_j] -= _innerprod(iter, ind[piv_i+1] - 2,
                               jter, ind[jnd[piv_j]+1] - 2,
                               jnd, s)
        # TODO (cg 2022/10/12 10:05): Check these branches and potentially
        # flatten them. Maybe the indirection here can be reduced.
        if s[ind[jnd[piv_j]+1]-1] > zero(T)
          if jnd[piv_j] < piv_i
            s[piv_j] /= (s[ind[jnd[piv_j]+1]-1])
          else 
            s[piv_j] = sqrt(s[piv_j]) 
          end
        else
          s[piv_j] = zero(T)
        end
      end
    end
  end
  nothing
end

function icholU!(U::SparseMatrixCSC{T, Int64}) where{T}
  triu!(U)
  _el_icholRWHL!(U.nzval, U.colptr, U.rowval)
  U
end

# In-place Cholesky factorization of a symmetric matrix.
function icholU(U::SparseMatrixCSC{T, Int64}) where{T}
  _U = copy(U)
  icholU!(_U)
end

function ichol_nll(V::VecchiaApproximation{H,D,F}, params::AbstractVector{T},
                   errormodel) where{H,D,F,T}
  # Assemble the precision matrix for the process without the nugget, recalling
  # that this is the "reverse" Cholesky factor so that the precision is U*U',
  # even though U is upper triangular. This will automatically use
  # multithreading to assemble U, which is pretty optimized.
  U  = rchol(V, params, issue_warning=false)
  Us = sparse(U) # Can't make this UpperTriangular because...
  S  = Us*Us'    #...this method fails if Us isa UpperTriangular.
  # Add inverse of the nugget perturbation matrix, noting that the
  # parameterization this code currently uses gives the VARIANCE of the nugget,
  # not the standard deviation (so no square of params[end]).
  Rinv = error_precision(errormodel, params)
  S   += Rinv
  # Now factorize that:
  U_ichol = UpperTriangular(icholU!(S))
  # assemble the data and a buffer to solve:
  @assert isone(size(V.data[1], 2)) "Temporarily restricting to one observation"
  P       = promote_type(H, T)
  data    = convert(Vector{P}, vec(reduce(vcat, V.data)))
  # Now compute (Sigma + R)^{-1} y efficiently using the representation trick
  data_solved = applyUUt(U, data)
  ldiv!(U_ichol', data_solved)
  ldiv!(U_ichol,  data_solved)
  ldiv!(Rinv, data_solved)
  # now compute the log-determinant, spread over two lines for readability:
  _ldet  = -2*sum(log, diag(Us)) + 2*sum(log, diag(U_ichol)) 
  _ldet += error_logdet(errormodel, params)
  # return the full likelihood:
  return (_ldet + dot(data, data_solved))/2
end

