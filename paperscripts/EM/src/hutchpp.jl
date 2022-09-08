
# Can't pre-condition here, so this needs a bunch of new solves.
function hutchpp(fn, S, G) 
  Q    = Matrix(qr(fn(S)).Q)
  _G   = G - Q*(Q'*G)
  tr(Q'*(fn(Q))) + tr(_G'*(fn(_G)))/size(_G,2)
end
hutchpp(A::AbstractMatrix, S, G) = hutchpp(z->A*z, S, G) # use trait bound instead?

function quasisolve_fullrowrank(A, B)
  AAtf = cholesky!(A*A') 
  A'*(AAtf\B)
end

# TODO (cg 2022/05/26 14:49): Test with implementation that I trust above.
function hutchpp_na(fn, S, R, G)
  @assert size(S,2) < size(R,2) "Need size(S,2) < size(R,2)."
  # Operations with A:
  Z   = fn(R)
  W   = fn(S)
  AG  = fn(G)
  # Subsequent operations on smaller matrices:
  StZ = S'*Z
  WtZ = W'*Z
  GtZ = G'*Z
  WtG = W'*G
  qs1 = quasisolve_fullrowrank(StZ, WtZ)
  qs2 = quasisolve_fullrowrank(StZ, WtG)
  # result:
  tr(qs1) + (tr(G'*AG) - tr(GtZ*qs2))/size(G,2)
end
hutchpp_na(A::AbstractMatrix, S, R, G) = hutchpp_na(z->A*z, S, R, G)

