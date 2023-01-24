
# This file is sort of a document-by-showing of how to provide a custom struct
# that gives the covariance of your additive noise to be used in the EM case.
# This example shows the most basic case of just a scaled identity matrix, but
# it demonstrates the methods your custom struct should have.
#
# In particular, in your code, you would define 
#
# struct MyCoolErrorCovariance
#   [...]
# end
#
# and then add the methods
#
# function Vecchia.error_covariance(R::MyCoolErrorCovariance, p)
#   [...]
# end
#
# and so on.

struct ScaledIdentity
  size::Int64
end

# Methods one and two: constructors for the matrix and its inverse. This
# function needs to return something that can be ADDED to a sparse matrix. This
# UniformScaling object can, but if it is more general like that you'll probably
# want to return a Diagonal or a SparseMatrixCSC.
error_covariance(R::ScaledIdentity, p) = p[end]*I
error_precision(R::ScaledIdentity,  p) = inv(p[end])*I

# method three: evaluating the quadratic form v^T R^{-1} v. In the special case
# where R = x*I, you can pre-square-and-sum all the v elements, which can speed
# things up. So that squared sum is a fourth argument to this function, but you
# can just ignore it if it isn't useful to you.
error_qform(R::ScaledIdentity, p, y, yTy) = yTy/p[end]

# method four: the log-determinant of the error covariance. 
error_logdet(R::ScaledIdentity,  p) = R.size*log(p[end])

# method five: check that R is invertible.
error_isinvertible(R::ScaledIdentity, p) = p[end] > zero(eltype(p))

# method six: a generic kernel evaluation.
(S::ScaledIdentity)(x, y, params) = ifelse(x==y, params[end], zero(eltype(params)))


#
# OPTIONAL METHODS:
# 

# (optional) a specialized error nll. But you were already forced to do the
# quadratic form and the logdet, so unless there are additional savings to be
# made in computing them together you probably don't need to specialize this
# method to your type.
error_nll(R, p, y) = (error_logdet(R, p) + error_qform(R, p, nothing, sum(abs2, y)))/2

