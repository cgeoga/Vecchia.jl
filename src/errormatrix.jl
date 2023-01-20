
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
function error_qform(R::ScaledIdentity, p, presolved, presolved_sumsq) 
  presolved_sumsq/(2*p[end])
end

# method four: check that R is invertible.
error_isinvertible(R::ScaledIdentity, p) = p[end] > zero(eltype(p))


#
# OPTIONAL METHODS:
# 

# (optional) method four: a negative log-likelihood. For UniformScaling and
# Diagonal matrices, I have a generic nll for this. But you could define this
# method for your R::MyCoolType to specialize and exploit specific structure you
# want it to have.
error_nll(R, p, y) = generic_nll(error_covariance(R, p), y)

