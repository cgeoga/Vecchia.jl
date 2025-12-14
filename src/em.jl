
####
# FOR A CUSTOM ERROR MODEL IN THE EM ITERATOR:
####
#
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
#
####
# END CUSTOM MODEL SECTION.
####



struct ErrorKernel{K,E} <: Function
  kernel::K
  error::E
end

function (k::ErrorKernel{K})(x, y, p) where{K}
  k.kernel(x,y,p)+k.error(x,y,p)
end

struct ExpectedJointNll{H,D,F,R} <: Function
  cfg::VecchiaConfig{H,D,F}
  errormodel::R
  data_minus_z0::Matrix{Float64}
  presolved_saa::Matrix{Float64}
  presolved_saa_sumsq::Float64
end

# Trying to move to callable structs instead of closures so that the
# precompilation can be better...
function (E::ExpectedJointNll{H,D,F})(p::AbstractVector{T}) where{H,D,F,T}
  # Like with the normal nll function, this section handles the things that
  # create type instability, and then passes them to _nll so that the function
  # barrier means that everything _inside_ _nll, which we want to be fast and
  # multithreaded, is stable and non-allocating.
  Z     = promote_type(H,T)
  ndata = size(E.data_minus_z0, 2)
  # compute the following terms at once using the augmented data:
  # - nll(V, z0)
  # - (2M)^{-1} sum_j \norm[2]{U(\p)^T v_j}^2, w/ v_j the pre-solved SAA.
  pieces = split_nll_pieces(E.cfg, Val(Z), Threads.nthreads())
  (logdets, qforms) = _nll(pieces, p)
  out  = (logdets*ndata + qforms)/2
  # add on the generic nll for the measurement noise and the quadratic forms
  # with the error matrix that contribute to the trace term. 
  out += error_nll(E.errormodel, p, E.data_minus_z0)
  out += error_qform(E.errormodel, p, E.presolved_saa, E.presolved_saa_sumsq)/2
  out
end

# TODO (cg 2023/01/20 14:26): Write the non-symmetrized version of this that
# only requires a more generic solve(R, v) interface.
"""
prepare_z0_SR0(cfg::VecchiaConfig, arg::AbstractVector) -> (z0, SR0)

Compute E [z | y] to use in the E function.
"""
function prepare_z0_SR0(cfg, arg, data, errormodel)
  n    = size(data, 1)
  Rinv = error_precision(errormodel, arg)
  U    = sparse(Vecchia.rchol(cfg, arg, issue_warning=false))
  SR   = Hermitian(U*U' + Rinv) 
  SRf  = cholesky(SR, perm=n:-1:1) # for now
  (SRf\(Rinv*data), SRf)
end

function augmented_em_cfg(V::VecchiaConfig{H,D,F}, z0, presolved_saa) where{H,D,F}
  chunksix = chunk_indices(V.pts)
  new_data = map(chunksix) do ixj
    hcat(z0[ixj,:], presolved_saa[ixj,:])
  end
  Vecchia.VecchiaConfig{H,D,F}(V.kernel, new_data, V.pts, V.condix)
end

function em_step(cfg, arg, saa, errormodel, solver, box_lower, box_upper)
  checkthreads() 
  # check that the variance parameter isn't zero:
  @assert error_isinvertible(errormodel, arg) EM_NONUG_WARN
  # Get the data and points and compute z_0 = E_{arg}[ z | y ].
  dat   = reduce(vcat, cfg.data) 
  pts   = reduce(vcat, cfg.pts)
  (z0, SR0f) = prepare_z0_SR0(cfg, arg, dat, errormodel)
  # Pre-solve the SAA factors (SCALED BY inv(sqrt(2*M))). 
  # Note that I need to compute two traces that look like tr( [U * U^T] *
  # [(PtL)*(PtL^T)]^{-1}). Re-arranging those things, that means I am computing
  # tr(M*M^T), where M=U'*(PtL)^{-T}. So that's what the pre-solve is here.
  divisor = sqrt(size(saa, 2))
  pre_sf_solved_saa = (SR0f.PtL'\saa)./divisor 
  pre_sf_solved_saa_sumsq = sum(_square, pre_sf_solved_saa)
  # Now, unlike the v1 version, create a NEW configuration where the "data"
  # field is actually hcat(z0, pre_sf_solved_saa./sqrt(2*M)). This division is
  # important because the qform calculator in Vecchia.nll doesn't know which
  # columns are data vs saa vectors, and so that division has to happen before hand.
  tmp_cfg = augmented_em_cfg(cfg, z0, pre_sf_solved_saa)
  # create the struct that evaluates the expected joint nll:
  dat_minus_z0 = dat-z0
  ejnll = ExpectedJointNll(tmp_cfg, errormodel, dat_minus_z0, 
                           pre_sf_solved_saa, pre_sf_solved_saa_sumsq)
  optimize(ejnll, arg, solver; box_lower=box_lower, box_upper=box_upper)
end

"""
`vecchia_estimate_nugget(cfg::VecchiaConfig, init, solver, errormodel; kwargs...)`

Find the MLE under the Vecchia approximation specified by `cfg`. Initialization is provided by `init`, and the optimizer is specified by `solver`. In this case of measurement noise, however, the estimator is refined with an EM algorithm approach. See the example file for a detailed demonstration.
"""
function vecchia_estimate_nugget(cfg, init, solver, errormodel; kwargs...)
  nugkernel = Vecchia.ErrorKernel(cfg.kernel, errormodel)
  nugcfg    = Vecchia.VecchiaConfig(nugkernel, cfg.data, cfg.pts, cfg.condix)
  vecchia_estimate(nugcfg, init, solver; kwargs...)
end

struct EMVecchiaIterable{H,D,F,O,R}
  cfg::VecchiaConfig{H,D,F}
  step_new::Vector{Float64}
  step_old::Vector{Float64}
  saa::Matrix{Float64}
  status::Ref{Symbol}
  norm2tol::Float64
  max_em_iter::Int64
  errormodel::R
  solver::O
  box_lower::Vector{Float64}
  box_upper::Vector{Float64}
end

function Base.display(M::EMVecchiaIterable)
  display(M.cfg)
  println("EM with:")
  println("  - SAA vectors:        $(size(M.saa, 2))")
  println("  - ℓ₂ convergence tol: $(M.norm2tol)")
  println("  - Maximum iterations: $(M.max_em_iter)")
  println("  - solver:             $(M.solver)")
end

function EMiterable(cfg, init, saa, errormodel, solver, box_lower, box_upper; kwargs...)
  kwargsd = Dict(kwargs)
  EMVecchiaIterable(cfg, copy(init), copy(init), saa, 
                    Ref(:NOT_CONVERGED),
                    get(kwargsd, :norm2tol,    1e-2),
                    get(kwargsd, :max_em_iter, 20),
                    errormodel, solver, box_lower, box_upper)
end

function Base.iterate(it::EMVecchiaIterable{H,D,F,O,R}, 
                      iteration::Int=0) where{H,D,F,O,R}
  # check if 2-norm convergence is achieved:
  if iteration > 1
    if norm(it.step_new - it.step_old) < it.norm2tol
      it.status[] = :CONVERGED_ELL2NORM
      return nothing
    end
  end
  # check if maximum iteration count is reached:
  if iteration > it.max_em_iter 
    it.status[] = :FAIL_MAX_EM_ITER
    return nothing
  end
  # Otherwise, estimate the next step:
  newstep = em_step(it.cfg, it.step_new, it.saa, it.errormodel, it.solver,
                    it.box_lower, it.box_upper)
  it.step_old .= it.step_new
  it.step_new .= newstep
  (newstep, iteration+1)
end

function em_refine(cfg, errormodel, saa, init; solver, 
                   box_lower, box_upper, verbose=true, kwargs...)
  iter = EMiterable(cfg, init, saa, errormodel, solver, box_lower, box_upper; kwargs...)
  if verbose
    println("Refining parameter estimate $(round.(init, digits=3)):")
    display(iter)
    println()
  end
  path = [init]
  for (icount, new_est) in enumerate(iter)
    # TODO (cg 2022/06/03 17:42): Use Printf module to have some nicely
    # formatted output. 
    if verbose
      print("$icount: ")
      println("  Step difference norm: $(norm(iter.step_new - iter.step_old))")
    end
    push!(path, new_est)
  end
  (path=path,)
end

function em_estimate(cfg, saa, init; errormodel, solver,
                     box_lower=fill(0.0, length(init)),
                     box_upper=fill(Inf, length(init)),
                     warn_notation=true, verbose=true,
                     norm2tol=1e-2, max_em_iter=20)
  if warn_notation
    notify_disable("warn_notation=false")
    @warn NOTATION_WARNING maxlog=1
end
  # compute initial estimator using Vecchia with the nugget and nothing thoughtful:
  verbose && println("\nComputing initial MLE with standard Vecchia...")
  em_init = try
    vecchia_estimate_nugget(cfg, init, solver, errormodel; 
                            box_lower=box_lower, box_upper=box_upper)
  catch
    @warn "Computing the Vecchia-with-nugget initializer failed, falling back to your provided init..."
    init
  end
  # now use the iterator interface:
  (init_result=em_init,
   em_refine(cfg, errormodel, saa, em_init; solver=solver, box_lower=box_lower,
             box_upper=box_upper, verbose=verbose, norm2tol=norm2tol,
             max_em_iter=max_em_iter)...)
end

