
# Putting the struct here instead of in structstypes.jl as a violation of my own
# stye rules. It just only gets used here. Maybe there is a lesson about how to
# organize code in this choice somewhere...
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
  SR   = Symmetric(U*U' + Rinv) 
  SRf  = cholesky(SR, perm=n:-1:1) # for now
  (SRf\(Rinv*data), SRf)
end

function em_step(cfg, arg, saa, errormodel, optimizer; optimizer_kwargs...)
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
  divisor = sqrt(size(saa,2))
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
  optimizer(ejnll, arg; optimizer_kwargs...)
end

