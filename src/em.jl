
# Putting the struct here instead of in structstypes.jl as a violation of my own
# stye rules. It just only gets used here. Maybe there is a lesson about how to
# organize code in this choice somewhere...
struct ExpectedJointNll{C} <: Function
  cfg::C
  data_minus_z0::Matrix{Float64}
  presolved_saa_sumsq::Float64
end

# Trying to move to callable structs instead of closures so that the
# precompilation can be better...
function (E::ExpectedJointNll{C})(p) where{C}
  em_ejnll(E.cfg, p, E.data_minus_z0, E.presolved_saa_sumsq)
end

"""
prepare_z0_SR0(cfg::VecchiaConfig, arg::AbstractVector) -> (z0, SR0)

Compute Sig*(Sig + R)^{-1} z
"""
function prepare_z0_SR0(cfg, arg, data)
  n    = size(data, 1)
  s2   = arg[end]
  U    = sparse(Vecchia.rchol(cfg, arg, issue_warning=false))
  SR   = Symmetric(U*U' + inv(s2)*I) 
  SRf  = cholesky(SR, perm=1:n) # for now
  (SRf\(data./s2), SRf)
end

function em_step(cfg, arg, saa, optimizer; optimizer_kwargs...)
  # Get the data and points and compute z_0 = E_{arg}[ z | y ].
  dat   = reduce(vcat, cfg.data) 
  pts   = reduce(vcat, cfg.pts)
  (z0, SR0f) = prepare_z0_SR0(cfg, arg, dat)
  # Pre-solve the SAA factors (SCALED BY inv(sqrt(2*M))). 
  # Note that I need to compute two traces that look like tr( [U * U^T] *
  # [(PtL)*(PtL^T)]^{-1}). Re-arranging those things, that means I am computing
  # tr(M*M^T), where M=U'*(PtL)^{-T}. So that's what the pre-solve is here.
  divisor = sqrt(size(saa,2))
  pre_sf_solved_saa = (SR0f.PtL'\saa)./divisor
  pre_sf_solved_saa_sumsq = sum(_square, pre_sf_solved_saa)/2
  # Now, unlike the v1 version, create a NEW configuration where the "data"
  # field is actually hcat(z0, pre_sf_solved_saa./sqrt(2*M)). This division is
  # important because the qform calculator in Vecchia.nll doesn't know which
  # columns are data vs saa vectors, and so that division has to happen before
  # hand.
  tmp_cfg = augmented_em_cfg(cfg, z0, pre_sf_solved_saa)
  # create the closure that evaluates the expected joint nll:
  dat_minus_z0 = dat-z0
  ejnll = ExpectedJointNll(tmp_cfg, dat_minus_z0, pre_sf_solved_saa_sumsq)
  optimizer(ejnll, arg; optimizer_kwargs...)
end

