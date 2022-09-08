
"""
prepare_z0_SR0(cfg::VecchiaConfig, arg::AbstractVector) -> (z0, SR0)

Compute Sig*(Sig + R)^{-1} z
"""
function prepare_z0_SR0(cfg, arg, data)
  n    = length(data)
  s2   = arg[end]
  S    = precisionmatrix(cfg, arg, issue_warning=false)
  SR   = S + Diagonal(fill(inv(s2), n))
  SRf  = cholesky(SR, perm=1:n) # for now
  (SRf\(data./s2), SRf)
end

function expected_joint_nll_symhutch(p, cfg, data, z0, presolved_saa, 
                                     presolved_saa_sumsq)
  # Create the two matrices:
  U  = rchol(cfg, p, issue_warning=false)
  Rp = Diagonal(fill(p[end], length(z0)))
  # Compute the nll term:
  nll_term = nll(U, z0) + generic_nll(Rp, data-z0)
  # Compute the trace term:
  tr_term_U = sum(z->z^2, U'*presolved_saa)/2
  tr_term_R = presolved_saa_sumsq/(2*p[end])
  tr_term   = (tr_term_U + tr_term_R)/size(presolved_saa,2)
  # Return the sum:
  tr_term+nll_term
end

"""
em_step(cfg, arg, data, optimizer; kwargs...) -> arg'

Minimize (maximize) the negative (positive) E function and obtain the next\
parameter value in the fixed point iteration problem.
"""
function em_step(cfg, arg, saa, optimizer; kwargs...)
  # Get what we need here:
  dat = reduce(vcat, cfg.data) 
  pts = reduce(vcat, cfg.pts)
  (z0, SR0f) = prepare_z0_SR0(cfg, arg, dat)
  # Pre-solve the SAA factors. Note that I need to compute two traces that look
  # like tr( [U * U^T] * [(PtL)*(PtL^T)]^{-1}). Re-arranging those things, that
  # means I am computing tr(M*M^T), where M=U'*(PtL)^{-T}. So that's what the
  # pre-solve is here.
  pre_sf_solved_saa = SR0f.PtL'\saa
  pre_sf_solved_saa_sumsq = sum(z->z^2, pre_sf_solved_saa)
  # Now prepare the inner objective, the expected JOINT nll under the
  # conditional law, and optimize it with respect to the new parameters:
  ejnll = p -> expected_joint_nll_symhutch(p, cfg, dat, z0, pre_sf_solved_saa,
                                           pre_sf_solved_saa_sumsq)
  optimizer(ejnll, arg; kwargs...)
end

