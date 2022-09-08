
"""
prepare_z0_SR0(cfg::VecchiaConfig, arg::AbstractVector) -> (z0, SR0)

Compute Sig*(Sig + R)^{-1} z
"""
function prepare_z0_SR0(cfg, arg, data)
  n    = length(data)
  s2   = arg[end]
  S    = Vecchia.precisionmatrix(cfg, arg, issue_warning=false)
  SR   = S + Diagonal(fill(inv(s2), n))
  SRf  = cholesky(SR, perm=1:n) # for now
  (SRf\(data./s2), SRf)
end

function expected_joint_nll_symhutch(p, cfg, data, z0, presolved_saa, 
                                     presolved_saa_sumsq, saa_normalized)
  # Create the two matrices:
  U  = Vecchia.rchol(cfg, p, issue_warning=false)
  Rp = Diagonal(fill(p[end], length(z0)))
  # Compute the nll term:
  nll_term = Vecchia.nll(U, z0) + generic_nll(Rp, data-z0)
  # Compute the trace term:
  tr_term_U = sum(z->z^2, U'*presolved_saa)/2
  tr_term_R = presolved_saa_sumsq/(2*p[end])
  tr_term   = tr_term_U + tr_term_R
  saa_normalized || (tr_term /= size(presolved_saa,2))
  # Return the sum:
  tr_term+nll_term
end

function expected_joint_nll_symhutch_tronly(p, cfg, data, z0, presolved_saa, 
                                            presolved_saa_sumsq, saa_normalized)
  # Create the two matrices:
  U  = Vecchia.rchol(cfg, p, issue_warning=false)
  Rp = Diagonal(fill(p[end], length(z0)))
  # Compute the trace term:
  tr_term_U = sum(z->z^2, U'*presolved_saa)/2
  tr_term_R = presolved_saa_sumsq/(2*p[end])
  tr_term   = tr_term_U + tr_term_R
  saa_normalized || (tr_term /= size(presolved_saa,2))
  # Return the sum:
  tr_term
end

# TODO (cg 2022/07/06 14:48): At present, this doesn't work when p is dual,
# because the sparse PtL\[...] doesn't work (tmp2 will be dual when p is). So
# there's something to think about there. Could potentially just write the
# method myself I suppose. This is definitely more expensive, but I wonder if it
# is worth it to offer this at least when p isa Vector{BLASFloat} or something
# in case it makes the linesearch better?
function expected_joint_nll_hutchpp(p, cfg, data, z0, saa1, saa2, SR0f)
  # Create the two matrices:
  U  = Vecchia.rchol(cfg, p, issue_warning=false)
  Rp = Diagonal(fill(p[end], length(z0)))
  # Compute the nll term:
  nll_term = Vecchia.nll(U, z0) + generic_nll(Rp, data-z0)
  # Compute the trace term:
  tr_term  = hutchpp(z->begin
                       tmp1 = SR0f.PtL'\z
                       tmp2 = Vecchia.applyUUt(U, tmp1) + Rp\tmp1
                       SR0f.PtL\tmp2
                     end,
                     saa1, saa2)/2
  # Return the sum:
  tr_term+nll_term
end

"""
em_step(cfg, arg, data; kwargs...) -> arg'

Minimize (maximize) the negative (positive) E function and obtain the next\
estimator in the case of Picard iteration, or just the value F(arg) in\
the case of Anderson acceleration or whatever else.
"""
function em_step(cfg, arg, saa; saa_normalized=false,
                 trace_method=:symsaa, return_ejnll=false, 
                 optimizer=:KNITRO, solve_score=false, kwargs...)
  # Get what we need here:
  dat   = reduce(vcat, cfg.data) 
  pts   = reduce(vcat, cfg.pts)
  (z0, SR0f) = prepare_z0_SR0(cfg, arg, dat)
  # Pre-solve the SAA factors. Note that I need to compute two traces that look
  # like tr( [U * U^T] * [(PtL)*(PtL^T)]^{-1}). Re-arranging those things, that
  # means I am computing tr(M*M^T), where M=U'*(PtL)^{-T}. So that's what the
  # pre-solve is here.
  if trace_method == :symsaa
    pre_sf_solved_saa = SR0f.PtL'\saa
    pre_sf_solved_saa_sumsq = sum(z->z^2, pre_sf_solved_saa)
    # Now prepare the inner objective, the expected JOINT nll under the
    # conditional law:
    ejnll = p -> expected_joint_nll_symhutch(p, cfg, dat, z0, pre_sf_solved_saa,
                                             pre_sf_solved_saa_sumsq, saa_normalized)
  elseif trace_method == :hutchpp
    nsaa_half = div(size(saa,2),2)
    saa1  = saa[:,1:nsaa_half]
    saa2  = saa[:,(nsaa_half+1):end]
    ejnll = p -> expected_joint_nll_hutchpp(p, cfg, dat, z0, saa1, saa2, SR0f)
  elseif trace_method == :symsaa_tronly
    pre_sf_solved_saa = SR0f.PtL'\saa
    pre_sf_solved_saa_sumsq = sum(z->z^2, pre_sf_solved_saa)
    # Now prepare the inner objective, the expected JOINT nll under the
    # conditional law:
    ejnll = p -> expected_joint_nll_symhutch_tronly(p, cfg, dat, z0, pre_sf_solved_saa,
                                             pre_sf_solved_saa_sumsq, saa_normalized)
  else
    throw(error("The only options for trace_method at \
                the moment are :symsaa or :hutchpp."))
  end
  return_ejnll && return ejnll # this is for debugging.
  # Now optimize the expected joint nll:
  if optimizer == :IPOPT
    if solve_score
      println("Using a zero finder on the gradient, not using objective.")
      return ipopt_nlsolve_grad_multi(ejnll, arg; kwargs...)
    else
      return ipopt_optimize(ejnll, arg; kwargs...)
    end
  elseif optimizer == :KNITRO
    if solve_score
      return knitro_nlsolve_grad_multi(ejnll, arg; kwargs...)
    else
      return knitro_optimize(ejnll, arg; kwargs...)
    end
  else
    throw(error("Options are :IPOPT or :KNITRO."))
  end
end

# Doesn't work terribly, but also not amazing.
function em_anderson(cfg, init, saa; kwargs...)
  @warn "This really seems to work worse than straight Picard, just use for tests..."
  F = p -> em_step(cfg, p, saa; kwargs...).minimizer
  NLsolve.fixedpoint(F, init, show_trace=true, xtol=1e-4, ftol=1e-3, m=2, 
                     beta=0.75, store_trace=true, extended_trace=true)
end

