
# TODO (cg 2022/06/03 11:52): This should probably ust be the inner objective,
# and the closure is made in em_step.
"""
prepare_inner_obj_extrace(cfg, y, z0, SR0f) -> inner_obj(p)

Prepare the E function to be optimized in the M step. Equation 2.2 \
in Proposition 1 of the manuscript.
"""
function prepare_inner_obj_extrace(cfg, y, z0, SR0f)
  SR0f_inv_dense = Symmetric(inv(Matrix(sparse(SR0f))))
  p -> begin
    s2 = p[end]
    U  = Vecchia.rchol(cfg, p)
    Ud = UpperTriangular(Matrix(sparse(U)))
    UdUdt = Ud*Ud'
    R     = Diagonal(fill(s2, length(z0)))
    R_nll = 0.5*(logdet(R) + dot(y-z0, R\(y-z0)))
    R_tr  = 0.5*(tr(SR0f_inv_dense*R))
    S_nll = Vecchia.nll(U, z0)
    S_tr  = 0.5*tr(SR0f_inv_dense*UdUdt) # for now
    R_nll + R_tr + S_nll + S_tr
  end
end

# Temporary. Used for testing.
#
# Status: Looks good. Now just need to prepare a Vecchia version of this. The
# first test should be with a cfg that gives back the exact likelihood, and then
# the simulator should take a seed and I should try to get back the exact same
# results. Second step will be to then verify it for non-trivial Vecchia.
function prepare_inner_obj_exact(cfg, arg)
  # Get what we need here:
  dat   = vec(reduce(vcat, cfg.data)) # for now, assuming just one sample
  pts   = reduce(vcat, cfg.pts)
  Sa    = Symmetric([cfg.kernel(x, y, arg) for x in pts, y in pts])
  Ra    = Diagonal(fill(arg[end], length(pts)))
  SRa   = Symmetric(Sa+Ra)
  SRaf  = cholesky(SRa)
  cmu   = Sa*(SRaf\dat)
  ccov  = Symmetric(Sa - Sa*(SRaf\Sa))
  ccovf = cholesky(ccov)
  # Now prepare the inner objective, the expected JOINT nll under the
  # conditional law:
  ejnll = (p, verbose=false) -> begin
    Sp  = cholesky(Symmetric([cfg.kernel(x, y, p) for x in pts, y in pts]))
    Rp  = Diagonal(fill(p[end], length(pts)))
    iSp = inv(Sp)
    iRp = inv(Rp)
    tr_term  = 0.5*tr(ccov*(iSp + iRp))
    nll_term = generic_nll(Sp, cmu) + generic_nll(Rp, dat-cmu)

    if verbose
      println("nll_term: $nll_term")
      println("tr_term:  $tr_term")
    end

    tr_term+nll_term
  end
  (cond_mean=cmu, cond_var=ccov, pts=pts, dat=dat, enll=ejnll)
end

function prepare_inner_obj_vecc_debug(cfg, arg)
  # Get what we need here:
  dat   = vec(reduce(vcat, cfg.data)) # for now, assuming just one sample
  pts   = reduce(vcat, cfg.pts)
  Sa    = vecchia_sigma(cfg, arg) 
  Ra    = Diagonal(fill(arg[end], length(pts)))
  SRa   = Symmetric(Sa+Ra)
  SRaf  = cholesky(SRa)
  cmu   = Sa*(SRaf\dat)
  ccov  = Symmetric(Sa - Sa*(SRaf\Sa))
  ccovf = cholesky(ccov)
  # Now prepare the inner objective, the expected JOINT nll under the
  # conditional law:
  ejnll = (p, verbose=false) -> begin
    Sp  = cholesky(vecchia_sigma(cfg, p))
    Rp  = Diagonal(fill(p[end], length(pts)))
    iSp = inv(Sp)
    iRp = inv(Rp)
    tr_term  = 0.5*tr(ccov*(iSp + iRp))
    nll_term = generic_nll(Sp, cmu) + generic_nll(Rp, dat-cmu)

    if verbose
      println("nll_term: $nll_term")
      println("tr_term:  $tr_term")
    end

    tr_term+nll_term
  end
  (cond_mean=cmu, cond_var=ccov, pts=pts, dat=dat, enll=ejnll)
end

function prepare_inner_obj_vecc_exacttr(cfg, arg)
  # Get what we need here:
  dat   = vec(reduce(vcat, cfg.data)) # for now, assuming just one sample
  pts   = reduce(vcat, cfg.pts)
  (z0, SR0f) = prepare_z0_SR0(cfg, arg, dat)
  # Now prepare the inner objective, the expected JOINT nll under the
  # conditional law:
  ejnll = (p, verbose=false) -> begin
    U    = Vecchia.rchol(cfg, p)
    Us   = sparse(U)
    iSp  = Matrix(Us*Us')
    Rp  = Diagonal(fill(p[end], length(pts)))
    tr_term  = 0.5*tr(SR0f\(iSp + inv(Rp)))
    nll_term = Vecchia.nll(U, hcat(z0)) + generic_nll(Rp, dat-z0)

    if verbose
      println("nll_term: $nll_term")
      println("tr_term:  $tr_term")
    end

    tr_term+nll_term
  end
  (cond_mean=z0, cond_var=Symmetric(inv(Matrix(sparse(SR0f)))), 
   pts=pts, dat=dat, enll=ejnll)
end

function prepare_inner_obj_vecc_saa(cfg, arg, saa, saa_normalized=false)
  # Get what we need here:
  dat   = vec(reduce(vcat, cfg.data)) # for now, assuming just one sample
  pts   = reduce(vcat, cfg.pts)
  (z0, SR0f) = prepare_z0_SR0(cfg, arg, dat)
  # Now prepare the inner objective, the expected JOINT nll under the
  # conditional law:
  ejnll = (p, verbose=false) -> begin
    U   = Vecchia.rchol(cfg, p)
    Rp  = Diagonal(fill(p[end], length(pts)))
    nll_term = Vecchia.nll(U, hcat(z0)) + generic_nll(Rp, dat-z0)

    tr_term_U = sum(z->dot(z, SR0f\Vecchia.applyUUt(U,hcat(z))), eachcol(saa))/2
    tr_term_R = sum(z->dot(z, SR0f\(Rp\z)), eachcol(saa))/2
    tr_term   = tr_term_U + tr_term_R
    if !saa_normalized
      tr_term  /= size(saa,2)
    end

    if verbose
      println("nll_term:   $nll_term")
      println("tr_term:    $tr_term")
      println("tr_term_U:  $tr_term_U")
      println("tr_term_R:  $tr_term_R")
    end

    tr_term+nll_term
  end
  (cond_mean=z0, cond_var=Symmetric(inv(Matrix(sparse(SR0f)))), 
   pts=pts, dat=dat, enll=ejnll)
end

function exact_vecchia_nll_noad_debug(cfg, arg)
  dat = reduce(vcat, cfg.data)
  S   = Symmetric(inv(Matrix(Vecchia.precisionmatrix(cfg, arg, issue_warning=false))))
  R   = Diagonal(fill(arg[end], length(dat)))
  Sf  = cholesky(S)
  SRf = cholesky(Symmetric(S+R))
  # numerator: joint nll.
  SRJ   = [S+R S; S S]
  SRJf  = cholesky(Symmetric(SRJ))
  yz    = vcat(dat, zeros(length(dat)))
  numer = 0.5*(logdet(SRJf) + dot(yz, SRJf\yz))
  # denominator:
  zhat  = S*(SRf\dat)
  Sc    = cholesky(Symmetric(S - S*(SRf\S)))
  denom = 0.5*(logdet(Sc) + dot(zhat, Sc\zhat))
  # final value:
  numer - denom
end

function debug_conditional(cfg, arg, data)
  n    = length(data)
  s2   = arg[end]
  S    = Vecchia.precisionmatrix(cfg, arg, issue_warning=false)
  iR   = Diagonal(fill(inv(s2), n))
  SR   = S + iR
  Sf   = cholesky(S, perm=1:n)
  SRf  = cholesky(SR, perm=1:n) # for now
  (z=SRf\(data./s2), SR=SR, SRf=SRf, Sf=Sf, iR=iR)
end

function cond_nll_spec(kernel, pts, cpts, dat, cdat, p)
  K_cc = cholesky!(Symmetric([kernel(x, y, p) for x in cpts, y in cpts]) + I*p[4])
  K_cp = [kernel(x, y, p) for x in cpts, y in pts]
  K_pp = Symmetric([kernel(x, y, p) for x in pts, y in pts] + I*p[4])
  K_cond = cholesky!(Symmetric(K_pp - K_cp'*(K_cc\K_cp)))
  generic_nll(K_cond, dat - K_cp'*(K_cc\cdat))
end

function cond_nll_compare(pts::Vector{SVector{D,Float64}}, 
                          data::Vector{Float64},
                          kernel,
                          pred_ixs::UnitRange{Int64}, 
                          vecchia_cond_ixs, test_params) where{D}
  @assert issorted(pred_ixs) "please sort prediction set indices"
  @assert issorted(vecchia_cond_ixs) "please sort conditioning set indices"
  @assert isdisjoint(pred_ixs, vecchia_cond_ixs) "prediction and cond sets not disjoint."
  # The exact cond nll:
  ex_cnll = p -> begin
    full_cond_ixs = 1:(pred_ixs[1]-1)
    cpts = pts[full_cond_ixs]
    ppts = pts[pred_ixs]
    cdat = data[full_cond_ixs]
    pdat = data[pred_ixs]
    cond_nll_spec(kernel, ppts, cpts, pdat, cdat, p)
  end
  # Now the approximated cond nll:
  vc_cnll = p -> begin
    cpts = pts[vecchia_cond_ixs]
    ppts = pts[pred_ixs]
    cdat = data[vecchia_cond_ixs]
    pdat = data[pred_ixs]
    cond_nll_spec(kernel, ppts, cpts, pdat, cdat, p)
  end
  (ForwardDiff.gradient(ex_cnll, test_params),
   ForwardDiff.gradient(vc_cnll, test_params))
end

