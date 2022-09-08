
using NLopt, ForwardDiff
include("../shared.jl")
include("kernel.jl")

Base.floatmax(x::ForwardDiff.Dual{T,V,N}) where{T,V,N} = floatmax(V)

function maximin_permutation(pts)
  ptsm = hcat(getindex.(pts, 1), getindex.(pts, 2))
  writedlm("tmp.csv", ptsm, ',')
  run(`Rscript maximin.R`)
  run(`rm -f tmp.csv`)
  perm = vec(Int64.(readdlm("tmp_maximin_order.csv", ',')))
  run(`rm -f tmp_maximin_order.csv`)
  perm
end

function simulate_setup(init_seed, npts, true_parms)
  rng     = StableRNG(init_seed)
  pts     = rand(rng, SVector{2,Float64}, npts)
  K       = Symmetric([matern_ganiso_withnugget(x,y,true_parms) for x in pts, y in pts])
  Kf      = cholesky!(K)
  data    = Kf.L*randn(rng, length(pts))
  saa     = rand(rng, (-1.0, 1.0), length(pts), 72)
  mmperm  = maximin_permutation(pts)
  pts_s   = pts[mmperm]
  dat_s   = data[mmperm]
  cfg     = maximinconfig(matern_ganiso_nonugget, pts_s, dat_s, 1, 5)
  (pts=pts, data=data, cfg=cfg, saa=saa)
end

if !(@isdefined SETUP)
  const TRUE_PARAMS = [5.0, 5.0, 3.0, 0.5, 0.025, 0.09, 1.25, 0.5]
  const INIT_PARAMS = ones(length(TRUE_PARAMS))
  const SETUP = simulate_setup(1234, 2_500, TRUE_PARAMS)
  const CFG   = SETUP.cfg
  const SAA   = SETUP.saa
end

function neldermead_objective(p,g)
  length(g)>0 && throw(error("This shouldn't happen"))
  try
    out = EMVecchia2.exact_vecchia_nll_noad(CFG, p)
    return out
  catch er
    println("Broke with error")
    display(er)
  end
end

function fit_neldermead(;method=:LN_NELDERMEAD, maxeval=1_000)
  opt = Opt(method, length(INIT_PARAMS))
  opt.lower_bounds  = [1e-4, 0.0, 0.0, -pi, 1e-4, 1e-4, 0.4, 0.0]
  opt.upper_bounds  = [Inf, Inf, Inf, pi,  Inf,  Inf,  Inf, Inf]
  opt.min_objective = neldermead_objective
  opt.ftol_rel = 1e-8
  opt.xtol_rel = 1e-8
  opt.ftol_abs = 1e-8
  opt.xtol_abs = 1e-8
  opt.maxeval  = maxeval
  optimize(opt, INIT_PARAMS)
end

function prepare_table(; compute_emle=false)
  est_nlm = fuck_nlm #fit_neldermead()
  est_emv = fuck #EMVecchia2.em_estimate(CFG, SAA, INIT_PARAMS, return_trace=true, max_em_iter=30)
  nlm_nll = EMVecchia2.exact_nll(CFG, est_nlm[2])
  emv_nll = EMVecchia2.exact_nll(CFG, est_emv[2][end])
  if compute_emle
    eml = EMVecchia2.exact_mle(CFG, INIT_PARAMS).minimizer
    eml_nll = EMVecchia2.exact_nll(CFG, eml)
  else
    eml = fill(NaN, length(INIT_PARAMS))
    eml_nll = NaN
  end
  table_body = vcat(INIT_PARAMS', TRUE_PARAMS', eml', est_emv[2][end]', est_nlm[2]')
  col_titles = ["\\sigma_1", "\\sigma_2", "\\sigma_3", "\\theta", "\\lambda_1", 
                "\\lambda_2", "\\nu", "\\eta^2"]
  row_titles = ["initialization", "true parameters",
                string("Exact MLE (\\ell(\\hat{\\bm{\\theta}}) = ", round(eml_nll, digits=4), ")"),
                string("EM (\\ell(\\hat{\\bm{\\theta}}) = ", round(emv_nll, digits=4), ")"),
                string("Nelder-Mead (\\ell(\\hat{\\bm{\\theta}}) = ", round(nlm_nll, digits=4), ")")]
  (table_body, col_titles, row_titles)
end

