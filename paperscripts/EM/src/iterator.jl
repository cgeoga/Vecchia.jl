
const ITER_DEFAULTS = Dict(:norm2tol=>1e-2, :max_em_iter=>20,
                           :ipopt_print_level=>0, :ipopt_tol=>1e-5,
                           :ipopt_box_l=>1e-3, :ipopt_box_u=>1e22, 
                           :ipopt_max_iter=>100, :optimizer=>:KNITRO,
                           :solve_score=>false)

struct EMVecchiaIterable{H,D,F}
  cfg::VecchiaConfig{H,D,F}
  step_new::Vector{Float64}
  step_old::Vector{Float64}
  saa::Matrix{Float64}
  norm2tol::Float64
  max_em_iter::Int64
  status::Ref{Symbol}
  ipopt_print_level::Int64
  ipopt_max_iter::Int64
  ipopt_tol::Float64
  ipopt_box_l::Vector{Float64}
  ipopt_box_u::Vector{Float64}
  optimizer::Symbol
  solve_score::Bool
end

function Base.display(M::EMVecchiaIterable{H,D,F}) where{H,D,F}
  display(M.cfg)
  println("EM with:")
  println("  - SAA vectors:        $(size(M.saa, 2))")
  println("  - ℓ₂ convergence tol: $(M.norm2tol)")
  println("  - Maximum iterations: $(M.max_em_iter)")
  println("  - optimizer:          $(M.optimizer)")
  println("  - score solver:       $(M.solve_score)")
  if M.optimizer == :IPOPT
    println("Ipopt with:")
    println("  - print_level: $(M.ipopt_print_level)")
    println("  - max_iter:    $(M.ipopt_max_iter)")
    println("  - tol:         $(M.ipopt_tol)")
    println("  - lower box:   $(M.ipopt_box_l)")
    println("  - upper box:   $(M.ipopt_box_u)")
  end
end

function EMiterable(cfg, init, saa; kwargs...)
  args = merge(ITER_DEFAULTS, Dict(kwargs))
  EMVecchiaIterable(cfg, 
                    copy(init),
                    copy(init),
                    saa, 
                    args[:norm2tol], 
                    args[:max_em_iter], 
                    Ref(:NOT_CONVERGED),
                    args[:ipopt_print_level], 
                    args[:ipopt_max_iter], 
                    args[:ipopt_tol],
                    prepare_bounds(length(init), args[:ipopt_box_l]),
                    prepare_bounds(length(init), args[:ipopt_box_u]),
                    args[:optimizer],
                    args[:solve_score])
end

function Base.iterate(it::EMVecchiaIterable{H,D,F}, iteration::Int=0) where{H,D,F}
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
  newstep = em_step(it.cfg, it.step_new, it.saa; 
                    print_level=it.ipopt_print_level,
                    max_iter=it.ipopt_max_iter,
                    tol=it.ipopt_tol,
                    box_lower=it.ipopt_box_l,
                    box_upper=it.ipopt_box_u,
                    optimizer=it.optimizer,
                    solve_score=it.solve_score)
  # Check for literal errors:
  if !isnothing(newstep.error)
    if newstep.status == 12345
      println("Ipopt optimizer failed with an error in user code, either in your \
              supplied code or from something in this package or its deps. Here \
              is the error:")
      throw(newstep.error)
    end
    it.status[] = ipopt_check_code(newstep.status)
    return nothing
  end
  # Check for bad returns:
  if !in(newstep.status, (0,1)) 
    @warn "Optimizer failed to converge and returned status $(newstep.status), \
    not killing the job but proceed with caution."
  end
  it.step_old .= it.step_new
  it.step_new .= newstep.minimizer
  (newstep.minimizer, iteration+1)
end

function em_refine(cfg, saa, init; verbose=true, kwargs...)
  iter = EMiterable(cfg, init, saa; kwargs...)
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
      println("$icount: $new_est")
      println("  Step difference norm: $(norm(iter.step_new - iter.step_old))")
    end
    push!(path, new_est)
  end
  (status=iter.status[], path=path)
end

function em_estimate(cfg, saa, init; verbose=true, kwargs...)
  # compute initial estimator using Vecchia with the nugget and nothing thoughtful:
  plevel = verbose ? 5 : 0
  verbose && println("\nComputing initial MLE estimator using Vecchia with nugget...")
  mle_withnugget = vecchia_mle_withnug(cfg, init; print_level=plevel)
  # check for success:
  init_res = check_nuggetvecchia_result(mle_withnugget)
  if !init_res
    @warn "Trying to generate decent init with generic fallback defaults..."
    mle_withnugget = vecchia_mle_withnug(cfg, [1.0, 0.1, 0.5, 0.01]; print_level=plevel)
    init_res = check_nuggetvecchia_result(mle_withnugget)
    if !init_res
      @warn "Even with fallback inits, something went wrong generating the init. So beware."
    end
  end
  # now use the iterator interface:
  (init_result=init_res,
   em_refine(cfg, saa, mle_withnugget.minimizer; verbose=verbose, kwargs...)...)
end

