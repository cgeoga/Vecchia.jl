
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
end

function Base.display(M::EMVecchiaIterable)
  display(M.cfg)
  println("EM with:")
  println("  - SAA vectors:        $(size(M.saa, 2))")
  println("  - ℓ₂ convergence tol: $(M.norm2tol)")
  println("  - Maximum iterations: $(M.max_em_iter)")
  println("  - solver:             $(M.solver)")
end

function EMiterable(cfg, init, saa, errormodel, solver, box_lower; kwargs...)
  kwargsd = Dict(kwargs)
  EMVecchiaIterable(cfg, copy(init), copy(init), saa, 
                    Ref(:NOT_CONVERGED),
                    get(kwargsd, :norm2tol,    1e-2),
                    get(kwargsd, :max_em_iter, 20),
                    errormodel, solver, box_lower)
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
  newstep = em_step(it.cfg, it.step_new, it.saa, it.errormodel, it.solver)
  it.step_old .= it.step_new
  it.step_new .= newstep
  (newstep, iteration+1)
end

function em_refine(cfg, errormodel, saa, init; solver, 
                   box_lower, verbose=true, kwargs...)
  iter = EMiterable(cfg, init, saa, errormodel, solver, box_lower; kwargs...)
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
                     warn_notation=true, verbose=true,
                     norm2tol=1e-2, max_em_iter=20)
  if warn_notation
    notify_disable("warn_notation=false")
    @warn NOTATION_WARNING maxlog=1
end
  # compute initial estimator using Vecchia with the nugget and nothing thoughtful:
  verbose && println("\nComputing initial MLE with standard Vecchia...")
  em_init = try
    vecchia_estimate_nugget(cfg, init, solver, errormodel; box_lower=box_lower)
  catch
    @warn "Computing the Vecchia-with-nugget initializer failed, falling back to your provided init..."
    init
  end
  # now use the iterator interface:
  (init_result=em_init,
   em_refine(cfg, errormodel, saa, em_init; solver=solver, box_lower=box_lower,
             verbose=verbose, norm2tol=norm2tol, max_em_iter=max_em_iter)...)
end

