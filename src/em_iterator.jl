
struct EMVecchiaIterable{H,D,F,O,R}
  cfg::VecchiaConfig{H,D,F}
  step_new::Vector{Float64}
  step_old::Vector{Float64}
  saa::Matrix{Float64}
  status::Ref{Symbol}
  norm2tol::Float64
  max_em_iter::Int64
  errormodel::R
  optimizer::O
  optimizer_kwargs::Dict{Symbol,Any}
end

function Base.display(M::EMVecchiaIterable)
  display(M.cfg)
  println("EM with:")
  println("  - SAA vectors:        $(size(M.saa, 2))")
  println("  - ℓ₂ convergence tol: $(M.norm2tol)")
  println("  - Maximum iterations: $(M.max_em_iter)")
  println("  - optimizer:          $(M.optimizer)")
end

function EMiterable(cfg, init, saa, errormodel; kwargs...)
  kwargsd = Dict(kwargs)
  EMVecchiaIterable(cfg, 
                    copy(init),
                    copy(init),
                    saa, 
                    Ref(:NOT_CONVERGED),
                    get(kwargsd, :norm2tol,    1e-2),
                    get(kwargsd, :max_em_iter, 20),
                    errormodel,
                    get(kwargsd, :optimizer, sqptr_optimize),
                    Dict{Symbol,Any}(get(kwargsd, :optimizer_kwargs, ())))
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
  newstep = em_step(it.cfg, it.step_new, it.saa, it.errormodel, 
                    it.optimizer; it.optimizer_kwargs...)
  # Check for bad returns:
  if !in(newstep.status, (0,1)) 
    if haskey(newstep, :error)
      err = newstep.error
      if !isnothing(err)
        println("Your optimizer returned an error:")
        throw(err)
      end
    end
    # TODO (cg 2023/01/20 15:04): I'm not sure whether or not letting users know
    # when the return code wasn't in (0,1) is best. If there is no error and the
    # iteration improves at all, maybe I shouldn't?
    notify_optfail(newstep.status)
  end
  it.step_old .= it.step_new
  it.step_new .= newstep.minimizer
  (newstep.minimizer, iteration+1)
end

function em_refine(cfg, errormodel, saa, init; verbose=true, kwargs...)
  iter = EMiterable(cfg, init, saa, errormodel; kwargs...)
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
      pretty_print_vec(new_est, true)
      println("  Step difference norm: $(norm(iter.step_new - iter.step_old))")
    end
    push!(path, new_est)
  end
  (status=iter.status[], path=path)
end

function em_estimate(cfg, saa, init; 
                     errormodel,
                     warn_optimizer=true,
                     warn_notation=true,
                     verbose=true,
                     optimizer=sqptr_optimize, 
                     norm2tol=1e-2,
                     max_em_iter=20,
                     optimizer_kwargs=())
  if warn_notation
    notify_disable("warn_notation=false")
    @warn NOTATION_WARNING maxlog=1
end
  # compute initial estimator using Vecchia with the nugget and nothing thoughtful:
  verbose && println("\nComputing initial MLE with standard Vecchia...")
  mle_withnugget = vecchia_estimate_nugget(cfg, init, optimizer, errormodel; 
                                           optimizer_kwargs...)
  if !in(mle_withnugget.status, (0,1))
    if haskey(mle_withnugget, :error)
      err = mle_withnugget.error
      !isnothing(err) && throw(err)
    end
  end
  if optimizer==sqptr_optimize && warn_optimizer
    notify_disable("warn_optimizer=false")
    @warn  OPTIMIZER_WARNING maxlog=1
  end
  # now use the iterator interface:
  (init_result=mle_withnugget,
   em_refine(cfg, errormodel, saa, mle_withnugget.minimizer; optimizer=optimizer, 
             verbose=verbose, norm2tol=norm2tol, max_em_iter=max_em_iter,
             optimizer_kwargs=optimizer_kwargs)...)
end

