
struct EMVecchiaIterable{H,D,F,O,K}
  cfg::VecchiaConfig{H,D,F}
  step_new::Vector{Float64}
  step_old::Vector{Float64}
  saa::Matrix{Float64}
  status::Ref{Symbol}
  norm2tol::Float64
  max_em_iter::Int64
  optimizer::O
  optimizer_kwargs::K
end

function Base.display(M::EMVecchiaIterable{H,D,F}) where{H,D,F}
  display(M.cfg)
  println("EM with:")
  println("  - SAA vectors:        $(size(M.saa, 2))")
  println("  - ℓ₂ convergence tol: $(M.norm2tol)")
  println("  - Maximum iterations: $(M.max_em_iter)")
  println("  - optimizer:          $(M.optimizer)")
end

function EMiterable(cfg, init, saa; kwargs...)
  kwargsd = Dict(kwargs)
  EMVecchiaIterable(cfg, 
                    copy(init),
                    copy(init),
                    saa, 
                    Ref(:NOT_CONVERGED),
                    get(kwargsd, :norm2tol,    1e-2),
                    get(kwargsd, :max_em_iter, 20),
                    get(kwargsd, :optimizer, sqptr_optimize),
                    get(kwargsd, :optimizer_kwargs, ()))
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
  newstep = em_step(it.cfg, it.step_new, it.saa, it.optimizer; it.optimizer_kwargs...)
  # Check for bad returns:
  if !in(newstep.status, (0,1)) 
    @warn "Optimizer failed to converge and returned status $(newstep.status). This isn't necessarily a problem, but if it keeps happening perhaps you should investigate."
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

function vecchia_mle_withnugget(cfg, init, optimizer; optimizer_kwargs...)
  nugkernel = NuggetKernel(cfg.kernel) 
  nug_cfg   = Vecchia.VecchiaConfig(cfg.chunksize, cfg.blockrank,
                                    nugkernel, cfg.data, cfg.pts, cfg.condix)
  likelihood = WrappedLogLikelihood(nug_cfg)
  optimizer(likelihood, init; optimizer_kwargs...)
end

function em_estimate(cfg, saa, init; 
                     warn_optimizer=true,
                     warn_notation=true,
                     verbose=true,
                     optimizer=sqptr_optimize, 
                     norm2tol=1e-2,
                     max_em_iter=20,
                     optimizer_kwargs=())
  if warn_notation
    @info "You can turn off this warning with the kwarg warn_notation=false." maxlog=1
    @warn "In the notation of the paper introducing this method, this code currently only supports R = η^2 I, where the last parameter in your vector is η^2 directly. Support for more generic R matrices is easy and incoming, but if you need it now please open an issue or PR or otherwise poke me (CG) somehow." maxlog=1
end
  # compute initial estimator using Vecchia with the nugget and nothing thoughtful:
  verbose && println("\nComputing initial MLE estimator using Vecchia with nugget...")
  mle_withnugget = vecchia_mle_withnugget(cfg, init, optimizer; optimizer_kwargs...)
  # check for success:
  if !in(mle_withnugget.status, (0,1))
    @warn "Trying to generate decent init with generic fallback defaults..."
    mle_withnugget = vecchia_mle_withnugget(cfg, vcat(init[1:(end-1)], 0.01), optimizer; 
                                         optimizer_kwargs...)
    if !in(mle_withnugget.status, (0,1))
      @warn "Even with fallback inits, something went wrong generating the init. Maybe you should check your model/code/data basic things again before continuing."
    end
  end
  if optimizer==sqptr_optimize && warn_optimizer
    @info "You can turn off this warning with the kwarg warn_optimizer=false." maxlog=1
    @warn "Since trust region methods work much better than linesearch ones for this problem in my experience, the default optimizer here is a simple TR method that I (CG) coded by hand. That is not ideal, and if you have any issues please experiment with different optimizers. Please open a github issue for more discussion." maxlog=1
  end
  # now use the iterator interface:
  (init_result=mle_withnugget,
   em_refine(cfg, saa, mle_withnugget.minimizer; optimizer=optimizer, 
             verbose=verbose, norm2tol=norm2tol, max_em_iter=max_em_iter,
             optimizer_kwargs=optimizer_kwargs)...)
end

