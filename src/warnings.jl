
function notify_disable(kw)
  @info "You can turn off this warning with the kwarg $kw." maxlog=1
end

function notify_optfail(ret)
  @warn "Optimizer failed to converge and returned status $ret. This isn't necessarily a problem, but if it keeps happening perhaps you should investigate."
end

const NOTATION_WARNING = "In the notation of the paper introducing this method, this code currently only supports R = η^2 I, where the last parameter in your vector is η^2 directly. Support for more generic R matrices is easy and incoming, but if you need it now please open an issue or PR or otherwise poke me (CG) somehow."

const FALLBACK_WARNING = "Trying to generate decent init with generic fallback defaults..."

const FALLBACK_FAIL_WARNING = "Even with fallback inits, something went wrong generating the init. Maybe you should check your model/code/data basic things again before continuing."

const OPTIMIZER_WARNING = "Since trust region methods work much better than linesearch ones for this problem in my experience, the default optimizer here is a simple TR method that I (CG) coded by hand. That is not ideal, and if you have any issues please experiment with different optimizers. Please open a github issue for more discussion."

const THREAD_WARN = "It looks like you started Julia with multiple threads but are also using multiple BLAS threads. The Julia multithreading isn't composable with BLAS multithreading, so please run BLAS.set_num_threads(1) before executing this function."

const BOUNDS_WARN = "This function defaults to lower bounds on parameters of 1e-5 as a sensible default. But if that's not right for your problem, you can pass in a vector of lower (and upper) bounds with box_lower=[...] and box_upper=[...]."

const RCHOL_INSTANTIATE_ERROR = "This instantiation function makes extensive use of in-place algebraic operations and makes certain assumptions about the values of those buffers coming in. Please make a new struct to pass in here, or manually reset your current one."

const RCHOL_WARN = "Note that this is the reverse Cholesky factor for your data enumerated according to the permutation of the VecchiaConfig structure, so if you plan to apply this to vectors be sure to be mindful of potentially re-permuting. The simplest way to permute your data correct is with reduce(vcat, my_config.data)."

const EM_NONUG_WARN = "Your perturbation/error variance is zero, which violates the requirement that its covariance matrix be invertible. Perhaps you could get away with modeling without the nugget effect after all? Alternatively, check your code for the covariance function perhaps."

