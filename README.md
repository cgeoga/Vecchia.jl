
# Vecchia.jl

This package offers a flexible and optimized framework for fitting parametric
Gaussian process models to large datasets using the linear-cost [*Vecchia
approximation*](https://en.wikipedia.org/wiki/Vecchia_approximation). This
library offers several features that set it apart from others in the space:
- Because it is written in Julia, you can provide any covariance function you
  want (as opposed to being restricted to a list of pre-implemented models)
- So long as your covariance function is amenable, the approximation is
  compatible with autodiff and can give you gradients and Hessians without any
  additional coding on your part (and still with good parallelization!).
- It is implemented in a way that allows *chunking*, where the fundamental unit
  in each small likelihood approximation can be a vector of measurements, not
  just a singleton. 
- It implements a special EM algorithm-based refinement method for process
  models with noise.
- Because of how great the Julia optimization ecosystem is, you can hook into
  *much* more powerful optimizers than is possible in, for example, R.

The fundamental object is a `Vecchia.VecchiaConfig`, which specifies your
conditioning sets for each sub-problem and also acts as a functor with methods
for the negative log-likelihood. Here is a pseudocode example specifying a
Vecchia approximation with a random ordering and k nearest neighbor conditioning
sets.
```julia
using LinearAlgebra, StaticArrays, Vecchia
using BesselK # provides the AD-compatible Matern model 

# VERY IMPORTANT FOR MULTITHREADING, since this is many small BLAS/LAPACK calls:
BLAS.set_num_threads(1)

# Say you have:
# pts::Vector{SVector{D,Float64}}, which are the locations of your measurements.
# data::Matrix{Float64}, where each column is an iid replicate from the process.

# The covariance function, in this case Matern. You can provide any function
# here, but it should have this signature of (location_1, location_2, params).
kernel(x,y,p) = matern(x,y,p)

# Pick the number of conditioning points you want to use in each sub-problem.
# If you want to use a different number for different problems, you can also
# provide a vector of different values.
k = 10

# There are several options for constructing the configuration object that use
# different strategies for designing conditioning sets. But in general, I think
# it's hard to go wrong with this one and would suggest it as a default.
const cfg = knnconfig(data, pts, k, kernel)
```
To keep the base package lean, estimation tools are partially given via
extensions, meaning that you will have to load other packages for individual
optimizers and such separately. My default recommendation is to use
[Uno](https://github.com/cvanaret/uno) with the
[NLPModels.jl](https://github.com/JuliaSmoothOptimizers/NLPModels.jl) framework.
Here is a demonstration of using that extension framework and fitting your model
with `Uno`:
```julia
# Instead of UnoSolver, could also use, e.g., NLPModelsIpopt, or NLPModelsKNITRO.
using ForwardDiff, NLPModels, UnoSolver # necessary to load extension(s)

solver  = NLPModelsSolver(uno; preset="filtersqp")
#solver = NLPModelsSolver(ipopt; tol=1e-4) # for Ipopt
#solver = NLPModelsSolver(knitro; algorithm=4 # for KNITRO
mle     = vecchia_estimate(cfg, some_init, solver)
```
But feel free to bring your own solver! There are also extensions for the JuMP
framework. Moreover, `cfg` has methods for the likelihood itself and can be
treated directly as an objective function, so you can hand it to anything that
eats a function a returns a minimizer.

**See the example files for heavily commented demonstrations.**

# Additional functionality

## Sparse precision matrix and ("reverse") Cholesky factors

While it is not necessary for evaluating the negative log-likelihood, in some
settings it can be useful to use this approximation to produce a sparse
approximation to the model-implied precision matrix. For this purpose,
`rchol(cfg, params)` gives a sparse Upper triangular matrix `U` (the "reverse"
Cholesky factor) such that your precision is approximated with `U*U'`.  **Note
that these objects correspond to permuted data, though, not the ordering in
which you provided the data**.
```julia
using SparseArrays 

# Note that the direct output of Vecchia.rchol is an internal object with just
# a few methods. But this sparse conversion will give you a good old SparseMatrixCSC.
U = UpperTriangular(sparse(Vecchia.rchol(vecc, sample_p)))
```
You'll get a warning the first time you call `rchol` re-iterating the issue
about permutations. If you want to avoid that, you can pass in the kwarg
`issue_warning=false`.


## Estimation with a nugget/measurement error

As mentioned above, measurement error can really hurt the accuracy of these
approximations. If your model is effectively given by `data(x) = good_gp(x) +
iid_noise(x)`, where `good_gp` is something that screens well that you actually
want to use Vecchia on and `iid_noise` has VARIANCE `eta^2`, then you can
estimate all parameters, including `eta^2`, with the built in EM algorithm
procedure that is demonstrated in `./example/example_estimate_noise.jl`. See
also the [paper](https://arxiv.org/abs/2208.06877) for a lot more information.

This method works equally well for **any** perturbation whose covariance matrix
admits a fast solve, although ideally also a fast log-determinant. The code now
allows you to provide an arbitrary struct for working with the error covariance
matrix, and you can inspect `./src/errormatrix.jl` for a demonstration of the
methods that you need to provide that struct for everything to "just work".

**If you use this method, please cite [this paper](https://arxiv.org/abs/2208.06877)**.

# Advanced usage

## Expensive or Complicated Kernel Functions

`Vecchia.jl` is pretty judicious about when and where the covariance function is
evaluated. For sufficiently fancy kernels that involve a lot of
side-computations or carrying around additional objects, there might be some
performance to be gained by "specializing" the internal function
`Vecchia.updatebuf!`, which is the only place where the kernel function is
called. Here is an example of this syntax:
```julia

# Create some struct to carry around all of your extra pieces that, for example,
# would otherwise need to be computed redundantly.
struct MyExpensiveKernel
  # ... 
end

# Now write a special method of Vecchia.updatebuf!. This might technically be
# type piracy, but I won't tell anybody if you won't.
#
# Note that you could also instead do fn::typeof(myspecificfunction) if you just
# wanted a special method for one specific function instead of a struct.
function Vecchia.updatebuf!(buf, pts1, pts2, fn::MyExpensiveKernel,
                            params; skipltri=false)
  println("Wow, neat!") 
  # ... (now do things to update buf)
end

# Create Vecchia config object:
const my_vecc_config = Vecchia.kdtreeconfig(..., MyExpensiveKernel(...))

# Now when you call this function, you will see "Wow, neat!" pop up every time
# that Vecchia.updatebuf! gets called. Once you're done testing and want to
# actually go fast, I would obviously recommend getting rid of the print
# statement.
Vecchia.nll(my_vecc_config, params)
```
In general, this probably won't be necessary for you. But I know I for one work
with some pretty exotic kernels regularly. And from experience I can attest
that, with some creativity, you can really cram a lot of efficient complexity
into the approximation with this approach without having to develop any new
boilerplate.

## Mean functions

...are currently not super officially supported. But you can now pass AD through
the `VecchiaConfig` struct itself. So a very simple hacky way to get your mean
function going would be a code pattern like
```julia
# see other examples for the rest of the args to the kdtreeconfig and stuff.
function my_nonzeromean_nll(params, ...)
  parametric_mean = mean_function(params, ...) 
  cfg = Vecchia.kdtreeconfig(data - parametric_mean, ...) 
  Vecchia.nll(cfg, params)
end
```
This will of course mean you rebuild the `VecchiaConfig` every time you evaluate
the likelihood, which isn't ideal and is why I say that mean functions aren't
really in this package yet. But then, at least the generic KD-tree configs get
built pretty quickly, and so if you have enough data that Vecchia approximations
are actually helpful, you probably won't feel it too much. And now you can just
do `ForwardDiff.{gradient, hessian}(my_nonzeromean_nll, params)` without any
additional code. If you wanted to fit billions of points, this probably isn't
taking the problem seriously enough. But until your data sizes get there, this
slight inefficiency probably won't be the bottleneck either.

I'm very open to feedback/comments/suggestions on the best way to incorporate
mean functions. It just isn't obvious to me how best to do it, and I don't
really need them myself (at least, not beyond what I can do with this current
pattern) so I'm not feeling super motivated to think hard about the best design
choice.

# Wanted/planned changes (contributions welcome!)

- More docstrings!
- It might be nice to add a new `ScalarVecchiaConfig` or something similar for
  cases where the prediction sets are singletons. The `VecchiaConfig` object has
  a bit of extra indirection that is necessary for chunked prediction sets.
  But maybe with a little reworking something simpler could be given in the
  scalar case.
- Conditional simulations were recently added, but that implementation would
  hugely benefit from somebody kicking the tires and playing with details and
  smart defaults/guardrails.
- Prediction design is pretty hacky at this point. A more careful look at the
  literature in this space and a better design would be good.
- It would be interesting to at some point benchmark the potential improvement
  from using memoization for kernel evaluations (with a potential extra twist of
  memoizing over stationary kernel evaluations as well). In the rchol approach,
  there is the `use_tiles={true,false}` kwarg, which effectively does manual
  book-keeping to avoid ever evaluating the kernel for the same pair of points
  twice. But it may be more elegant and just as fast to use memoization. This is
  probably 10-20 lines of code and an hour to benchmark and play with, so it would
  be a great first way to tinker with Vecchia stuff.
- API refinement/seeking feedback. For the most part, it is me and students in
  my orbit that use this package. But I'd love for it to be more widely adopted,
  and so I'd love for the interface to be polished. For example: figuring out
  how best to more properly support mean functions would be nice. Another
  example: I've just put a bunch of print warnings in the code about permutation
  footguns. But obviously it would be better to just somehow design the
  interface that there is no chance of a user getting mixed up by that.

# Citation

If you use this software in your work, **particularly if you actually use
second-order optimization with the real Hessians**,  please cite the package itself:
````
@software{Geoga_Vecchia_jl,
  author = {Geoga, Christopher J.},
  title  = {Vecchia.jl},
  url    = {https://github.com/cgeoga/Vecchia.jl},
  year   = {2021},
  publisher = {Github}
}
````
I would also be curious to see/hear about your application if you're willing to
share it or take the time to tell me about it. 

