
# Vecchia.jl

This package offers a flexible and optimized framework for fitting parametric
Gaussian process models to large datasets using the linear-cost [*Vecchia
approximation*](https://en.wikipedia.org/wiki/Vecchia_approximation). This
library offers several features that set it apart from others in the space:
- Because it is written in Julia, you can provide any covariance function you
  want (as opposed to being restricted to a list of pre-implemented models).
- If autodiff works on your kernel function, it will work for the approximation
  (and have good parallel performance for gradients and Hessians!). See
  [BesselK.jl](https://github.com/cgeoga/BesselK.jl) for a
  `ForwardDiff.jl`-compatible Matern function, *including* the smoothness
  parameter.
- It is implemented in a way that allows *chunking*, where the fundamental unit
  in each small likelihood approximation can be a vector of measurements, not
  just a singleton. 
- It offers fully arbitrary mean functions, not just linear ones.
- Because of how great the Julia optimization ecosystem is, you can hook into
  *much* more powerful optimizers than is possible in, for example, R.

**See the example files for commented and run-able demonstrations.** 

## Simple usage demonstration (mean zero)

If you have data `data` measured at locations `pts::Vector{SVector{D,Float64}}`
and you want to fit the parameters of covariance function `kernel(pt_1, pt_2, params)`, 
here is a very simple template for doing so. This code is optimized to scale
very well with multiple threads, so please remember to start Julia with
`julia -t $NTHREADS` among your other arguments.
```julia
using Vecchia

# See below, docstrings, and examples for details on specifying options.
# This object itself also can be used as a function to give the negative
# log-likelihood, so `appx(params)` (and autodiff of appx) is supported.
appx = VecchiaApproximation(pts, kernel, data)

# Load in some extensions for estimation. To be modular, this package doesn't
# hard-code this dependence. The syntax for using alternative solvers is
# demonstrated in the example files.
using ForwardDiff, NLPModels, UnoSolver

# Specify a solver and hand everything to vecchia_estimate.
solver = NLPModelsSolver(uno; preset="filtersqp")
mle    = vecchia_estimate(cfg, some_init, solver)

# perhaps now you wanted to predict with your fitted model?
# (see the docstrings for options, but hopefully the defaults will serve you well!)
preds  = predict(appx, prediction_pts, mle)
```
It's that easy! Enjoy your linear-cost (approximate) MLEs, predictions,
conditional simulations, preconditioners, and more.

If you have a mean function, you have two options. You can write your functions
with
```julia
covfun(x, y, params) = # uses params[1:whatever]
meanfun(x, params)   = # manually uses params[(whatever+1):end]
```
in which case you can just modify your approximation constructor with
```julia
appx = VecchiaApproximation(pts, kernel, data; meanfun=meanfun)
```
and hand everything to the optimizer and get a `Vector{Float64}` back as usual.
Alternatively, if you want to write your functions as
```julia
covfun(x, y, params) = # uses params[1:whatever_it_needs]
meanfun(x, params)   = # manually uses params[1:whatever_it_needs]
```
then you should additionally modify your estimation code with
```julia
fancy_init = Parameters(cov_init, mean_init)
mle    = vecchia_estimate(cfg, fancy_init, solver)
```
Now what comes back will be a `Vecchia.Parameters` object, which has separate
fields `.cov_params` and `.mean_params`. You can also hand that object right to
`predict` instead of a `Vector{Float64}` and everything will work as expected.


## Sparse precision matrix and ("reverse") Cholesky factors

While it is not necessary for evaluating the negative log-likelihood, in some
settings it can be useful to use this approximation to produce a sparse
approximation to the model-implied precision matrix. For this purpose,
`rchol(appx, params)` gives a (permuted) sparse inverse Cholesky factor for your
matrix. If `ordering=NoPermutation()` in your `VecchiaApproximation`
constructor, then one precisely has the approximation that your precision matrix
is approximated with `U*U'`, where `U` is `UpperTriangular`. Build it like so:
```julia
using SparseArrays 

# This object has some standard methods: mul!/*, ldiv!/\, logdet, etc.
U = rchol(appx, sample_p)
```
Additionally, this package offers the convenience constructor of
`rchol_preconditioner`, which gives a preconditioner object for your implied
kernel matrix. Here is an example building the preconditioner and solving a
linear system with `Krylov.jl`.
```julia
using Krylov

# Note that we don't need to pass the data in here when building the
# VecchiaApproximation, as we won't need it.
#
# For a preconditioner, I recommend cranking up the number of conditioning
# points a bit past the defaults, which are tuned for likelihoods.
appx = VecchiaApproximation(pts, matern; conditioning=KNNConditioning(25))
pre  = rchol_preconditioner(appx, [5.0, 0.1, 2.25])

# the dense exact covariance matrix for comparison that will be used for cg.
M = [matern(x, y, [5.0, 0.1, 2.25]) for x in pts, y in pts]

# a sample RHS.
v = collect(1.0:length(pts))

# a Vecchia preconditioner in action (try re-running this without the
# preconditioner and see how slowly it converges...)
sol2 = cg(Symmetric(M), v; M=pre, ldiv=false, verbose=1) # ~ 15 iterations
```

## Approximation configuration options

The full syntax for the constructor looks like this:
```julia
appx = VecchiaApproximation(pts, kernel, data;
                            ordering=default_ordering(pts),
                            predictionsets=default_predictionsets(),
                            conditioning=default_conditioning())
```
where:
- `ordering` refers to the way that the points are ordered, which can heavily
  impact approximation accuracy. The default is a canonical sorting in 1D and a
  random ordering in 2+ dimensions. Very heuristically, the "best" generic
  ordering is the most efficiently space-filling one. Random ordering is a
  pretty decent fast approximation to that. 
- `predictionsets` is used to specify whether you are "chunking" your
  approximation. This package supports doing block Vecchia approximations, where
  the fundamental unit isn't a univariate conditional distribution. At the
  moment due to the migration, I haven't re-implemented any blocked constructors
  yet, so the only implemented option is `SingletonPredictionSets()`, which is
  the default.
- `conditioning` refers to how conditioning sets are chosen given the ordering.
  The default is `KNNConditioning(k)`, where `k=5` in 1D and `k=10` in 2+D. You
  can also specify a `metric` from `Distances.jl` with `KNNConditioning(k,
  metric)`. If you're on a sphere and `pts` are polar coordinates, for example,
  you should use `KNNConditioning(k, Haversine(1.0))`.

**The ultimate goal of this design is to make adding new specialized
constructors simple and straightforward**, so that fancier and more specific
configuration options can be offered via extensions or implemented on your own
in custom code.  With time, I will provide more options that also demonstrates
how one might do that.


# Roadmap to 1.0:

- A careful investigation into memoization or similar approaches to speed up
  construction and negative log-likelihood evaluation.
- Some time for users to kick the tires on the new design.

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

