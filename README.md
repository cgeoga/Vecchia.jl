
# Vecchia.jl

**Note:** this package is undergoing an extensive rewrite on the `main` branch.
This README is at the time of writing up-to-date on the interface, but until
`v0.11` is released, please be mindful that this README is subject to big
changes and should not be what you use for documentation. I apologize for the
inconvenience and hope to tag the release soon.

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
- It implements a special EM algorithm-based refinement method for process
  models with noise.
- Because of how great the Julia optimization ecosystem is, you can hook into
  *much* more powerful optimizers than is possible in, for example, R.

## Simple usage demonstration

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
```
That easy! Enjoy your linear-cost (approximate) MLEs.

**See the example files for heavily commented demonstrations.**

# Additional functionality

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

## Sparse precision matrix and ("reverse") Cholesky factors

While it is not necessary for evaluating the negative log-likelihood, in some
settings it can be useful to use this approximation to produce a sparse
approximation to the model-implied precision matrix. For this purpose,
`rchol(appx, params)` gives a sparse Upper triangular matrix `U` (the "reverse"
Cholesky factor) such that your precision is approximated with `U*U'`.  **Note
that these objects correspond to permuted data, though, not the ordering in
which you provided the data**.
```julia
using SparseArrays 

# Note that the direct output of Vecchia.rchol is an internal object with just
# a few methods. But this sparse conversion will give you a good old SparseMatrixCSC.
U = UpperTriangular(sparse(rchol(appx, sample_p)))
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

# Roadmap to 1.0:

- More thoughtful interfaces for:
    - How internal permutations are handled. It would probably be best to
      abstract that away from the user, but it would also be nice for the sparse
      `rchol` construction to give back an `UpperTriangular` so that fast
      backsolves didn't require any additional factorization or anything.
    - Mean functions.
    - Predictions and conditional simulation. Right now, there is a
      `PredictionConfig` that is implemented, but it needs much more testing and
      design TLC.
- A careful investigation into memoization or similar approaches to speed up
  construction and negative log-likelihood evaluation.

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

