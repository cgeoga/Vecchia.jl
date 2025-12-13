
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
- It implements a special EM algorithm-based refinement method for process
  models with noise.
- Because of how great the Julia optimization ecosystem is, you can hook into
  *much* more powerful optimizers than is possible in, for example, R.

The fundamental object is a `Vecchia.VecchiaConfig`, which specifies your
conditioning sets for each sub-problem and also acts as a functor with methods
for the negative log-likelihood. Here is a pseudocode example specifying a
Vecchia approximation with a random ordering and knn conditioning
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
const cfg = knnconfig(data, pts, k, kernel;
                      randomize=false, # recommend true for lattice data
                      metric=Vecchia.Euclidean()) # use Haversine if data on a sphere!
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
U = UpperTriangular(sparse(Vecchia.rchol(cfg, sample_p)))
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
    - A `VecchiaConfig` constructor where conditioning set design is more
      modular, and possibly also amenable to extensions. Perhaps `HNSW` could
      move from a dep to an extension, and also a Hilbert curve extension could
      be supported for banded approximation.
    - Predictions and conditional simulation. Right now, there is a
      `PredictionConfig` that is implemented, but it needs much more testing and
      design TLC.
- A careful investigation into memoization or similar approaches to speed up

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

