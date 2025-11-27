
# Vecchia.jl

A terse Julia implementation of Vecchia approximations to the Gaussian
likelihood, which work very well in many settings and run in *linear complexity
with data size* (assuming O(1) sized conditioning sets). As of now this is only
implemented for mean-zero processes. Implemented with chunked observations
instead of singleton observations as in Stein/Chi/Welty 2004 JRSSB [1].
Reasonably optimized for minimal allocations so that multithreading  really
works well while still being AD-compatible.  **To my knowledge, this is the only
program that offers true Hessians of Vecchia likelihoods.** 

The accuracy of Vecchia approximations depends on the *screening effect* [2],
which can perhaps be considered as a substantially weakened Markovian-like
property. But the screening effect even for covariance functions that do exhibit
screening can be significantly weakened by measurement noise (corresponding to a
"nugget" in the spatial statistics terminology), for example, and so I highly
recommend investigating whether or not you have reason to expect that your
specific model exhibits screening to an acceptable degree. In some cases, like
with measurement noise, there are several workarounds and some are pretty easy
(including one based on the EM algorithm that this package now offers).  But for
some covariance functions screening really doesn't hold and so this
approximation scheme may not perform well. This isn't something that the code
can enforce, so user discretion is required.

Here is a very quick demo. The only real step here is to create a
`VecchiaConfig` object, which specifies the point ordering/leaves/conditioning
sets/and so on. One convenience constructor for doing that is shown here:
```julia
using LinearAlgebra, StaticArrays, Vecchia

# VERY IMPORTANT FOR MULTITHREADING, since this is many small BLAS/LAPACK calls:
BLAS.set_num_threads(1)

# Covariance function, in this case Matern(v=3/2):
kfn(x,y,p) = p[1]*exp(-norm(x-y)/p[2])*(1.0+norm(x-y)/p[2])

# Locations for fake measurements, in this case 2048 of them, and fake data 
# (data NOT from the correction distribution, this is just a maximally simple demo):
pts = rand(SVector{2, Float64}, 2048)
dat = Vecchia.generic_dense_simulate(pts, kfn, (1.0, 0.1, 0.75))

# Built-in configuration option 1: random ordering and knn-based conditioning sets.
# If you have multiple i.i.d. samples, pass in a matrix where each column is a sample.
const cfg = knnconfig(dat, pts, 10, kfn)

# Built-in configuration option 2: a KD-tree based configuration on chunks
# (as opposed to singletons), as in Stein/Chi/Welty JRSSB 2004.
# If you have multiple i.i.d. samples, pass in a matrix where each column is a sample.
chunksize = 10 
num_conditioning_chunks = 3
const cfg_chunk = Vecchia.kdtreeconfig(dat, pts, chunksize, num_conditioning_chunks, kfn)
```
But note that you can directly construct the `VecchiaConfig` yourself pretty
easily. If you have a specific type of configuration you would like to achieve,
please feel free to open an issue and we can talk about how that is done, or
submit a PR. I would be very happy to merge in a few other generic constructors
here, because this chunked KD-tree one is not always the best choice.

For estimation, thanks to Julia's cool weakdep/extension framework, you have a
few options. The first is to use [JuMP](https://github.com/jump-dev/JuMP.jl).
Here is a demonstration of fitting your model with JuMP and
[Ipopt](https://github.com/jump-dev/Ipopt.jl):
```julia
using ForwardDiff, NLPModels # necessary to load the optimizer extension!
using UnoSolver # or NLPModelsIpopt, or NLPModelsKNITRO, etc

solver = NLPModelsSolver(uno; preset="filtersqp")
mle    = vecchia_estimate(cfg, some_init, solver)
```
Note that JuMP has bindings to many different optimizers, and this extension
only requires loading JuMP. Ipopt is, in my experience and option, among the
best and most battle-tested libre optimizers available. It is a great default
choice. If you have it, [KNITRO.jl](https://github.com/jump-dev/KNITRO.jl) is
another good choice to consider here---but it is not libre, and it is certainly
not gratis.

**See the example files for heavily commented demonstrations.**

The code is organized with modularity and user-specific applications in mind, so
the primary way to interact with the approximation is to create a
`VecchiaConfig` object that specifies the chunks and conditioning sets for each
chunk. The only provided one is a very basic option that orders the points with
a KD-tree with a specified terminal leaf size (so that each leaf is a chunk),
re-orders those chunks based on the leaf centers, and then picks conditioning
sets based on the user-provided size. 

# Advanced Usage

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

## Sparse precision matrix and ("reverse") Cholesky factors

While it will almost always be faster to just evaluated the likelihood with
`Vecchia.nll(cfg, params)`, you *can* actually obtain the precision matrix `S`
such that `Vecchia.nll(cfg, params) == -logdet(S) + dot(data, S, data)`. You can
*also* obtain the upper triangular matrix `U` such that `S = U*U'`. **Note that
these objects correspond to permuted data, though, not the ordering in which you
provided the data**. 

While this package originally offered both, the direct assembly of `U` is much
simpler and in order to streamline this code I have removed the option to
directly assemble `S` that used the different algorithm of Sun and Stein (2016). 

Here is an example usage:
```julia
# Note that this is NOT given in the form of a sparse matrix, it is a custom
# struct with just two methods: U'*x and logdet(U), which is all you need to
# evaluate the likelihood. 
U = Vecchia.rchol(vecc, sample_p)

# If you want the sparse matrix (don't forget to wrap as UpperTriangular!):
U_SparseMatrixCSC = UpperTriangular(sparse(U))

# If you want S back, for example:
S = U_SparseMatrixCSC*U_SparseMatrixCSC'

# Here is how I'd recommend getting your data in the correct permutation out:
data_perm = reduce(vcat, vecc.data)
```
You'll get a warning the first time you call `rchol` re-iterating the issue
about permutations. If you want to avoid that, you can pass in the kwarg
`issue_warning=false`.

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

# References

[1] https://rss.onlinelibrary.wiley.com/doi/abs/10.1046/j.1369-7412.2003.05512.x

[2] https://arxiv.org/pdf/1203.1801.pdf
