
# Vecchia.jl

A terse Julia implementation of Vecchia approximations to the Gaussian
likelihood, which work very well in many settings and run in *linear complexity
with data size* (assuming O(1) sized conditioning sets). As of now this is only
implemented for mean-zero processes. Implemented with chunked observations
instead of singleton observations as in Stein/Chi/Welty 2004 JRSSB [1].
Reasonably optimized for minimal allocations so that multithreading (via the
excellent FLoops ecosystem) really works well while still being AD-compatible.
**To my knowledge, this is the only program that offers true Hessians of Vecchia
likelihoods.** 

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

Here is a very quick demo:

```julia
using LinearAlgebra, StaticArrays, Vecchia

# VERY IMPORTANT FOR MULTITHREADING, since this is many small BLAS/LAPACK calls:
BLAS.set_num_threads(1)

# Covariance function, in this case Matern(v=3/2):
kfn(x,y,p) = p[1]*exp(-norm(x-y)/p[2])*(1.0+norm(x-y)/p[2])

# Locations for fake measurements, in this case 2048 of them, and fake data 
# (data NOT from the correction distribution, this is just a maximally simple demo):
const pts = [SVector{2, Float64}(randn(2)) for _ in 1:2048]
const dat = randn(length(pts))

# Create the VecchiaConfig: 
# If you have multiple i.i.d. samples, pass in a matrix where each column is a sample.
const chunksize = 10 
const num_conditioning_chunks = 3
const cfg = Vecchia.kdtreeconfig(dat, pts, chunksize, num_conditioning_chunks, kfn)

# Estimate like so, with the default optimizer being Ipopt and using autodiff
# for all gradients and Hessians. TRUE Hessians are used in this estimation by
# default, not expected Fisher matrices.
const mle = vecchia_estimate(cfg, some_init)
```

**See the example files for a heavily commented demonstration.**

The code is organized with modularity and user-specific applications in mind, so
the primary way to interact with the approximation is to create a
`VecchiaConfig` object that specifies the chunks and conditioning sets for each
chunk. The only provided one is a very basic option that orders the points with
a KD-tree with a specified terminal leaf size (so that each leaf is a chunk),
re-orders those chunks based on the leaf centers, and then picks conditioning
sets based on the user-provided size. 

If you want something fancier, for example the maximin ordering of Guinness 2018
technometrics with the NN-based conditioning sets, which was recently proved to
have some nice properties (Schafer et al 2021 SISC), that shouldn't be very hard
to implement after skimming the existing constructor to see what the struct
fields in `VecchiaConfig` mean and stuff. I really made an effort to design this
in such a way that you can specialize how you want but then just enjoy the
painfully optimized generic log-likelihood, precision matrix, and sparse
(reverse)-Cholesky functionality without having to rebuild from scratch every
time.

# Advanced Usage

## Estimation with a nugget/measurement error

As mentioned above, measurement error can really hurt the accuracy of these
approximations. If your model is effectively given by `data(x) = good_gp(x) +
iid_noise(x)`, where `good_gp` is something that screens well that you actually
want to use Vecchia on and `iid_noise` has VARIANCE `eta^2`, then you can
estimate all parameters, including `eta^2`, like so:
```julia
# importantly, your kernel function here should NOT include the nugget:
cfg = Vecchia.kdtreeconfig(data, pts, chunksize, num_cond_chunks, kernel_no_nug)

# draw some iid Rademacher vectors that are used in a stochastic trace
# calculation in the estimation routine:
saa = rand((-1.0, 1.0), n_data, n_saa)

# Estimate. This object currently returns a lot of diagnostic information and at
# some point you should expect me to clean this up a bit so you don't have to do
# ugly reference to get your estimator.
em_mle = em_estimate(cfg, [init_good_gp_params, init_eta^2], saa)[3][end]
```
This is of course too terse of a discussion here, but see the example file for
more information and see also the [paper](https://arxiv.org/abs/2208.06877) for
a lot more information. **If you use this method, please cite this paper**.

## Sparse precision matrix and ("reverse") Cholesky factors

While it will almost always be faster to just evaluated the likelihood with
`Vecchia.nll(cfg, params)`, you *can* actually obtain the precision matrix `S`
such that `Vecchia.nll(cfg, params) == -logdet(S) + dot(data, S, data)`. You can
*also* obtain the upper triangular matrix `U` such that `S = U*U'`. **Note that
these objects correspond to permuted data, though, not the ordering in which you
provided the data**. Here is an example usage:
```julia
# using the VecchiaConfig called vecc that was created above:
S = Vecchia.precisionmatrix(vecc, sample_p)

# Note that this is NOT given in the form of a sparse matrix, it is a custom
# struct with just two methods: U'*x and logdet(U), which is all you need to
# evaluate the likelihood. 
U = Vecchia.rchol(vecc, sample_p)

# If you want the sparse matrix (don't forget to wrap as UpperTriangular!):
U_SparseMatrixCSC = UpperTriangular(sparse(U))

# Here is how I'd recommend getting your data in the correct permutation out:
data_perm = reduce(vcat, vecc.data)
```
You'll get a warning the first time you call `rchol` or `precisionmatrix`
re-iterating the issue about permutations. If you want to avoid that, you can
pass in the kwarg `issue_warning=false`.

## SIMD via `LoopVectorization.jl`

In some cases, for example with a sufficiently simple covariance function and a
sufficiently advanced CPU, using SIMD can really speed up likelihood
evaluations.  To take advantage of this, we need to change the format of the
stored locations.  This is entirely abstracted away from the user and not your
problem, but you will need to write a new version of your covariance function
that takes all of the location coordinates as scalars. For example:
```julia
# Note that this is equal to kfn above, but instead of expecting x and y as
# AbstractVector types (or anything where norm(x-y) works), now you pass in 
# each component. 
function kfn_scalar(x1, x2, y1, y2, p)
  nrm = sqrt((x1-y1)^2 + (x2-y2)^2)
  p[1]*exp(-nrm/p[2])*(1+nrm/p[2])
end

# We still need to keep track of the dimension information somehow, so for now
# the fix is simply to construct the "scalarized" version from the normal version:
const vecc_s   = Vecchia.scalarize(vecc, kfn_scalar)

# With just that little bit of extra code, now this nll function call will use SIMD
# to assemble the covariance matrices. On my computer with an intel i5-11600K, which
# has the AVX-512 instruction set, this is a factor of two faster than the nll with
# the standard VecchiaConfig struct. 
Vecchia.nll(vecc_s, sample_p)
```
Much gratitude to Chris Elrod for helping me understand how to correctly use
`@generated` functions to make the assembly functions efficient for arbitrary
coordinate dimensions. And for creating `LoopVectorization.jl`, of course.

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
