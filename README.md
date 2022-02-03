
# Vecchia.jl

A terse Julia implementation of Vecchia approximations to the Gaussian
likelihood, which work very well in many settings and run in *linear complexity
with data size* (assuming O(1) sized conditioning sets). As of now this is only
implemented for mean-zero processes. Implemented with chunked observations
instead of singleton observations as in Stein/Chi/Welty 2004 JRSSB [1].
Reasonably optimized for minimal allocations so that multithreading (via the
excellent FLoops ecosystem) really works well while still being AD-compatible. 

The accuracy of Vecchia approximations depends on the *screening effect* [2],
which can perhaps be considered as a substantially weakened Markovian-like
property. But the screening effect even for covariance functions that do exhibit
screening can be significantly weakened by measurement noise (corresponding to a
"nugget" in the spatial statistics terminology), for example, and so I highly
recommend investigating whether or not you have reason to expect that your
specific model exhibits screening to an acceptable degree. In some cases, like
with measurement noise, there are several workarounds and some are pretty easy.
But for some covariance functions screening really doesn't hold and so this
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
const chunksize = 64
const num_conditioning_chunks = 3
const vecc = Vecchia.kdtreeconfig(dat, pts, chunksize, num_conditioning_chunks, kfn)

# Now you can evaluate the likelihood:
const sample_p = ones(2)
Vecchia.nll(vecc, sample_p)

# More interestingly, you can use AD very easily:
using ForwardDiff
obj(p) = Vecchia.nll(vecc, p)
ForwardDiff.hessian(obj, sample_p)

# You can also make the induced sparse precision matrix from the model
# (BUT, keep in mind there is also a permutation of the data here. So this is
# the approximated precision matrix for your data re-ordered as 
# reduce(vcat, vecc.data)) :
const sparse_Omega = Vecchia.precisionmatrix(vecc, sample_p)
```

There is also an "expert mode" for those interested in additionally utilizing
SIMD where possible using `LoopVectorization.jl`. To do this, we need to change
the format of the stored locations. This is entirely abstracted away from the
user and not your problem, but you will need to write a new version of your
covariance function that takes all of the location coordinates as scalars. For
example:
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

See the example files for a complete demonstration of using `ForwardDiff`'s
caching and the Ipopt optimizer for some seriously powerful and efficient
maximum likelihood estimation.

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
fields in `VecchiaConfig` mean and stuff.

# Citation

If you use this software in your work, please cite the package itself:
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
