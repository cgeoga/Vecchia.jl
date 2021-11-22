
# Vecchia.jl

A terse Julia implementation of Vecchia approximations to the Gaussian
likelihood, which work very well and run in *linear complexity with data size*
(assuming O(1) sized conditioning sets).  Implemented with chunked observations
instead of singleton observations as in Stein/Chi/Welty 2004 JRSSB. Reasonably
optimized for minimal allocations so that multithreading (via the excellent
FLoops ecosystem) really works well while still being AD-compatible. 

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
the approximated precision matrix for your data re-ordered as reduce(vcat,
vecc.data) :
const sparse_Omega = Vecchia.precisionmatrix(vecc, sample_p)
```

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

If you use this software in your work, please cite the package itself. I would
also be curious to see/hear about your application if you're willing to share
it or take the time to tell me about it.

