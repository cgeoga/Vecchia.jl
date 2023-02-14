
using GPMaxlik, ForwardDiff

include("shared.jl")

# Just to be extra sure.
BLAS.set_num_threads(5)

function extract_data(j)
  pts_data = readdlm("./data/data.csv", ',')
  pts   = [SVector{2,Float64}(x...) for x in zip(pts_data[:,1], pts_data[:,2])]
  _data = pts_data[:,3:end]
  (pts, _data[:,j])
end

const (pts, data) = extract_data(1)
const nugkern = Vecchia.NuggetKernel(kernel_nonugget)

_nll(p) = GPMaxlik.gnll_forwarddiff(p, pts, data, nugkern)

@time ForwardDiff.gradient(_nll, [5.0, 0.2, 1.0, 0.125]) # about 110 seconds on my box

