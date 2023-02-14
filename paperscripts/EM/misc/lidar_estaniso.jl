
using StandaloneIpopt, GPMaxlik

include("shared.jl")

sgnsqrt(x) = sqrt(abs(x))*sign(x)

# the same parameterization as usual, but just with a geometric anisotropy.
function kernel_nonugget_anisotropic(x, y, p)
  (sg2, rho, tmp_rho, nu, nug2) = p
  _x = @SVector [x[1], x[2]*exp(tmp_rho)]
  _y = @SVector [y[1], y[2]*exp(tmp_rho)]
  _p = @SVector [sg2, rho, nu, nug2]
  kernel_nonugget(_x, _y, _p)
end

function _ipopt_optimize(fn, x; kwargs...)
  wrapfn = StandaloneIpopt.WrappedObjective(fn)
  ipopt_optimize(wrapfn, x; kwargs...)
end

function estimate_lidar_exact(n=5_000)
  M    = readdlm("./data/lidar.csv", ',')[(end-n):end,:] # note the subsetting
  pts  = [SVector{2,Float64}(x...) for x in zip(M[:,1], M[:,2])]
  dat  = M[:,3]
  nugkern = Vecchia.NuggetKernel(kernel_nonugget_anisotropic)
  exact_obj = p -> GPMaxlik.gnll_forwarddiff(p, pts, dat, nugkern)
  ipopt_optimize(exact_obj, [0.600, 0.072, 4.362, 0.981, 1e-8], 
                 box_lower=[1e-5, 1e-5, -10.0, 0.25, 1e-10],
                 hessian_approximation="limited-memory",
                 max_iter=500, tol=1e-5)
end

