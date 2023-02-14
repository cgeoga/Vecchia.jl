
using StandaloneIpopt

include("shared.jl")

using Vecchia.GPMaxlik

BLAS.set_num_threads(1)

# another optimizer option that is free and very robust: Ipopt's BFGS.
function _ipopt_estimate(obj, ini; kwargs...)
  low = [-30, -30, -30, 1e-8, -30, -30, 1/4, log(1e-8), log(1e-8), log(1e-8)]
  ipopt_optimize(obj, ini; wrap=true, box_lower=low, tol=1e-5,
                 hessian_approximation="limited-memory")
end

function param_interpolate(x, params)
  # some constants for compile time magic:
  knots = (0.2, 0.75, 1.1)
  scale = 5.0
  # the actual code:
  dists = (exp(-abs(x-knots[1])*scale), 
           exp(-abs(x-knots[2])*scale), 
           exp(-abs(x-knots[3])*scale))
  (dists[1]*params[1] + dists[2]*params[2] + dists[3]*params[3])/sum(dists)
end

# custom additive error:
struct AltitudeGrowingNugget
  pts::Vector{SVector{2,Float64}} # to make the matrix
end

function (agn::AltitudeGrowingNugget)(pt1, pt2, params)
  pt1 != pt2 && return zero(eltype(params)) # check if we're on the diagonal
  param_interpolate(pt1[1], exp.((params[8], params[9], params[10])))
end

function Vecchia.error_covariance(agn::AltitudeGrowingNugget, params)
  Diagonal([agn(x, x, params) for x in agn.pts])
end

function Vecchia.error_precision(agn::AltitudeGrowingNugget, params)
  Diagonal([inv(agn(x, x, params)) for x in agn.pts])
end

# Not the fastest way, but in the interest of self-documentation:
function Vecchia.error_qform(agn::AltitudeGrowingNugget, params, y, yTy)
  Rinv = Diagonal([inv(agn(x, x, params)) for x in agn.pts])
  dot(y, Rinv, y)
end

# Not the fastest way, but in the interest of self-documentation:
function Vecchia.error_logdet(agn::AltitudeGrowingNugget, params)
  R = Diagonal([agn(x, x, params) for x in agn.pts])
  logdet(R)
end

Vecchia.error_isinvertible(agn::AltitudeGrowingNugget, params) = true

# Not the fastest way, but in the interest of self-documentation:
function Vecchia.error_nll(agn::AltitudeGrowingNugget, params, y)
  R = Diagonal([agn(x, x, params) for x in agn.pts])
  0.5*(logdet(R) + dot(y, R\y))
end

# This kernel extends on the one used to compare with SGV in a few ways:
# 1) it has a fully generic geometric anisotropy, parameterized in terms of the
#    Cholesky factor of the inverse norm matrix.
# 2) it has a nonstationary scale.
# 3) it will use a nonstationary nugget (see the AltitudeGrowingNugget code above)
function kernel_ns_nonugget(x, y, params)
  _one    = one(eltype(params))
  _zero   = zero(eltype(params))
  scale_x = param_interpolate(x[1], (exp(params[1]), exp(params[2]), exp(params[3])))
  scale_y = param_interpolate(y[1], (exp(params[1]), exp(params[2]), exp(params[3])))
  A_inv_L = @SMatrix [params[4] _zero ; params[5] exp(params[6])]
  tmp_p   = @SVector [_one, _one, params[7], _zero]
  scale_x*scale_y*kernel_nonugget(A_inv_L*x, A_inv_L*y, tmp_p)
end

function prepare_lidar()
  # read in the objects:
  M    = readdlm("./data/lidar_highalt.csv", ',')
  pts  = [SVector{2,Float64}(x...) for x in zip(M[:,1], M[:,2])]
  dat  = M[:,3]
  # create the point and data chunks:
  chunks    = Iterators.partition(1:length(dat), 20)
  pts_dat_s = [(pts[chunk], hcat(dat[chunk])) for chunk in chunks]
  condix_init = [Int64[], [1], [1,2]]
  condix_rest = map(4:length(pts_dat_s)) do j
    isodd(j) ? [j-4, j-2, j-1] : [j-3, j-2, j-1]
  end
  condix = vcat(condix_init, condix_rest)
  Vecchia.VecchiaConfig(20, 3, kernel_ns_nonugget, 
                        getindex.(pts_dat_s, 2),
                        getindex.(pts_dat_s, 1),
                        condix)
end

function sqp_bfgs(obj, ini; kwargs...)
  ddxs = Vecchia.AutoFwdBFGS(obj, length(ini))
  Vecchia.sqptr_optimize(obj, ini; fgh=ddxs, kwargs...)
end

function estimate_lidar()
  if isfile("./data/lidar_highalt_estimates.jls") 
    println("Found estimates file, skipping re-computation...")
    return
  end
  cfg = prepare_lidar()
  err = AltitudeGrowingNugget(reduce(vcat, cfg.pts))
  saa = rand(StableRNG(123), (-1.0, 1.0), sum(length, cfg.pts), 72)
  low = [-30, -30, -30, 1e-8, -30, -30, 1/4, log(1e-8), log(1e-8), log(1e-8)]
  kw  = (:box_lower=>low, :delta_min=>1e-8)
  ini = [log(0.6), log(0.6), log(0.6), 10.0, 0.0, 7.0, 1.0, 
         log(0.001), log(0.01), log(0.1)]
  est = em_estimate(cfg, saa, ini;
                    errormodel=err,
                    optimizer=sqp_bfgs,
                    optimizer_kwargs=kw,
                    warn_optimizer=false,
                    warn_notation=false)
  serialize("./data/lidar_highalt_estimates.jls", est)
end


function estimate_lidar_indepblocks()
  M    = readdlm("./data/lidar_highalt.csv", ',')
  pts  = [SVector{2,Float64}(x...) for x in zip(M[:,1], M[:,2])]
  dat  = M[:,3]
  pts_chunk = collect.(Iterators.partition(pts, 80))
  dat_chunk = hcat.(collect.(Iterators.partition(dat, 80)))
  condix    = fill(Int64[], length(pts_chunk))
  cfg = Vecchia.VecchiaConfig(80, 0, kernel_ns_nonugget, dat_chunk, pts_chunk, condix)
  low = [-30, -30, -30, 1e-8, -30, -30, 1/4, log(1e-8), log(1e-8), log(1e-8)]
  ini = [log(0.6), log(0.6), log(0.6), 10.0, 0.0, 7.0, 1.0, 
         log(0.001), log(0.01), log(0.1)]
  est = Vecchia.vecchia_estimate(cfg, ini, box_lower=low, 
                                 optimizer=_ipopt_estimate, delta_min=1e-8)
  serialize("./data/lidar_highalt_estimates_iblocks.jls", est)
end

# Careful because the buf you need to give to this function is about 10 GiB in
# size! Would suggest having at least 16 GiB of RAM to re-run this.
#
# independent blocks: -27316.703
# naive Vecchia:      -31897.905
# EM+Vecchia:         -31915.803
function careful_exact_likelihood(buf, params)
  cfg = prepare_lidar()
  pts = reduce(vcat, cfg.pts)
  dat = vec(reduce(vcat, cfg.data))
  krn = Vecchia.ErrorKernel(kernel_ns_nonugget, AltitudeGrowingNugget(pts))
  Vecchia.updatebuf!(buf, pts, pts, krn, params; skipltri=true)
  cho = cholesky!(Symmetric(buf))
  0.5*(logdet(cho) + sum(abs2, cho.U'\dat))
end

if !isinteractive()
  estimate_lidar()
  estiamte_lidar_indepblocks()
end

