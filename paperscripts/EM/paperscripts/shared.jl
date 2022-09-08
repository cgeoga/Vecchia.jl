
using LinearAlgebra, StableRNGs, StaticArrays, BesselK, NearestNeighbors, Serialization
using Vecchia, EMVecchia2

if true #!isinteractive()
  nthr = Threads.nthreads()
  if nthr == 1
    println("Running with just one thread...") 
  else
    BLAS.set_num_threads(1)
    println("Running with $nthr threads, setting BLAS threads to one...") 
  end
end

function kernel_nonugget(x, y, p)
  (sg2, rho, nu, nug2) = p
  scaledist  = norm(x-y)/rho
  iszero(scaledist) && return sg2
  normalizer = sg2/((2^(nu-1))*BesselK.gamma(nu))
  normalizer*(scaledist^nu)*BesselK.adbesselk(nu, scaledist)
end

function prepare_dat()
  M   = readdlm("./data/data.csv", ',')
  pts = [SVector{2,Float64}(x...) for x in zip(M[:,1], M[:,2])]
  dat = M[:,3:end]
  s_perm = vec(Int64.(readdlm("./data/maximin_permutation_R_m10.csv", ',')))
  (pts[s_perm], dat[s_perm,:])
end

function brutal_kd_maximin(points_ordered, dat_ordered, m)
  @warn "This assumes that the points input is already maximin ordered."
  condix = Vector{Vector{Int64}}()
  push!(condix, Int64[])
  for j in 2:length(points_ordered)
    ptj   = points_ordered[j]
    tre_j = NearestNeighbors.KDTree(points_ordered[1:(j-1)])
    idxs  = NearestNeighbors.knn(tre_j, ptj, min(j-1,m))[1]
    push!(condix, sort(idxs))
  end
  (map(x->[x], points_ordered), map(x->[x;;], dat_ordered), condix)
end

function maximinconfig(kernel, pts, data, chunksize, blockrank)
  (_pts, _dat, condix) = brutal_kd_maximin(pts, data, blockrank)
  Vecchia.VecchiaConfig(chunksize, blockrank, kernel, _dat, _pts, condix)
end

