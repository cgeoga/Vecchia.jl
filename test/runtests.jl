
using Test, LinearAlgebra, StaticArrays, StableRNGs, Vecchia, SparseArrays

BLAS.set_num_threads(1)

function exact_nll(cfg, p)
  pts = reduce(vcat, cfg.pts)
  dat = reduce(vcat, cfg.data)
  S   = [cfg.kernel(x, y, p) for x in pts, y in pts]
  Vecchia.generic_dense_nll(S, dat)
end

kernel(x, y, p) = p[1]*exp(-norm(x-y)/p[2])

# quick testing values:
rng    = StableRNG(123)
test_p = [0.1, 0.1]
pts    = rand(rng, SVector{2,Float64}, 200)
sim    = randn(rng, length(pts))

# Create a vecc object that uses enough block-conditioning points that the
# likelihood evaluation is exact.
vecc2      = VecchiaApproximation(pts, kernel, sim; 
                                  ordering=NoPermutation(),
                                  conditioning=KNNConditioning(10))
vecc_exact = VecchiaApproximation(pts, kernel, sim; 
                                  ordering=NoPermutation(),
                                  conditioning=KNNConditioning(100000))

# Test 1: nll gives the exact likelihood for a vecchia config where the
# conditioning set is every prior point.
@testset "nll" begin
vecchia_nll  = Vecchia.nll(vecc_exact, ones(3))
debug_nll    = exact_nll(vecc_exact, ones(3))
@test isapprox(vecchia_nll, debug_nll)
end

#=
# Test 5: the nll with multiple data sources agrees with the sum of two
# single-data nlls.
@testset "multiple data nll" begin
new_data  = range(0.0, 1.0, length=length(sim))
joint_cfg = VecchiaApproximation(pts, kernel, hcat(sim, new_data); 
                                  ordering=NoPermutation(),
                                  conditioning=KNNConditioning(10))
new_cfg   = VecchiaApproximation(pts, kernel, new_data; 
                                  ordering=NoPermutation(),
                                  conditioning=KNNConditioning(10))
@test isapprox(Vecchia.nll(joint_cfg, ones(3)), 
               Vecchia.nll(vecc2, ones(3)) + Vecchia.nll(new_cfg, ones(3)))
end
=#

# Test 8: confirm that the rchol built with tiles and without tiles gives the
# same result:
@testset "rchol tiles" begin
U = Vecchia._rchol(vecc2, ones(3))
U_tiles = Vecchia._rchol(vecc2, ones(3), use_tiles=true)
@test U.diagonals  == U_tiles.diagonals
@test U.odiagonals == U_tiles.odiagonals
end

@testset "rchol solve and mul" begin
  pts = [SA[x] for x in sort(rand(rng, 500))]
  M   = [kernel(x, y, (1.0, 0.01)) for x in pts, y in pts]
  ppx = VecchiaApproximation(pts, kernel; 
                             conditioning=KNNConditioning(1))
  pre = rchol(ppx, [1.0, 0.01])
  U   = pre.U
  v   = collect(1.0:500.0)
  P   = I(length(pre.p))[pre.p,:]
  @test maximum(abs, inv(P'*M*P) - Matrix(U*U')) < 1e-7
  @test maximum(abs, pre*(pre'*v) - M\v) < 1e-7
  @test maximum(abs, pre'\(pre\v) - M*v) < 1e-10
end

