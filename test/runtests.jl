
using Test, LinearAlgebra, StaticArrays, StableRNGs, Vecchia, SparseArrays

BLAS.set_num_threads(1)

# TODO (cg 2022/12/23 15:18): 
# 1) Any EM tests
# 2) Any sqp/tr tests

function exact_nll(cfg, p)
  pts = reduce(vcat, cfg.pts)
  dat = reduce(vcat, cfg.data)
  Vecchia.GPMaxlik.gnll_forwarddiff(p, pts, dat, cfg.kernel)
end

kernel(x, y, p) = p[1]*exp(-norm(x-y)/p[2])

# quick testing values:
rng    = StableRNG(123)
test_p = [0.1, 0.1]
pts    = rand(SVector{2,Float64}, 200)
saa    = rand(rng, (-1.0, 1.0), length(pts), 72)
sim    = randn(rng, length(pts))

# Create a vecc object that uses enough block-conditioning points that the
# likelihood evaluation is exact.
vecc       = Vecchia.kdtreeconfig(sim, pts, 5, 3, kernel)
vecc_exact = Vecchia.kdtreeconfig(sim, pts, 5, 10000, kernel)

# Test 1: nll gives the exact likelihood for a vecchia config where the
# conditioning set is every prior point.
@testset "nll" begin
vecchia_nll  = nll(vecc_exact, ones(3))
debug_nll    = exact_nll(vecc_exact, ones(3))
@test isapprox(vecchia_nll, debug_nll)
end

# Test 5: the nll with multiple data sources agrees with the sum of two
# single-data nlls.
@testset "multiple data nll" begin
new_data  = range(0.0, 1.0, length=length(sim))
joint_cfg = Vecchia.kdtreeconfig(hcat(sim, new_data), pts, 5, 3, kernel)
new_cfg   = Vecchia.kdtreeconfig(new_data, pts, 5, 3, kernel)
@test isapprox(Vecchia.nll(joint_cfg, ones(3)), 
               Vecchia.nll(vecc, ones(3)) + Vecchia.nll(new_cfg, ones(3)))
end

# Test 7: confirm that the rchol-based nll is equal to the standard nll.
@testset "rchol nll" begin
rchol_nll = Vecchia.nll_rchol(vecc, ones(3), issue_warning=false)
@test isapprox(rchol_nll, Vecchia.nll(vecc, ones(3)))
end

# Test 8: confirm that the rchol built with tiles and without tiles gives the
# same result:
@testset "rchol tiles" begin
U = Vecchia.rchol(vecc, ones(3), issue_warning=false)
U_tiles = Vecchia.rchol(vecc, ones(3), use_tiles=true, issue_warning=false)
@test U.diagonals == U_tiles.diagonals
@test U.odiagonals == U_tiles.odiagonals
end

# Test 9: make sure the Vecchia-based conditional distributions agree with the
# exact ones when you condition on every prior point.
@testset "conditional distributions" begin
  pts  = rand(SVector{2,Float64}, 30)
  ppts = rand(SVector{2,Float64}, 10)
  data = cholesky([kernel(x, y, (1.0, 0.1)) for x in pts, y in pts]).L*randn(length(pts))

  cfg  = Vecchia.nosortknnconfig(data, pts, 50, kernel)
  (test_cond_mean, test_cond_var) = Vecchia.dense_posterior(cfg, [1.0, 0.1], ppts, ncondition=50)

  S1   = [kernel(x, y, (1.0, 0.1)) for x in pts,  y in pts]
  S12  = [kernel(x, y, (1.0, 0.1)) for x in pts,  y in ppts]
  S2   = [kernel(x, y, (1.0, 0.1)) for x in ppts, y in ppts]

  # test 1: conditional mean.
  @test test_cond_mean ≈ S12'*(S1\data)    
  @test test_cond_var  ≈ S2 - S12'*(S1\S12)
end

