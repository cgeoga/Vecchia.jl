
using Test, LinearAlgebra, StaticArrays, StableRNGs, Vecchia, SparseArrays

# TODO (cg 2022/12/23 15:18): 
# 1) Any EM tests
# 2) Any sqp/tr tests

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
#const vecc_s     = Vecchia.scalarize(vecc, kernel_scalar)

# Test 1: nll gives the exact likelihood for a vecchia config where the
# conditioning set is every prior point.
println("testing nll...")
vecchia_nll  = nll(vecc_exact, ones(3))
debug_nll    = Vecchia.exact_nll(vecc_exact, ones(3))
@test isapprox(vecchia_nll, debug_nll)

# Test 5: the nll with multiple data sources agrees with the sum of two
# single-data nlls.
println("Testing multiple data nll...")
new_data  = range(0.0, 1.0, length=length(sim))
joint_cfg = Vecchia.kdtreeconfig(hcat(sim, new_data), pts, 5, 3, kernel)
new_cfg   = Vecchia.kdtreeconfig(new_data, pts, 5, 3, kernel)
@test isapprox(Vecchia.nll(joint_cfg, ones(3)), 
               Vecchia.nll(vecc, ones(3)) + Vecchia.nll(new_cfg, ones(3)))

# Test 7: confirm that the rchol-based nll is equal to the standard nll.
println("Testing rchol nll...")
rchol_nll = Vecchia.nll_rchol(vecc, ones(3); issue_warning=false)
@test isapprox(rchol_nll, Vecchia.nll(vecc, ones(3)))

