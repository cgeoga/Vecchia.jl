
include("../example/example_setup.jl")

using Test, Vecchia, SparseArrays

# Create a vecc object that uses enough block-conditioning points that the
# likelihood evaluation is exact.
const vecc       = Vecchia.kdtreeconfig(sim, pts, 5, 3, matern)
const vecc_exact = Vecchia.kdtreeconfig(sim, pts, 5, 10000, matern)
const vecc_s     = Vecchia.scalarize(vecc, matern_scalar)

# Test 1: nll gives the exact likelihood for a vecchia config where the
# conditioning set is every prior point.
println("testing nll...")
vecchia_nll  = nll(vecc_exact, ones(3))
debug_nll    = Vecchia.exact_nll(vecc_exact, ones(3))
@test isapprox(vecchia_nll, debug_nll)

# Test 2: precisionmatrix gives the exact precision when it should....modulo
# some numerical noise, which is a little bit higher than I would expect.
println("testing precision...")
pmat = Vecchia.precisionmatrix(vecc_exact, ones(3))
ptsv = reduce(vcat, vecc_exact.pts)
cmat = [matern(x,y,ones(3)) for x in ptsv, y in ptsv]
@test maximum(abs, inv(Matrix(pmat))-cmat) < 1e-5

# Test 3: the nll agrees with the one obtained with the Vecchia precision.
println("testing precision nll...")
datv = reduce(vcat, vecc_exact.data)
pnll = Vecchia.nll_precision(pmat, datv)
@test isapprox(pnll, debug_nll)

# Test 4: the scalarized nll agrees with the non-scalarized one.
println("Testing scalar nll...")
scalarized_nll    = Vecchia.nll(vecc_s, ones(3))
nonscalarized_nll = Vecchia.nll(vecc, ones(3))
@test isapprox(scalarized_nll, nonscalarized_nll)

# Test 5: the nll with multiple data sources agrees with the sum of two
# single-data nlls.
println("Testing multiple data nll...")
const new_data  = range(0.0, 1.0, length=length(sim))
const joint_cfg = Vecchia.kdtreeconfig(hcat(sim, new_data), pts, 5, 3, matern)
const new_cfg   = Vecchia.kdtreeconfig(new_data, pts, 5, 3, matern)
@test isapprox(Vecchia.nll(joint_cfg, ones(3)), 
               Vecchia.nll(vecc, ones(3)) + Vecchia.nll(new_cfg, ones(3)))

# Test 6: make sure that with multiple data the precision matrices are the same.
println("Testing multiple data precision...")
const pmat_1 = Vecchia.precisionmatrix(vecc, ones(3))
const pmat_2 = Vecchia.precisionmatrix(joint_cfg, ones(3))
@test isapprox(pmat_1, pmat_2)

# Test 7: confirm that the rchol-based nll is equal to the standard nll.
println("Testing rchol nll...")
rchol_nll = Vecchia.nll_rchol(vecc, ones(3))
@test isapprox(rchol_nll, Vecchia.nll(vecc, ones(3)))

# Test 8: the same, but with the sparse cholesky version.
println("Testing rchol and precision...")
const _U = sparse(Vecchia.rchol(vecc, ones(3)))
const _S = Symmetric(_U*_U')
@test isapprox(rchol_nll, Vecchia.nll_precision(_S, datv))

