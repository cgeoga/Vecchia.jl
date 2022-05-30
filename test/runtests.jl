
include("../example/example_setup.jl")

using Test, Vecchia

# Create a vecc object that uses enough block-conditioning points that the
# likelihood evaluation is exact.
const vecc_exact = Vecchia.kdtreeconfig(sim, pts, 64, 10000, kfn)

# Test 1: nll gives the exact likelihood for a vecchia config where the
# conditioning set is every prior point.
vecchia_nll  = nll(vecc_exact, ones(2))
debug_nll    = Vecchia.exact_nll(vecc_exact, ones(2))
@test isapprox(vecchia_nll, debug_nll)

# Test 2: precisionmatrix gives the exact precision when it should....modulo
# some numerical noise, which is a little bit higher than I would expect.
pmat = Vecchia.precisionmatrix(vecc_exact, ones(2))
ptsv = reduce(vcat, vecc_exact.pts)
cmat = [kfn(x,y,ones(2)) for x in ptsv, y in ptsv]
@test maximum(abs, inv(Matrix(pmat))-cmat) < 1e-5

# Test 3: the nll agrees with the one obtained with the Vecchia precision.
datv = reduce(vcat, vecc_exact.data)
pnll = Vecchia.nll_precision(pmat, datv)
@test isapprox(pnll, debug_nll)

# Test 4: the scalarized nll agrees with the non-scalarized one.
scalarized_nll    = Vecchia.nll(vecc_s, ones(2))
nonscalarized_nll = Vecchia.nll(vecc, ones(2))
@test isapprox(scalarized_nll, nonscalarized_nll)

# Test 5: the nll with multiple data sources agrees with the sum of two
# single-data nlls.
const new_data  = range(0.0, 1.0, length=length(sim))
const joint_cfg = Vecchia.kdtreeconfig(hcat(sim, new_data), pts, 64, 3, kfn)
const new_cfg   = Vecchia.kdtreeconfig(new_data, pts, 64, 3, kfn)
@test isapprox(Vecchia.nll(joint_cfg, ones(2)), 
               Vecchia.nll(vecc, ones(2)) + Vecchia.nll(new_cfg, ones(2)))

# Test 6: make sure that with multiple data the precision matrices are the same.
const pmat_1 = Vecchia.precisionmatrix(vecc, ones(2))
const pmat_2 = Vecchia.precisionmatrix(joint_cfg, ones(2))
@test isapprox(pmat_1, pmat_2)

# Test 7: confirm that the rchol-based nll is equal to the standard nll.
rchol_nll = Vecchia.nll_rchol(vecc, ones(2))
@test isapprox(rchol_nll, Vecchia.nll(vecc, ones(2)))

