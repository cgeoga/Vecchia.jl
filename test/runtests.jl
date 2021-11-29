
include("../example/example_setup.jl")

using Test, Vecchia

# Create a vecc object that uses enough block-conditioning points that the
# likelihood evaluation is exact.
const vecc_exact = Vecchia.kdtreeconfig(sim, pts, 64, 10000, kfn)

# Test 1: nll gives the exact likelihood for a vecchia config where the
# conditioning set is every prior point.
vecchia_nll  = nll(vecc_exact, ones(2))
debug_nll    = Vecchia.negloglik(Val(0), kfn, ones(2), pts, sim, zeros(sz, sz))
@test isapprox(vecchia_nll, debug_nll)

# Test 2: precisionmatrix gives the exact precision when it should....modulo
# some numerical noise, which is a little bit higher than I would expect.
pmat = Vecchia.precisionmatrix(vecc_exact, ones(2))
ptsv = reduce(vcat, vecc_exact.pts)
cmat = [kfn(x,y,ones(2)) for x in ptsv, y in ptsv]
@test maximum(abs, inv(Matrix(pmat))-cmat) < 1e-5

# Test 3: the nll agrees with the one obtained with the Vecchia precision.
datv = reduce(vcat, vecc_exact.data)
pnll = Vecchia.negloglik_precision(pmat, zeros(length(sim)), datv)
@test isapprox(pnll, debug_nll)

# Test 4: the scalarized nll agrees with the non-scalarized one.
scalarized_nll    = Vecchia.nll(vecc_s, ones(2))
nonscalarized_nll = Vecchia.nll(vecc, ones(2))
@test isapprox(scalarized_nll, nonscalarized_nll)

