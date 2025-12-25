
using Test, LinearAlgebra, StaticArrays, StableRNGs, Vecchia

function exact_nll(cfg::Vecchia.SingletonVecchiaApproximation, p)
  S   = [cfg.kernel(x, y, p) for x in cfg.pts, y in cfg.pts]
  Vecchia.generic_dense_nll(S, cfg.data)
end

function singleton_to_exact(c::Vecchia.SingletonVecchiaApproximation)
  sp = Vecchia.SingletonPredictionSets()
  (pts_ch, data_ch) = Vecchia.chunk_format_points_and_data(c.pts, c.data, sp)
  Vecchia.ChunkedVecchiaApproximation(c.meanfun, c.kernel, data_ch, 
                                      pts_ch, c.condix, c.perm)
end

kernel(x, y, p) = p[1]*exp(-norm(x-y)/p[2])

# quick testing values:
rng    = StableRNG(123)
test_p = [0.1, 0.1]
pts    = rand(rng, SVector{2,Float64}, 100)
sim1   = randn(rng, length(pts))
sim2   = randn(rng, length(pts))

appxe  = VecchiaApproximation(pts, kernel, sim1; 
                              ordering=RandomOrdering(StableRNG(12)),
                              conditioning=KNNConditioning(1000))

appx1  = VecchiaApproximation(pts, kernel, sim1; 
                              ordering=RandomOrdering(StableRNG(12)),
                              conditioning=KNNConditioning(10))

appx1c = singleton_to_exact(appx1)

appx2  = VecchiaApproximation(pts, kernel, sim2; 
                              ordering=RandomOrdering(StableRNG(12)),
                              conditioning=KNNConditioning(10))

appx12 = VecchiaApproximation(pts, kernel, hcat(sim1, sim2); 
                              ordering=RandomOrdering(StableRNG(12)),
                              conditioning=KNNConditioning(10))


@testset "nll" begin
  @test isapprox(appxe(ones(2)), exact_nll(appxe, ones(2)))
end

@testset "chunked versus singleton nll" begin
  @test isapprox(appx1(ones(2)), appx1c(ones(2)))
end

@testset "multiple data nll" begin
  @test isapprox(appx1(ones(2)) + appx2(ones(2)), appx12(ones(2)))
end

@testset "rchol tiles" begin
  U = Vecchia._rchol(appx1c, ones(2))
  U_tiles = Vecchia._rchol(appx1c, ones(2), use_tiles=true)
  @test U.diagonals  == U_tiles.diagonals
  @test U.odiagonals == U_tiles.odiagonals
end

@testset "chunked versus singleton rchol" begin
  Us = rchol(appx1,  ones(2)).U
  Uc = rchol(appx1c, ones(2)).U
  @test Us ≈ Uc
end

@testset "rchol solve and mul" begin
  pts1d = rand(rng, SVector{1,Float64}, 100)
  M     = [kernel(x, y, (1.0, 0.01)) for x in pts1d, y in pts1d]
  appx  = VecchiaApproximation(pts1d, kernel; conditioning=KNNConditioning(1))
  pre   = rchol(appx, [1.0, 0.01])
  v     = collect(1.0:length(pts1d))
  @test maximum(abs, pre*(pre'*v) - M\v) < 1e-10
  @test maximum(abs, pre'\(pre\v) - M*v) < 1e-10
end

