
using Vecchia
using Vecchia.LinearAlgebra
using Test, StaticArrays, StableRNGs
using BesselK, ForwardDiff, GPMaxlik

function exact_nll(cfg::Vecchia.SingletonVecchiaApproximation, p)
  S   = [cfg.kernel(x, y, p) for x in cfg.pts, y in cfg.pts]
  mu  = [cfg.meanfun(x, p) for x in cfg.pts]
  Vecchia.generic_dense_nll(S, cfg.data - mu)
end

function singleton_to_chunk(c::Vecchia.SingletonVecchiaApproximation)
  sp = Vecchia.SingletonPredictionSets()
  (pts_ch, data_ch) = Vecchia.chunk_format_points_and_data(c.pts, c.data, sp)
  Vecchia.ChunkedVecchiaApproximation(c.meanfun, c.kernel, data_ch, 
                                      pts_ch, c.condix, c.perm)
end

function dense_posterior(kernel, params, pts_have, pts_pred, data, meanfun)
  Σ_have  = [kernel(x, y, params) for x in pts_have, y in pts_have]
  Σ_cross = [kernel(x, y, params) for x in pts_pred, y in pts_have]
  Σ_pred  = [kernel(x, y, params) for x in pts_pred, y in pts_pred]
  cmean   = Σ_cross*(Σ_have\(data - meanfun.(pts_have)))
  cvar    = Σ_pred - Σ_cross*(Σ_have\Σ_cross')
  (cmean, cvar)
end

kernel(x, y, p) = p[1]*exp(-norm(x-y)/p[2])
meanfun(x, p)   = exp(x[1]*p[3])

# quick testing values:
rng    = StableRNG(123)
test_p = [0.1, 0.2, 0.3]
pts    = rand(rng, SVector{2,Float64}, 100)
sim1   = randn(rng, length(pts))
sim2   = randn(rng, length(pts))

appxe  = VecchiaApproximation(pts, kernel, sim1; 
                              meanfun=meanfun,
                              ordering=RandomOrdering(StableRNG(12)),
                              conditioning=KNNConditioning(1000))

appx1  = VecchiaApproximation(pts, kernel, sim1; 
                              meanfun=meanfun,
                              ordering=RandomOrdering(StableRNG(12)),
                              conditioning=KNNConditioning(10))

appx1c = singleton_to_chunk(appx1)

appx2  = VecchiaApproximation(pts, kernel, sim2; 
                              meanfun=meanfun,
                              ordering=RandomOrdering(StableRNG(12)),
                              conditioning=KNNConditioning(10))

appx12 = VecchiaApproximation(pts, kernel, hcat(sim1, sim2); 
                              meanfun=meanfun,
                              ordering=RandomOrdering(StableRNG(12)),
                              conditioning=KNNConditioning(10))


@testset "nll" begin
  @test isapprox(appxe(test_p), exact_nll(appxe, test_p))
end

@testset "chunked versus singleton nll" begin
  @test isapprox(appx1(test_p), appx1c(test_p))
end

@testset "multiple data nll" begin
  @test isapprox(appx1(test_p) + appx2(test_p), appx12(test_p))
end

@testset "rchol tiles" begin
  U = Vecchia._rchol(appx1c, test_p)
  U_tiles = Vecchia._rchol(appx1c, test_p, use_tiles=true)
  @test U.diagonals  == U_tiles.diagonals
  @test U.odiagonals == U_tiles.odiagonals
end

@testset "chunked versus singleton rchol" begin
  Us = rchol(appx1,  test_p).U
  Uc = rchol(appx1c, test_p).U
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

@testset "lazy_rchol mul" begin
  U   = rchol(appx1, [1.0, 0.01])
  lU  = lazy_rchol(appx1, [1.0, 0.01])
  v1  = collect(1.0:length(pts))
  v5  = Float64.([j+k for j in 1:length(pts), k in 1:5])
  @test maximum(abs, U*v1 - lU*v1) < 1e-12
  @test maximum(abs, U*v5 - lU*v5) < 1e-12
end

@testset "Manual expected Fisher" begin
  fish_pts = rand(rng, SVector{2,Float64}, 50)
  fish_dat = randn(rng, length(fish_pts), 3)
  fish_cfg = VecchiaApproximation(fish_pts, matern, fish_dat; 
                                  ordering=NoPermutation(),
                                  conditioning=KPastIndicesConditioning(50))
  ref_nll(p) = GPMaxlik.gnll_forwarddiff(p, fish_pts, fish_dat, matern; efish=true)
  ref_efish  = ForwardDiff.hessian(ref_nll, [1.5, 0.5, 1.25])

  (_nll, _grad, _fish) = Vecchia.nll_grad_fish(fish_cfg, [1.5, 0.5, 1.25])
  @test _nll ≈ fish_cfg([1.5, 0.5, 1.25])
  @test ForwardDiff.gradient(fish_cfg, [1.5, 0.5, 1.25]) ≈ _grad
  @test ref_efish.*3 ≈ _fish
end

@testset "Prediction" begin
  _pts = rand(StableRNG(1234), SVector{2,Float64}, 110)
  prms = [5.0, 0.5, 2.25]
  sim  = randn(length(_pts))

  (pts_train,  data_train)  = (_pts[1:100], sim[1:100])
  (pts_pred, data_pred)     = (_pts[101:end], sim[101:end])


  (cmean, cvar) = dense_posterior(matern, prms, pts_train, 
                                  pts_pred, data_train, x->0.0)


  dense_appx = VecchiaApproximation(pts_train, matern, data_train;
                                    conditioning=KNNConditioning(1000))

  predictions = predict(dense_appx, pts_pred, prms; conditioning=KNNConditioning(1000))


  @test cmean ≈ conditional_mean(predictions)
  @test cvar  ≈ full_conditional_covariance(predictions)
  @test diag(cvar) ≈ conditional_variances(predictions)
end

