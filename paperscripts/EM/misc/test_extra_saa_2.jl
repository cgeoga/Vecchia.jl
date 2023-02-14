
# A simple test just to see how much extra SAA vectors affects things. This is a
# sort of reduced/copied version of the file fit.jl.

include("shared.jl")

# Just to be extra sure.
BLAS.set_num_threads(1)

function extract_data(j)
  pts_data = readdlm("./data/data.csv", ',')
  pts   = [SVector{2,Float64}(x...) for x in zip(pts_data[:,1], pts_data[:,2])]
  _data = pts_data[:,3:end]
  (pts, _data[:,j])
end

function mle_withnugget(j)
  ini = vec(readdlm("./data/initparameters.csv", ','))
  (pts, dataj) = extract_data(j)
  cfg = maximinconfig(kernel_nonugget, pts, dataj, 1, 10)
  kw  = (:box_lower=>[1e-3, 1e-3, 0.4, 0.0],) 
  (cfg, Vecchia.vecchia_mle_withnugget(cfg, ini, Vecchia.sqptr_optimize; kw...))
end

function refine_index(cfg, saa, ini)
  kw = (:box_lower=>[1e-3, 1e-3, 0.4, 0.0],) 
  el = @elapsed res = Vecchia.em_refine(cfg, saa, ini; optimizer_kwargs=kw)
  (runtime=el, path=res.path)
end

function give_saa_vectors(n, sizes, rng=StableRNG(12345))
  saa_matrices = [rand(rng, (-1.0, 1.0), 15_000, sizes[1])]
  for j in 2:length(sizes)
    new_saa = rand(rng, (-1.0, 1.0), 15_000, sizes[j] - sizes[j-1])
    push!(saa_matrices, hcat(saa_matrices[end], new_saa))
  end
  saa_matrices
end

function sequential_saa_estimates(j, saa_matrices)
  # get the MLE with the nugget as the base of the refinement:
  (cfg, mlewnugget) = mle_withnugget(j)
  mle_ini = mlewnugget.minimizer
  # compute and return all the EM-refined estimates for each SAA matrix:
  Dict([size(saa_j, 2)=>refine_index(cfg, saa_j, mle_ini) for saa_j in saa_matrices])
end

const saa_matrices = give_saa_vectors(15_000, (5, 25, 50, 75, 100, 125))
const saa_trials   = [sequential_saa_estimates(j, saa_matrices) for j in 1:10]

serialize("./misc/saa_results_multipledata.jls", saa_trials)

