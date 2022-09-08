
include("shared.jl")
import EMVecchia2: em_estimate

function extract_data(j)
  pts_data = readdlm("./data/data.csv", ',')
  pts = [SVector{2,Float64}(x...) for x in zip(pts_data[:,1], pts_data[:,2])]
  _data = pts_data[:,3:end]
  (pts, _data[:,j])
end

function extract_saa(j)
  _saa  = deserialize("./data/saa.jls") # v1.7 serialization!
  _saa[:,:,j]
end

# Wrapping in a function just to be obviously sure that there is no global scope
# issue slowing things down.
#
# This function writes each estimate to a separate file so that a single error
# doesn't cost all the progress. At some point, each computation should check if
# that file already exists before re-running the job.
function estimate_loop(j)

  # Check if the file exists:
  stem = string("trial_", j, "_")
  em_vec_outname  = string("./data/tmp/", stem*"em_vecchia.csv")
  em_path_outname = string("./data/tmp/", stem*"em_fulloutput.jls")
  if isfile(em_vec_outname)
    println("Already found EM-Vecchia estimator tmp file, skipping...")
    return nothing
  end

  # load in data:
  ini   = vec(readdlm("./data/initparameters.csv", ','))
  # optimization loop:
  #foreach(1:size(_data,2)) do j
    println("Running trial $j/50...\n")
    # pull out individual pieces, which will hopefully trigger the garbage
    # collector:
    stem  = string("trial_", j, "_")
    saaj  = extract_saa(j)
    (pts, dataj) = extract_data(j)
    # Create file paths variables:
    em_vec_outname  = string("./data/tmp/", stem*"em_vecchia.csv")
    em_path_outname = string("./data/tmp/", stem*"em_fulloutput.jls")
    #ex_vec_mle_outname = string("./data/tmp/", stem*"exact_vec_mle.csv")

    # Create the config:
    cfg = maximinconfig(kernel_nonugget, pts, dataj, 1, 10)

    # Compute the EM estimator if you don't have it already:
    println("Computing EM-Vecchia estimator:")
    try
      el  = @elapsed est = em_estimate(cfg, saaj, ini; 
                                       max_em_iter=20,
                                       ipopt_max_iter=30,
                                       ipopt_print_level=5,
                                       ipopt_box_l=[1e-3, 1e-3, 0.4, 0.0],
                                       return_trace=true)
      # append the runtime to the file:
      open("./data/times_jl.csv", "a") do io
        redirect_stdout(io) do
          println("$j,$el")
        end
      end
      # write output:
      writedlm(em_vec_outname, est.path[end], ',')
      # serialize the whole path:
      serialize(em_path_outname, est)
    catch er
      println("optimization failed with error $er, writing NaNs to files...")
      el = NaN
      nan_v = fill(NaN, 4)
      open("./data/times_jl.csv", "a") do io
        redirect_stdout(io) do
          println("$j,$el")
        end
      end
      # write output:
      writedlm(em_vec_outname, nan_v, ',')
      # serialize the whole path:
      serialize(em_path_outname, nan_v)
    end
    
    #=
    # Compute the MLE if you don't have it already:
    ex_mle_outname = string("./data/tmp/", stem*"exact_mle.csv")
    if isfile(ex_mle_outname)
      println("Already found exact MLE tmp file, skipping...")
    else
      println("\nComputing exact MLE:")
      nthr    = Threads.nthreads()
      BLAS.set_num_threads(nthr)
      pts     = reduce(vcat, cfg.pts)
      dat     = vec(reduce(vcat, cfg.data))
      kerneln = (x, y, p) -> begin
        out = cfg.kernel(x, y, p)
        ifelse(x == y, out+p[end], out)
      end
      ex_mle = EMVecchia2.exact_mle_efish(pts, dat, kerneln, ini, saaj)
      if !iszero(ex_mle.status)
        @warn "The exact MLE computation didn't return a success code, please inspect results."
      end
      writedlm(ex_mle_outname, ex_mle.minimizer, ',')
      (nthr > 1) && BLAS.set_num_threads(1)
    end
    =#
  #end
  nothing
end

# TODO (cg 2022/07/11 17:22): I should have made this a function that handled
# one case that I called multiple times. Oops.
function concatenate_tmp_files()
  j = 1
  em_estimates = Vector{Vector{Float64}}()
  #ex_estimates = Vector{Vector{Float64}}()
  while true
    stem  = string("trial_", j, "_")
    em_vec_outname = string("./data/tmp/", stem*"em_vecchia.csv")
    #ex_mle_outname = string("./data/tmp/", stem*"exact_mle.csv")
    files = (em_vec_outname,)# ex_mle_outname)
    if !all(isfile, files)
      println("Missing at least one file for trial $j, stopping concatenation.")
      break
    end
    # add to collection:
    push!(em_estimates, vec(readdlm(em_vec_outname, ',')))
    #push!(ex_estimates, vec(readdlm(ex_mle_outname, ',')))
    # now delete:
    #map(f->run(`rm -f $f`), files)
    j += 1
  end
  writedlm("./data/estimates_em.csv",       reduce(hcat, em_estimates), ',')
  #writedlm("./data/estimates_mle.csv",      reduce(hcat, ex_estimates), ',')
end


# Just to make sure that this long loop runs and there aren't longer-term GC
# issues or something, this is unfortunately going to be in a bash loop that
# will thus pre-compile every.single.time. So thankfully I did the benchmarks
# for timing already...
if !isinteractive()
  estimate_loop(parse(Int64, ARGS[1]))
  if parse(Int64, ARGS[1]) == 50
    concatenate_tmp_files()
  end
end

