
include("shared.jl")

# Just to be extra sure.
BLAS.set_num_threads(1)

# if you want to use KNITRO (and you HAVE it!), uncomment this and also mind the
# comment on line 49 below.
#include("knitro_substitute.jl")

function extract_data(j)
  pts_data = readdlm("./data/data.csv", ',')
  pts = [SVector{2,Float64}(x...) for x in zip(pts_data[:,1], pts_data[:,2])]
  _data = pts_data[:,3:end]
  (pts, _data[:,j])
end

function extract_saa(j)
  _saa  = deserialize("./data/saa.jls") # v1.7 serialization!
  booltosgn.(_saa[:,:,j])
end

# Wrapping in a function just to be obviously sure that there is no global scope
# issue slowing things down.
#
# This function writes each estimate to a separate file so that a single error
# doesn't cost all the progress. At some point, each computation should check if
# that file already exists before re-running the job.
function estimate_index(j, sleeptime=45)
  # Create file paths variables:
  stem            = string("trial_", j, "_")
  em_vec_outname  = string("./data/tmp/", stem*"em_vecchia.csv")
  em_path_outname = string("./data/tmp/", stem*"em_fulloutput.jls")
  # Check if the file exists and exit if so:
  if isfile(em_vec_outname)
    println("Already found EM-Vecchia estimator tmp file, skipping...")
    return nothing
  end
  # Otherwise, we continue:
  println("Running trial $j/50...\n")
  # Load in data:
  ini   = vec(readdlm("./data/initparameters.csv", ','))
  saaj  = extract_saa(j)
  (pts, dataj) = extract_data(j)
  # Create the config:
  cfg = maximinconfig(kernel_nonugget, pts, dataj, 1, 10)
  # Try the estimation:
  try
    # If you're using KNITRO, just set kw=() and un-comment the line that
    # defines optimizer=_knitro[...] below. If you're using the free
    # trust region optimizer I put into Vecchia.jl, no need to touch anything.
    kw = (:box_lower=>[1e-3, 1e-3, 0.4, 0.0],) 
    el = @elapsed est = em_estimate(cfg, saaj, ini; 
                                    #optimizer=_knitro_optimize_box,
                                    optimizer_kwargs=kw,
                                    warn_optimizer=false, 
                                    warn_notation=false)
    println("Concluded with status $(est.status).")
    # append the runtime to the file:
    open("./data/times_jl.csv", "a") do io
      redirect_stdout(io) do
        println("$j,$el")
      end
    end
    # write output:
    writedlm(em_vec_outname, est.path[end], ',')
    # serialize the whole path:
    serialize(em_path_outname, est.path)
  catch er
    println("optimization failed with error $er, writing NaNs to files...")
    el = NaN
    nan_v = fill(NaN, 4)
    open("./data/times_jl.csv", "a") do io
      redirect_stdout(io) do
        println("$j,$el")
      end
    end
    # write output and serialize the whole path:
    writedlm(em_vec_outname, nan_v, ',')
    serialize(em_path_outname, nan_v)
  end
  println("Sleeping for $sleeptime seconds...")
  sleep(sleeptime)
  nothing
end

function estimate_loop(starting_index, sleeptime=60)
  for j in starting_index:50
    estimate_index(j, sleeptime)
    GC.gc()
  end
  nothing
end

function concatenate_tmp_files()
  j = 1
  em_estimates = Vector{Vector{Float64}}()
  while true
    em_vec_outname = string("./data/tmp/", "trial_", j, "_", "em_vecchia.csv")
    if !isfile(em_vec_outname)
      println("Missing at least one file for trial $j, stopping concatenation.")
      break
    end
    # add to collection:
    push!(em_estimates, vec(readdlm(em_vec_outname, ',')))
    j += 1
  end
  writedlm("./data/estimates_em.csv", reduce(hcat, em_estimates), ',')
end


# Just to make sure that this long loop runs and there aren't longer-term GC
# issues or something, this is unfortunately going to be in a bash loop that
# will thus pre-compile every.single.time. So thankfully I did the benchmarks
# for timing already...
if !isinteractive()
  if isfile("./data/estimates_em.csv")
    println("EM estimate file already exists, exiting this script.") 
    exit(0)
  end
  estimate_loop(parse(Int64, ARGS[1]))
  concatenate_tmp_files()
end

