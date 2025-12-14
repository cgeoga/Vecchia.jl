
struct VecchiaConfig{H,D,F}
  kernel::F
  data::Vector{Matrix{H}}
  pts::Vector{Vector{SVector{D, Float64}}}
  condix::Vector{Vector{Int64}} 
end

function Base.display(V::VecchiaConfig)
  println("Vecchia configuration with:")
  println("  - chunksize:  $(chunksize(V))")
  println("  - block rank: $(blockrank(V))")
  println("  - data size:  $(sum(x->size(x,1), V.data))")
  println("  - nsamples:   $(size(V.data[1], 2))")
end

