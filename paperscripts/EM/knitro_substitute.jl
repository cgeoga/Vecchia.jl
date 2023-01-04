
# add this with ]add https://git.sr.ht/~cgeoga/StandaloneKNITRO.jl, and then go
# to the https://github.com/JuMP-dev/KNITRO.jl page for instructions to get
# KNITRO.jl itself working.
using StandaloneKNITRO

# A simple struct that will return a NaN if it fails to evaluate. My dumb
# optimizer won't handle this in a useful way, but a serious package like KNITRO
# will. So this is useful to have.
struct WrappedObjective{F} <: Function
  fn::F
end

# Here is the one method that returns the NaN if there is an error.
function (f::WrappedObjective{F})(x) where{F}
  try
    return f.fn(x)
  catch
    return NaN
  end
end

# And the lazy function to provide to Vecchia.em_estimate.
function _knitro_optimize_box(fn, ini; kwargs...)
  knitro_optimize(WrappedObjective(fn), ini, 
                  box_lower=[1e-3, 1e-3, 0.4, 0.0],
                  param_file="sqp_loose.opt")
end

