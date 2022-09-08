
using Printf

_fmt1(x) = ifelse(x < zero(x), @sprintf("%.1f", x), @sprintf(" %.1f", x))
_fmt2(x) = ifelse(x < zero(x), @sprintf("%.2f", x), @sprintf(" %.2f", x))
_fmt3(x) = ifelse(x < zero(x), @sprintf("%.3f", x), @sprintf(" %.3f", x))
_fmt4(x) = ifelse(x < zero(x), @sprintf("%.4f", x), @sprintf(" %.4f", x))
_fmt5(x) = ifelse(x < zero(x), @sprintf("%.5f", x), @sprintf(" %.5f", x))
const _fmtv = (_fmt1,_fmt2,_fmt3,_fmt4,_fmt5)

function texmatrix(filname, M::Matrix{Float64}; digits=3, pad=true, dollarsigns=true,
                   sidelabels=nothing, bolds=Vector{Tuple{Int64, Int64}}())
  @assert 1 <= digits <= 5 "Digits must be between 1 and 5."
  fn  = _fmtv[digits]
  # first pass: convert to string, make bold indices bold, etc.
  out = fn.(M)
  if !isempty(bolds)
    for ix in bolds
      (j,k) = ix
      val   = out[j,k]
      if M[j,k] < 0.0
        out[j,k] = string("-\\bm{", _fmtv[digits](abs(M[j,k])), "}")
      else
        out[j,k] = string("\\bm{", val, "}")
      end
    end
  end
  # next pass: add dollar signs.
  (s1, s2) = size(M)
  if dollarsigns
    for j in 1:s1
      for k in 1:s2
        out[j,k] = "\$"*out[j,k]*"\$"
      end
    end
  end
  for j in 1:(s1-1)
    for k in 1:(s2-1)
      out[j,k] *= " &"
    end
    out[j,s2] *= " \\\\"
  end
  for k in 1:(s2-1)
    out[s1,k] *= " &"
  end
  if pad
    for k in 1:s2
      sizes = [length(split(x,".")[1]) for x in out[:,k]]
      for j in 1:s1
        out[j,k] = lpad(out[j,k], length(out[j,k])+maximum(sizes)-sizes[j])
      end
    end
  end
  if !isnothing(sidelabels)
    out = hcat(sidelabels.*"&", out)
  end
  writedlm(filname, out)
  nothing
end
