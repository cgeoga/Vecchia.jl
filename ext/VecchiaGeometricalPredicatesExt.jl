
module VecchiaGeometricalPredicatesExt

  using Vecchia, GeometricalPredicates
  using Vecchia.StaticArraysCore

  function _sortperm(x, x_sorted)
    indices = Dict(zip(x, eachindex(x)))
    [indices[x] for x in x_sorted]
  end

  # This function was lifted from 
  #
  # github.com/adolgert/BijectiveHilbert.jl/blob/main/src/sort.jl
  #
  # and is based on PR #41, written by myself (@cgeoga) and Andrew Dolgert
  # (@adolgert).  BijectiveHilbert.jl is distributed under the MIT license:
  #
  # Copyright (c) 2020 Andrew Dolgert <adolgert@uw.edu>
  #
  # Permission is hereby granted, free of charge, to any person obtaining a copy
  # of this software and associated documentation files (the "Software"), to deal
  # in the Software without restriction, including without limitation the rights
  # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  # copies of the Software, and to permit persons to whom the Software is
  # furnished to do so, subject to the following conditions:
  #
  # The above copyright notice and this permission notice shall be included in all
  # copies or substantial portions of the Software.
  #
  # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  # SOFTWARE.
  function transform_to_12d(xv::Vector{SVector{D,Float64}}) where{D}
    extreme_axes = [extrema(x->x[j], xv) for j in 1:D]
    divisor = maximum(x->x[2]-x[1], extreme_axes)*(1 + 10*eps()) 
    minx    = SVector{D,Float64}([x[1] for x in extreme_axes])
    [SVector{D,Float64}(ntuple(j->(x[j] - minx[j])/divisor + 1, D)) for x in xv]
  end

  sv_to_internal(pts::Vector{SVector{2,Float64}}) = [Point2D(x[1], x[2])       for x in pts]
  sv_to_internal(pts::Vector{SVector{3,Float64}}) = [Point3D(x[1], x[2], x[3]) for x in pts]

  function Vecchia._hilbert_permutation(pts)
    error("The `GeometricalPredicates.jl` routines require points in 2 or 3D. Please structure your points as either SVector{2,Float64} or SVector{3,Float64}.")
  end

  function Vecchia._hilbert_permutation(pts::Vector{SVector{D,Float64}}) where{D}
    in(D, (2,3)) || error("`GeometricalPredicates.jl` only offers this routine for points in 2D and 3D.")
    _pts     = transform_to_12d(pts)
    pts_gp   = sv_to_internal(_pts)
    pts_gp_s = sv_to_internal(_pts)
    hilbertsort!(pts_gp_s)
    _sortperm(pts_gp, pts_gp_s)
  end

end

