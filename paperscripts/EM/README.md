
# A Scalable Method to Exploit Screening in Gaussian Process Models with Noise

This folder is a (slightly disorganized) collection of the exact code used to
generate the results for this paper. I developed it as a module/package, so
please refer to `./paperscripts/` to see the things that actually generated
paper results.

Note, though, that if you just want the methods, the actual package `Vecchia.jl`
now has the EM method in it, and you can pass in your own optimizer. See the
docstrings for `Vecchia.em_estimate` for more details. **PLEASE do not use this
code unless you are explicitly trying to literally reproduce things from the
paper. The code in `Vecchia.jl` will be faster and cleaner in all
circumstances.**

CG (Sep 2022)

# Update (Oct 31, 2022)

The implementation in this code (the module `EMVecchia2`) is now pretty diverged
from the implementation in `Vecchia.jl`, which now is even more efficient and
uses O(1) memory w.r.t. data size. **There is absolutely no reason to use the
code here unless you're trying to reproduce the exact results in the paper.**

