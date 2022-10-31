
function checkthreads()
  nthr = Threads.nthreads()
  if nthr > 1 && BLAS.get_num_threads() > 1
    @warn "It looks like you started Julia with multiple threads but are also using \
    multiple BLAS threads. The Julia multithreading isn't composable with BLAS \
    multithreading, so you should probably choose one or the other. If you want to \
    turn OFF the threading in this function, you could use the internal Vecchia.nll_floops \
    with the kwarg execmode=SequentialEx()." maxlog=1
  end
end

# A hacky function to return an empty Int64[] for the first conditioning set.
@inline cond_ixs(j, r) = j == 1 ? Int64[] : collect(max(1,j-r):max(1,j-1))

# number of elements in the lower triangle of an n x n matrix.
ltrisz(n) = div(n*(n+1), 2)

# Update kernel matrix buffer, exploiting redundancy for symmetric case. Not
# necessarily faster unless the kernel is very expensive to evaluate, but not
# slower in any case in my experimentation.
#
# Note that this does _NOT_ use threads, since I am already assuming that the
# nll function itself will be using threads, and in my benchmarking putting
# threaded constructors here slows things down a bit and increases allocations.
function updatebuf!(buf, pts1, pts2, kfun::F, params; skipltri=false) where{F}
  (F <: MemoizedKernel) && (@assert hash(params) === kfun.phash "params for memoized kernel don't agree with provided params. This shouldn't happen and is a bug.")
  if pts1 == pts2 && skipltri
    for k in eachindex(pts2)
      ptk = pts2[k]
      @inbounds buf[k,k] = kfun(ptk, ptk, params)
      @inbounds for j in 1:(k-1)
        buf[j,k] = kfun(pts1[j], ptk, params)
      end
    end
  elseif pts1 == pts2 && !skipltri
    for k in eachindex(pts2)
      ptk = pts2[k]
      @inbounds buf[k,k] = kfun(ptk, ptk, params)
      @inbounds for j in 1:(k-1)
        buf[j,k] = kfun(pts1[j], ptk, params)
        buf[k,j] = kfun(pts1[j], ptk, params)
      end
    end
  else
    @inbounds for k in eachindex(pts2), j in eachindex(pts1) 
      buf[j,k] = kfun(pts1[j], pts2[k], params)
    end
  end
  nothing
end

# This function works pretty differently: now we assume that the points have all
# been catted together and that the kernel function takes entirely scalar
# inputs. With this formatting, we can then actually use the SIMD tools of
# LoopVectorization.jl and get some serious speedup.
#
# Very grateful to Chris Elrod (@elrod on discourse, @chriselrod on Github) for
# the help in making this work.
@generated function updatebuf_avx!(buf, ::Val{D}, pts1, pts2, 
                                   kfun, params; skipltri=false) where{D}
  quote
    if pts1 == pts2 && skipltri
      for _k in 0:div(length(pts2)-1,$D)
        #@turbo for _j in 0:_k # @turbo
        @inbounds for _j in 0:_k 
          val = kfun($([:(pts1[_j*$D+$d]) for d in 1:D]...),
                     $([:(pts2[_k*$D+$d]) for d in 1:D]...),
                     params)
          buf[_j+1,_k+1] = val
        end
      end
    elseif pts1 == pts2 && !skipltri
      for _k in 0:div(length(pts2)-1,$D)
        #@turbo for _j in 0:_k # @turbo
        @inbounds for _j in 0:_k 
          val = kfun($([:(pts1[_j*$D+$d]) for d in 1:D]...),
                     $([:(pts2[_k*$D+$d]) for d in 1:D]...),
                      params)
          buf[_j+1,_k+1] = val
          buf[_k+1,_j+1] = val
        end
      end
    else
      #@turbo for  _k in 0:div(length(pts2)-1,$D),  _j in 0:div(length(pts1)-1,$D) # @turbo
      @inbounds for  _k in 0:div(length(pts2)-1,$D),  _j in 0:div(length(pts1)-1,$D) 
        val = kfun($([:(pts1[_j*$D+$d]) for d in 1:D]...),
                   $([:(pts2[_k*$D+$d]) for d in 1:D]...),
                    params)
        buf[_j+1,_k+1] = val
      end
    end
    nothing
  end
end

# TODO (cg 2022/04/21 16:16): This is totally not good.
function vec_of_vecs_to_matrows(vv)
  Matrix(reduce(hcat, vv)')
end

function prepare_v_buf!(buf, v, idxv)
  _ix = 1
  for ixs in idxv
    for ix in ixs
      @inbounds view(buf, _ix, :) .= view(v, ix, :)
      _ix += 1
    end
  end
  view(buf, 1:(_ix-1), :)
end

function updateptsbuf!(ptbuf, ptvv, idxs)
  ix = 1
  for idx in idxs
    for pt in ptvv[idx]
      @inbounds ptbuf[ix] = pt
      ix += 1
    end
  end
  view(ptbuf, 1:(ix-1))
end

function updatedatbuf!(datbuf, datvm, idxs)
  _start = 1
  stop   = 0
  for ix in idxs
    dat_ix = datvm[ix]
    sz_ix  = size(dat_ix, 1)
    stop   = _start+sz_ix-1
    view(datbuf, _start:stop, :) .= dat_ix
    _start += sz_ix
  end
  view(datbuf, 1:stop, :)
end

# Not a clever function at all,
function rchol_nnz(U::RCholesky{T}) where{T}
  # diagonal elements:
  out = sum(U.idxs) do ix
    n = length(ix)
    div(n*(n+1), 2)
  end
  # off-diagonal elements:
  out += sum(enumerate(U.condix)) do (j,ix_c)
    isempty(ix_c) && return 0
    tmp = 0
    len = length(U.idxs[j])
    for ix in ix_c
      @inbounds tmp += len*length(U.idxs[ix]) 
    end
    tmp
  end
  out
end

function debug_exactnll(cfg, params, nugget=false)
  pts = reduce(vcat, cfg.pts)
  dat = reduce(vcat, cfg.data)
  buf = [cfg.kernel(x,y,params) for x in pts, y in pts]
  if nugget
    buf += params[end]*I
  end
  S   = Symmetric(buf)
  Sf  = cholesky!(S)
  (logdet(Sf), sum(z->z^2, Sf.U'\dat))
end

generic_nll(R::Diagonal, data)  = 0.5*(logdet(R) + dot(data, R\data))

function generic_nll(R::UniformScaling, data)  
  n  = size(data, 1)
  m  = size(data, 2)
  ld = n*log(R.λ)
  qf = sum(t->(t^2)/R.λ, data)
  (m*ld + qf)/2
end

function gpmaxlik_optimize(obj, init; kwargs...)
  kwargsd = Dict(kwargs)
  objgh = p -> begin
    res = DiffResults.HessianResult(p)
    ForwardDiff.hessian!(res, obj, p)
    (DiffResults.value(res), DiffResults.gradient(res), DiffResults.hessian(res))
  end
  GPMaxlik.trustregion(obj, objgh, init; 
                       dmax=get(kwargsd, :dmax, Float64(length(init))),
                       dcut=get(kwargsd, :dcut, 1e-10),
                       kwargsd...)
end

function vecchia_estimate(cfg, init; optimizer=ipopt_optimize, optimizer_kwargs...)
  optimizer(p->Vecchia.nll(cfg, p), init; optimizer_kwargs...)
end

function exact_estimate_nugget(cfg, init; optimizer=ipopt_optimize, optimizer_kwargs...)
  pts = reduce(vcat, cfg.pts)
  dat = reduce(vcat, cfg.data)
  # TODO (cg 2022/09/08 13:06): fix this in GPMaxlik.
  @assert isone(size(dat, 2)) "GPMaxlik.gnll_forwarddiff does not presently work for multiple realizations."
  vdat = vec(dat)
  nugkernel = (x,y,p) -> cfg.kernel(x,y,p) + Float64(x==y)*p[end]
  obj  = p -> GPMaxlik.gnll_forwarddiff(p, pts, vdat, nugkernel)
  optimizer(obj, init; optimizer_kwargs...)
end

function exact_estimate(cfg, init; optimizer=ipopt_optimize, optimizer_kwargs...)
  pts = reduce(vcat, cfg.pts)
  dat = reduce(vcat, cfg.data)
  # TODO (cg 2022/09/08 13:06): fix this in GPMaxlik.
  @assert isone(size(dat, 2)) "GPMaxlik.gnll_forwarddiff does not presently work for multiple realizations."
  vdat = vec(dat)
  obj  = p -> GPMaxlik.gnll_forwarddiff(p, pts, vdat, cfg.kernel)
  optimizer(obj, init; optimizer_kwargs...)
end

# for simple debugging and testing.
function exact_nll(V::VecchiaConfig{H,D,F}, params::Vector{T}) where{H,D,T,F}
  pts = reduce(vcat, V.pts)
  dat = reduce(vcat, V.data)
  buf = Array{T}(undef, length(pts), length(pts))
  (ld, qf) = negloglik(V.kernel, params, pts, dat, buf)
  0.5*(ld + qf)
end

function chunk_indices(vv)
  szs    = [size(vj, 1) for vj in vv]
  starts = vcat(1, cumsum(szs).+1)[1:(end-1)]
  stops  = cumsum(szs)
  [x[1]:x[2] for x in zip(starts, stops)]
end

function augmented_em_cfg(V::VecchiaConfig{H,D,F}, z0, presolved_saa) where{H,D,F}
  chunksix = chunk_indices(V.pts)
  new_data = map(chunksix) do ixj
    hcat(z0[ixj,:], presolved_saa[ixj,:])
  end
  Vecchia.VecchiaConfig{H,D,F}(V.chunksize, V.blockrank, V.kernel, 
                               new_data, V.pts, V.condix)
end

