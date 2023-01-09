
_square(x::Real) = x*x

_mean(x) = sum(x)/length(x)

function checkthreads()
  nthr = Threads.nthreads()
  if nthr > 1 && BLAS.get_num_threads() > 1
    @warn "It looks like you started Julia with multiple threads but are also using multiple BLAS threads. The Julia multithreading isn't composable with BLAS multithreading, so please run BLAS.set_num_threads(1) before executing this function." maxlog=1
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
  (logdet(Sf), sum(_square, Sf.U'\dat))
end

generic_nll(R::Diagonal, data)  = 0.5*(logdet(R) + dot(data, R\data))

function generic_nll(R::UniformScaling, data)  
  n  = size(data, 1)
  m  = size(data, 2)
  ld = n*log(R.λ)
  qf = sum(_square, data)/R.λ
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

function vecchia_estimate(cfg, init; box_lower=fill(1e-5, length(init)), warn_box=true, 
                          optimizer=sqptr_optimize, optimizer_kwargs...)
  likelihood = WrappedLogLikelihood(cfg)
  if warn_box
    @info "You can turn off this warning with the kwarg warn_box=false." maxlog=1
    @warn "This function defaults to lower bounds on parameters of 1e-5 as a sensible default. But if that's not right for your problem, you can pass in a vector of lower (and upper) bounds with box_lower=[...] and box_upper=[...]." maxlog=1
  end
  optimizer(likelihood, init; box_lower=fill(1e-5, length(init)), optimizer_kwargs...)
end

function exact_estimate(cfg, init; add_nugget=false, optimizer=sqptr_optimize, 
                        box_lower=fill(1e-5, length(init)), warn_box=true, optimizer_kwargs...)
  if warn_box
    @info "You can turn off this warning with the kwarg warn_box=false." maxlog=1
    @warn "This function defaults to lower bounds on parameters of 1e-5 as a sensible default. But if that's not right for your problem, you can pass in a vector of lower (and upper) bounds with box_lower=[...] and box_upper=[...]." maxlog=1
  end
  pts  = reduce(vcat, cfg.pts)
  dat  = reduce(vcat, cfg.data)
  vdat = vec(dat)
  kernel = add_nugget ? NuggetKernel(cfg.kernel) : cfg.kernel
  obj  = p -> GPMaxlik.gnll_forwarddiff(p, pts, vdat, kernel)
  optimizer(obj, init; box_lower=box_lower, optimizer_kwargs...)
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

function globalidxs(datavv)
  (out, start) = (Vector{UnitRange{Int64}}(undef, length(datavv)), 1)
  for (j, datvj) in enumerate(datavv)
    len = size(datvj,1)
    out[j] = start:(start+len-1)
    start += len
  end
  out
end

@generated function allocate_cnll_bufs(::Val{N}, ::Val{D}, ::Val{Z}, 
                                       ndata, cpts_sz, pts_sz) where{N,D,Z}
  quote
    Base.Cartesian.@ntuple $N j->cnllbuf(Val(D), Val(Z), ndata, cpts_sz, pts_sz)
  end
end

@generated function allocate_crchol_bufs(::Val{N}, ::Val{D}, ::Val{Z}, 
                                         cpts_sz, pts_sz) where{N,D,Z}
  quote
    Base.Cartesian.@ntuple $N j->crcholbuf(Val(D), Val(Z), cpts_sz, pts_sz)
  end
end
