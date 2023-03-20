
_square(x::Real) = x*x

_mean(x) = sum(x)/length(x)

function checkthreads()::Nothing
  nthr = Threads.nthreads()
  if nthr > 1 && BLAS.get_num_threads() > 1
    @warn THREAD_WARN maxlog=1
  end
  nothing
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

generic_nll(R::Diagonal, data)  = 0.5*(logdet(R) + dot(data, R\data))

function generic_nll(R::UniformScaling, data)  
  eta2  = R.λ
  (n,m) = size(data)
  (m*n*log(eta2) + sum(_square, data)/eta2)/2
end

function vecchia_estimate(cfg, init; box_lower=fill(1e-5, length(init)), 
                          warn_box=true, optimizer=sqptr_optimize, 
                          optimizer_kwargs...)
  likelihood = WrappedLogLikelihood(cfg)
  if warn_box && all(==(1e-5), box_lower)
    notify_disable("warn_box=false")
    @warn BOUNDS_WARN maxlog=1
  end
  optimizer(likelihood, init; box_lower=fill(1e-5, length(init)), 
            optimizer_kwargs...)
end

function exact_estimate(cfg, init; errormodel=nothing, 
                        optimizer=sqptr_optimize, 
                        box_lower=fill(1e-5, length(init)), 
                        warn_box=true, optimizer_kwargs...)
  if warn_box
    notify_disable("warn_box=false")
    @warn BOUNDS_WARN maxlog=1
  end
  pts  = reduce(vcat, cfg.pts)
  dat  = reduce(vcat, cfg.data)
  n    = length(pts)
  vdat = vec(dat)
  kernel = isnothing(errormodel) ? cfg.kernel : ErrorKernel(cfg.kernel, ScaledIdentity(n))
  obj  = p -> GPMaxlik.gnll_forwarddiff(p, pts, vdat, kernel)
  optimizer(obj, init; box_lower=box_lower, optimizer_kwargs...)
end

function vecchia_estimate_nugget(cfg, init, optimizer, errormodel; 
                                 optimizer_kwargs...)
  nugkernel = ErrorKernel(cfg.kernel, errormodel) 
  nug_cfg   = Vecchia.VecchiaConfig(cfg.chunksize, cfg.blockrank,
                                    nugkernel, cfg.data, cfg.pts, cfg.condix)
  likelihood = WrappedLogLikelihood(nug_cfg)
  optimizer(likelihood, init; optimizer_kwargs...)
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

function allocate_cnll_bufs(N, ::Val{D}, ::Val{Z}, 
                            ndata, cpts_sz, pts_sz) where{D,Z}
  [cnllbuf(Val(D), Val(Z), ndata, cpts_sz, pts_sz) for _ in 1:N]
end

function split_nll_pieces(V::VecchiaConfig{H,D,F}, ::Val{Z}, m) where{H,D,F,Z}
  ndata   = size(first(V.data), 2)
  cpts_sz = V.chunksize*V.blockrank
  pts_sz  = V.chunksize
  chunks  = Iterators.partition(eachindex(V.pts), cld(length(V.pts), m))
  map(chunks) do chunk
    local_buf = cnllbuf(Val(D), Val(Z), ndata, cpts_sz, pts_sz)
    VecchiaLikelihoodPiece(V, local_buf, chunk)
  end
end

@generated function allocate_crchol_bufs(::Val{N}, ::Val{D}, ::Val{Z}, 
                                         cpts_sz, pts_sz) where{N,D,Z}
  quote
    Base.Cartesian.@ntuple $N j->crcholbuf(Val(D), Val(Z), cpts_sz, pts_sz)
  end
end

function pretty_print_number(x)
  if x < zero(x)
    @printf "-%06.3f " abs(x)
  else
    @printf " %06.3f " x
  end
  nothing
end

# A simple stand-in that pretty-prints a vector. I'm sure there is a smarter way
# to do this.
function pretty_print_vec(x, newline=false)
  print("x: [")
  pretty_print_number.(x)
  print("]")
  newline && println()
  nothing
end

