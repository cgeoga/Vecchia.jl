
_square(x::Real) = x*x
_square(x::Complex) = real(x*conj(x))

_mean(x) = sum(x)/length(x)

function checkthreads()::Nothing
  nthr = Threads.nthreads()
  if nthr > 1 && BLAS.get_num_threads() > 1
    @warn THREAD_WARN maxlog=1
  end
  nothing
end

# Given data x_init that has been re-ordered to x_sorted, this gives a vector p
# such that x_init[p] == x_sorted.
function outer_to_inner_perm(x_init, x_sorted)
  lookup = Dict(zip(x_init, eachindex(x_init)))
  [lookup[x] for x in x_sorted]
end

# Given data x_init that has been re-ordered to x_sorted, this gives a vector p
# such that x_sorted[p] == x_init.
inner_to_outer_perm(x_init, x_sorted) = invperm(outer_to_inner_perm(x_init, x_sorted))

blockrank(cfg::VecchiaConfig) = maximum(length, cfg.condix)
chunksize(cfg::VecchiaConfig) = maximum(length, cfg.pts)

function check_singleton_sets(cfg::VecchiaConfig)
  if chunksize(cfg) > 1
    throw(error("This method is only implemented for singleton prediction sets. Please open an issue if there is missing functionality you need."))
  end
  nothing
end

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

function updatebuf_tiles!(buf, tiles, jv, kv)
  (ix1, ix2, s1) = (1, 1, 0)
  for j in jv
    ix2 = 1
    for k in kv
      tilejk   = tiles[j, k]
      (s1, s2) = size(tilejk)
      view(buf, ix1:(ix1+s1-1), ix2:(ix2+s2-1)) .= tilejk
      ix2 += s2
    end
    ix1 += s1
  end
  nothing
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
function _nnz(vchunks, condix)
  # diagonal elements:
  out = sum(vchunks) do piece
    n = length(piece)
    div(n*(n+1), 2)
  end
  # off-diagonal elements:
  out += sum(enumerate(condix)) do (j,ix_c)
    isempty(ix_c) && return 0
    tmp = 0
    len = length(vchunks[j])
    for ix in ix_c
      @inbounds tmp += len*length(vchunks[ix]) 
    end
    tmp
  end
  out
end

function generic_dense_simulate(pts, kernel, params)
  S  = [kernel(x, y, params) for x in pts, y in pts]
  Sf = cholesky!(S)
  Sf.L*randn(length(pts))
end

function generic_dense_nll(S, data)
  Sf = cholesky(Hermitian(S))
  (logdet(Sf) + sum(abs2, Sf.U'\data))/2
end

generic_nll(R::Diagonal, data)  = 0.5*(logdet(R) + dot(data, R\data))

function generic_nll(R::UniformScaling, data)  
  eta2  = R.λ
  (n,m) = size(data)
  (m*n*log(eta2) + sum(_square, data)/eta2)/2
end

function chunk_indices(vv)
  szs    = [size(vj, 1) for vj in vv]
  starts = vcat(1, cumsum(szs).+1)[1:(end-1)]
  stops  = cumsum(szs)
  [x[1]:x[2] for x in zip(starts, stops)]
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

