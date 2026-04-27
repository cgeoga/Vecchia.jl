
n_data_samples(V::ChunkedVecchiaApproximation)   = size(first(V.data), 2)
n_data_samples(V::SingletonVecchiaApproximation) = size(V.data, 2)

_square(x::Real) = x*x
_square(x::Complex) = real(x*conj(x))

_mean(x) = sum(x)/length(x)

blockrank(cfg::VecchiaApproximation) = maximum(length, cfg.condix)
chunksize(cfg::ChunkedVecchiaApproximation) = maximum(length, cfg.pts)

function check_singleton_sets(cfg::VecchiaApproximation)
  if chunksize(cfg) > 1
    throw(error("This method is only implemented for singleton prediction sets. Please open an issue if there is missing functionality you need."))
  end
  nothing
end

# number of elements in the lower triangle of an n x n matrix.
ltrisz(n) = div(n*(n+1), 2)

# A simple algorithm for attempting to evenly distribute the work of likelihood
# evaluation to each core. This is an experimental optimization.
function even_work_chunks_greedy(kv::Vector{Int64}, nworkers)
  # Calculate work and get the indices sorted from largest to smallest work
  work = [k^3 for k in kv]
  sorted_idx = sortperm(work, rev=true)
  # Initialize trackers for each worker
  worker_loads = zeros(Int64, nworkers)
  assignments  = [Vector{Int64}() for _ in 1:nworkers]
  # Greedily assign the largest remaining job to the least loaded worker
  for idx in sorted_idx
    min_worker = argmin(worker_loads)
    push!(assignments[min_worker], idx)
    worker_loads[min_worker] += work[idx]
  end
  # Just for readability---will remove this after kicking the tires a bit.
  foreach(asj -> sort!(asj), assignments)
  assignments
end

function uniform_index_chunks(n, nworkers)
  collect(Iterators.partition(1:n, cld(n, Threads.nthreads())))
end

function updatebuf!(buf::AbstractMatrix, pts::AbstractVector{P},
                    kfun::F, params) where{P,F}
  @inbounds begin
  for k in eachindex(pts)
    for j in 1:k
      buf[j,k] = kfun(pts[j], pts[k], params)
      buf[k,j] = buf[j,k]
    end
  end
  end
  nothing
end

# Update kernel matrix buffer, exploiting redundancy for symmetric case. Not
# necessarily faster unless the kernel is very expensive to evaluate, but not
# slower in any case in my experimentation.
#
# Note that this does _NOT_ use threads, since I am already assuming that the
# nll function itself will be using threads, and in my benchmarking putting
# threaded constructors here slows things down a bit and increases allocations.
function updatebuf!(buf::AbstractMatrix, 
                    pts1::AbstractVector{P}, 
                    pts2::AbstractVector{P}, 
                    kfun::F, params; skipltri=false) where{P,F}
  @inbounds begin
  if pts1 == pts2 && skipltri
    for k in eachindex(pts2)
      ptk = pts2[k]
      buf[k,k] = kfun(ptk, ptk, params)
      for j in 1:(k-1)
        buf[j,k] = kfun(pts1[j], ptk, params)
      end
    end
  elseif pts1 == pts2 && !skipltri
    for k in eachindex(pts2)
      ptk = pts2[k]
      buf[k,k] = kfun(ptk, ptk, params)
      for j in 1:(k-1)
        buf[j,k] = kfun(pts1[j], ptk, params)
        buf[k,j] = buf[j,k] 
      end
    end
  else
    for k in eachindex(pts2), j in eachindex(pts1) 
      buf[j,k] = kfun(pts1[j], pts2[k], params)
    end
  end
  end
  nothing
end

function updatebuf!(buf::AbstractVector, ptsv::AbstractVector{P}, pt::P, 
                    kfun::F, params) where{P,F}
  @inbounds begin
  for j in eachindex(buf, ptsv)
    buf[j] = kfun(ptsv[j], pt, params)
  end
  end
  nothing
end

function updatebuf!(buf::AbstractVector, pt::P, ptsv::AbstractVector{P}, 
                    kfun::F, params) where{P,F}
  updatebuf!(buf, ptsv, pt, kfun, params)
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

function updateptsbuf!(ptbuf::Vector{T}, ptvv::Vector{Vector{T}}, idxs) where{T}
  ix = 1
  for idx in idxs
    for pt in ptvv[idx]
      @inbounds ptbuf[ix] = pt
      ix += 1
    end
  end
  view(ptbuf, 1:(ix-1))
end

function updateptsbuf!(ptbuf::Vector{T}, ptv::Vector{T}, idxs) where{T}
  for (j, idx) in enumerate(idxs)
    ptbuf[j] = ptv[idx]
  end
  view(ptbuf, 1:length(idxs))
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
  (logdet(Sf)*size(data, 2) + sum(abs2, Sf.U'\data))/2
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

check_canonical_sorted(pts) = false
check_canonical_sorted(pts::Vector{SVector{1,Float64}}) = issorted(getindex.(pts, 1))

function fastknn_routine_available(pts::Vector{SVector{D,Float64}}, 
                                   design::KNNConditioning{M}) where{D,M}
  in(D, (2,3,4)) || return false
  M == Euclidean || return false
  true
end

function chordal_completion(U_tri::UpperTriangular)
  U          = U_tri.data
  (n, I, J)  = (size(U_tri, 1), Int64[], Int64[])
  max_c      = maximum(diff(U.colptr))-1
  size_bound = n + n*max_c + n*div(max_c*(max_c-1), 2)
  sizehint!(I, size_bound)
  sizehint!(J, size_bound)
  # Generate the chordal completion that will indicate what values of S need to
  # be computed by selinv-type logic, allowing duplicates here that will be
  # handled by the sparse constructor.
  for c in 1:n
    push!(I, c)
    push!(J, c)
    ptr_start = U.colptr[c]
    ptr_end = U.colptr[c+1] - 1
    for ptr_k in ptr_start:ptr_end
      k = U.rowval[ptr_k]
      k >= c && continue
      push!(I, k)
      push!(J, c)
      for ptr_l in ptr_start:(ptr_k-1)
        rl = U.rowval[ptr_l]
        push!(I, rl)
        push!(J, k)
      end
    end
  end
  # Allocate the Sparse Matrix S (duplicates are safely merged by sparse)
  S = sparse(I, J, ones(Float64, length(I)), n, n)
  fill!(S.nzval, 0.0)
  S
end

function takahashi_diagonal(U_tri::UpperTriangular)
  (n, U) = (size(U_tri,1), U_tri.data)
  S      = chordal_completion(U_tri)
  for c in 1:n
    U_cc = U[c, c]
    (S_start, S_end) = (S.colptr[c], S.colptr[c+1] - 1)
    (U_start, U_end) = (U.colptr[c], U.colptr[c+1] - 1)
    # Calculate off-diagonals
    for S_ptr in S_start:S_end
      r = S.rowval[S_ptr]
      r == c && continue
      sum_val = 0.0
      for U_ptr in U_start:U_end
        k = U.rowval[U_ptr]
        k == c && continue
        row_idx = min(r, k)
        col_idx = max(r, k)
        sum_val += U[k, c] * S[row_idx, col_idx]
      end
      S.nzval[S_ptr] = -sum_val / U_cc
    end
    # Calculate diagonal
    sum_diag = 0.0
    for U_ptr in U_start:U_end
      k = U.rowval[U_ptr]
      k == c && continue
      sum_diag += U[k, c] * S[k, c]
    end
    S.nzval[S_end] = (1.0 / U_cc) * ((1.0 / U_cc) - sum_diag)
  end
  diag(S)
end
