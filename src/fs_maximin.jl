
# This code was kindly provided by Florian Schafer (https://f-t-s.github.io) of
# the Courant Institute at NYU. It has been adapted from its original repository
# of https://f-t-s/Kolesky.jl, which is distributed under the MIT license
# (https://github.com/f-t-s/Kolesky.jl/blob/master/LICENSE).

abstract type AbstractNode{Tval, Tid} end

struct Node{Tval, Tid} <: AbstractNode{Tval, Tid}
  val::Tval
  id::Tid
end

struct RankedNode{Tval, Tid, Trank} <: AbstractNode{Tval, Tid}
  val::Tval
  id::Tid
  rank::Trank
end

function vtype(node::AbstractNode)
  return typeof(node.val)
end

function vtype(type::Type{<:AbstractNode{Tval,Tid}}) where {Tval,Tid}
  return Tval
end

function getval(node::AbstractNode)
  return node.val
end

# creates a new node with a different value
function setval(node::Node, val)
  return typeof(node)(val, getid(node))
end

function idtype(node::AbstractNode)
  return typeof(node.id)
end

function idtype(type::Type{<:AbstractNode{Tval,Tid}}) where {Tval,Tid}
  return Tid
end

function getid(node::AbstractNode)
  return node.id
end

function ranktype(node::RankedNode)
  return typeof(node.rank)
end

function getrank(node::RankedNode)
  return node.rank
end

#Mutable Heap (maximal value first)
struct MutableHeap{Tn<:AbstractNode,Tid}
  # Vector containing the nodes of the heap
  nodes::Vector{Tn}
  # Vector providing a lookup for the nodes
  lookup::Vector{Tid}
  # We make sure none of the input nodes has the typmin of Tval, since this this value is reserved to implement popping from the heap
  function MutableHeap{Tn, Tid}(nodes::Vector{Tn}, lookup::Vector{Tid}) where {Tn<:AbstractNode,Tid} 
    !all(getval.(nodes) .> typemin(vtype(Tn))) ? error("typemin of value type reserved") : new{Tn,Tid}(nodes, lookup) 
  end
end

function MutableHeap(values::AbstractVector)
  Tval = eltype(values)
  Tid = Int
  nodes = Vector{Node{Tval,Tid}}(undef, length(values))
  lookup = Vector{Int}(undef, length(values))
  for (id, val) in enumerate(values)
    nodes[id] = Node{Tval,Tid}(val, id)
    lookup[id] = id
  end
  perm = sortperm(nodes,rev=true)
  nodes .= nodes[perm]
  lookup[perm] .= 1 : length(perm)
  return MutableHeap{eltype(nodes), eltype(lookup)}(nodes, lookup)
end

function nodetype(h::MutableHeap{TNode}) where TNode
  return TNode
end

#A function to swap two heapNodes in 
function _swap!(h, a, b)
  # Assigning the new values to the lookup table 
  h.lookup[ h.nodes[a].id ] = b
  h.lookup[ h.nodes[b].id ] = a
  tempNode = h.nodes[a]
  h.nodes[a] = h.nodes[b]
  h.nodes[b] = tempNode
end

#Node comparisons
function Base.isless(a::RankedNode, b::RankedNode)
  isless((-a.rank, a.val), (-b.rank, b.val))
end

function Base.isless(a::Node, b::Node)
  isless(a.val, b.val)
end

function Base.:>=(a::Node, b::Node)
  a.val >= b.val
end

function Base.:>=(a::RankedNode, b::RankedNode)
  (-a.rank, a.val) >= (-b.rank, b.val)
end

function Base.:>(a::RankedNode, b::RankedNode) 
  (-a.rank, a.val) > (-b.rank, b.val)
end

function Base.:>(a::AbstractNode, b::AbstractNode) 
  a.val > b.val
end

#Function that looks at element h.nodes[hInd] and moves it down the tree 
#if it is sufficiently small. Returns the new index if a move took place, 
#and lastindex(h.nodes), else
function _moveDown!(h::MutableHeap, hInd)
  pivot = h.nodes[hInd]
  #If both children exist:
  if 2 * hInd + 1 <= lastindex( h.nodes )
    #If the left child is larger:
    if h.nodes[2 * hInd] >= h.nodes[ 2 * hInd + 1]
      #Check if the child is larger than the parent:
      if h.nodes[2 * hInd] >= pivot
        _swap!( h, hInd, 2 * hInd )
        return 2 * hInd
      else
        #No swap occuring:
        return lastindex( h.nodes )
      end
    #If the left child is larger:
    else
      #Check if the Child is larger than the paren:
      if h.nodes[2 * hInd + 1] >= pivot
        _swap!( h, hInd, 2 * hInd + 1 )
        return  2 * hInd + 1
      else
        #No swap occuring:
        return lastindex( h.nodes )
      end
    end
    #If only one child exists:
  elseif 2 * hInd <= lastindex( h.nodes )
    if h.nodes[2 * hInd] > pivot
      _swap!( h, hInd, 2 * hInd )
      return 2 * hInd 
    end
  end
  #No swap occuring:
  return lastindex( h.nodes )
end

#Get the leading node
function top_node(h::MutableHeap)
  return first(h.nodes)
end

#Gets the leading node and moves it to the back
function top_node!(h::MutableHeap{Node{Tv,Ti},Ti}) where {Tv, Ti}
  out = first(h.nodes)
  # move the node to the very bottom of the list
  update!(h, getid(out), typemin(Tv))
  return out 
end

#Updates (decreases) an element of the heap and restores the heap property
function update!(h::MutableHeap{Node{Tv,Ti},Ti}, id::Ti, val::Tv) where {Tv,Ti}
  tempInd::Ti = h.lookup[id]
  if h.nodes[tempInd].val > val
    h.nodes[tempInd] = setval(h.nodes[tempInd], val)
    while ( tempInd < lastindex( h.nodes ) )
      tempInd = _moveDown!(h, tempInd)
    end
    return val
  else
    return h.nodes[id].val
  end
end

abstract type AbstractMeasurement end
abstract type AbstractPointMeasurement{d}<:AbstractMeasurement end

function get_coordinate(m::AbstractPointMeasurement)
    return m.coordinate
end
struct PointMeasurement{d}<:AbstractPointMeasurement{d}
    coordinate::SVector{d,Float64}
end 

# a point measurement that corresponds to a previously formed matrix
struct PointIndexMeasurement{d}<:AbstractPointMeasurement{d}
    coordinate::SVector{d,Float64}
    index::Int
end


struct ΔδPointMeasurement{Tv,d}<:AbstractPointMeasurement{d}
    coordinate::SVector{d,Float64}
    weight_Δ::Tv
    weight_δ::Tv
end

struct Δ∇δPointMeasurement{Tv,d}<:AbstractPointMeasurement{d} 
    coordinate::SVector{d,Float64}
    weight_Δ::Tv
    weight_∇::SVector{d,Float64}
    weight_δ::Tv
end

struct ∂∂PointMeasurement{Tv,d}<:AbstractPointMeasurement{d}
    coordinate::SVector{d,Float64}
    weight_∂11::Tv
    weight_∂12::Tv
    weight_∂22::Tv
end

function Δ∇δPointMeasurement(in::PointMeasurement{d}) where d
    return Δ∇δPointMeasurement{Float64,d}(in.coordinate, zero(Float64), SVector{d,Float64}(zeros(Float64,d)), one(Float64))
end

function ΔδPointMeasurement(in::PointMeasurement{d}) where d
    return ΔδPointMeasurement{Float64,d}(in.coordinate, zero(Float64), one(Float64))
end

function point_measurements(x::Matrix; dims=1)
    if dims == 2  
        x = x'
    elseif dims !=1
        error("keyword argumend \"dims\" should be 1 or 2")
    end
    d = size(x, 1)
    return [PointMeasurement{d}(SVector{d,Float64}(x[:, k])) for k = 1 : size(x, 2)]
end

function point_index_measurements(x::Matrix; dims=1)
    if dims == 2  
        x = x'
    elseif dims !=1
        error("keyword argumend \"dims\" should be 1 or 2")
    end
    d = size(x, 1)
    return [PointIndexMeasurement{d}(SVector{d,Float64}(x[:, k]), k) for k = 1 : size(x, 2)]
end

# In this file we introduce the data types for super nodes
abstract type AbstractSuperNode end

# A supernode that contains indices to measurements
struct IndexSuperNode{Ti}
    column_indices::Vector{Ti}
    row_indices::Vector{Ti}
end

function column_indices(node::IndexSuperNode) return node.column_indices end
function row_indices(node::IndexSuperNode) return node.row_indices end

function Base.size(in::IndexSuperNode)
    return (length(in.row_indices) , length(in.column_indices))
end

function Base.size(in::IndexSuperNode, dim)
    return (length(in.row_indices), length(in.column_indices))[dim]
end

abstract type AbstractSupernodalAssignment end

# A supernodal assigment given in terms of a 
struct IndirectSupernodalAssignment{Ti<:Integer, Tm<:AbstractMeasurement}
    # A vector containing the index supernodes
    supernodes::Vector{IndexSuperNode{Ti}}
    # A vector containing the measurements
    measurements::Vector{Tm}
end

function IndirectSupernodalAssignment(supernodes::Vector{IndexSuperNode{<:Ti}}, measurements::Vector{<:AbstractMeasurement}) where Ti<:Integer
    return IndirectSupernodalAssignment{Ti,eltyp72, 79e(measurements)}(supernodes, measurements)
end

# importing issorted to overload it. Okay, since only involving custom types
function Base.issorted(node::IndexSuperNode)
    return issorted(node.row_indices) && issorted(node.column_indices)
end

function Base.issorted(assignment::IndirectSupernodalAssignment) 
    # Checking whether each supernode is sorted, and whether the supernodes are sorted according to their first index.
    all(issorted.(assignment.supernodes)) && issorted(first.(getfield.(assignment.supernodes, :column_indices)))
end

# function to update the heap in the case of ordinary distance
function _update_distances!(nearest_distances::AbstractVector, id, new_distance)
    nearest_distances[id] = min(nearest_distances[id], new_distance)
    return new_distance
end

# including x, tree_function
function maximin_ordering(x::AbstractMatrix; init_distances=fill(typemax(eltype(x)), (size(x, 2))), Tree=KDTree)
    # constructing the tree
    N = size(x, 2)
    tree = Tree(x)
    nearest_distances= copy(init_distances)
    @assert length(nearest_distances) == N
    heap = MutableHeap(nearest_distances)
    ℓ = Vector{eltype(init_distances)}(undef, N)
    P = Vector{Int64}(undef, N)
    for k = 1 : N 
        pivot = top_node!(heap)
        ℓ[k] = getval(pivot)
        P[k] = getid(pivot)
        # a little clunky, since inrange doesn't have an option to return range and we want to avoid introducing a 
        # distance measure separate from the NearestNeighbors
        number_in_range = length(inrange(tree, x[:, P[k]], ℓ[k]))            
        ids, dists = knn(tree, x[:, P[k]], number_in_range)
        for (id, dist) in zip(ids, dists)
            if id != getid(pivot)
                # update the distance as stored in nearest_distances
                new_dist = _update_distances!(nearest_distances, id, dist)
                # decreases the distance as stored in the heap
                update!(heap, id, new_dist)
            end
        end
    end
    # returns the maximin ordering P together with the distance vector. 
    return P, ℓ
end

function _gather_assignments(assignments, first_parent) 
    perm = sortperm(assignments) 

    first_indices = unique(i -> assignments[perm[i]], 1 : length(perm))
    push!(first_indices, length(assignments) + 1) 
    ranges = [(first_indices[k] : (first_indices[k + 1] - 1)) for k = 1 : (length(first_indices) - 1)]
    return [perm[range] .+ (first_parent- 1) for range in ranges] 
end

# taking as input the maximin ordering and the associated distances, computes the associated reverse maximin sparsity pattern
# α determines what part of the sparsity pattern arises from the clustering as opposed to the the sparsity pattern of individual points. 
function supernodal_reverse_maximin_sparsity_pattern(x::AbstractMatrix, P, ℓ, ρ; lambda=1.5, alpha=1.0, Tree=KDTree, reconstruct_ordering=true)
    # want to avoid user facing unicode 
    λ = lambda
    α = alpha
    @assert λ > 1.0
    @assert 0.0 <= α <= 1.0
    @assert α * ρ > 1
    # constructing the tree
    N = size(x, 2)
    @assert N == length(P)
    # reordering x according to the reverse maximin ordering $P$ 
    # we will not use the original x. Note that this is a little wasteful for large size(x, 1). 
    x = x[:, P]

    # constructing a maximin ordering that are used as centers of the maximin ordering. 
    if reconstruct_ordering == true 
        P_temp, ℓ_temp = maximin_ordering(x; Tree)
        rev_P_temp = Vector{Int}(undef, N)
        rev_P_temp[P_temp] = 1 : N
    else
        P_temp = copy(P)
        ℓ_temp = copy(ℓ)
        rev_P_temp = Vector{Int}(undef, N)
        rev_P_temp[P_temp] = 1 : N
    end

    supernodes = IndexSuperNode{Int}[]
    children_tree = Tree(x) 
    min_ℓ = ℓ[findfirst(!isinf, ℓ)]
    last_aggregation_point = 1
    last_parent = 0
    # last_parent = findnext(l -> ℓ[l] < min_ℓ / λ, last_parent) - 1 
    while last_parent < N
        # finding the last aggregation index, for which the aggregaton points are sufficiently spread out away from each other
        last_aggregation_point = findnext(l -> (l == N + 1) || (ℓ_temp[l] < α * ρ * min_ℓ), 1 : (N + 1), last_aggregation_point) - 1
        # Constructing the aggregation tree containing only the admissible aggregation points
        aggregation_tree = Tree(x[:, P_temp[1 : last_aggregation_point]])

        # The first parent that we are treating in the present iteration of the while loop is first_parent
        first_parent = last_parent + 1
        # finding the last index l for which ℓ[l] is still within the admissible scale range
        last_parent = findnext(l -> (l + 1 == N + 1) || ℓ[l + 1] < min_ℓ, 1 : N, first_parent)

        # Computing the assignments to supernodes
        assignments = nn(aggregation_tree, x[:, first_parent : last_parent])[1]
        column_indices_list = _gather_assignments(assignments, first_parent)

        for column_indices in column_indices_list
            row_indices = Int[]
            for column_index in column_indices
                # possibly use second parameter here or make dependent on α
                new_row_indices = inrange(children_tree, x[:, column_index], ρ * ℓ[column_index])
                new_row_indices = new_row_indices[findall(new_row_indices .<= column_index)]
                # kind of clumsy way to remove points from previous 
                # levels that are of smaller length scale
                ####################################
                # using distance of Tree function to compute distance of each new row index to the column index
                row_dists = nn(Tree(x[:, column_index:column_index]), x[:, new_row_indices])[2]
                # prune the list of row indices to those that are within distance criterion of the row index length scale, as well
                new_row_indices = new_row_indices[findall(row_dists .<= ρ * ℓ[new_row_indices])]
                ####################################
                append!(row_indices, new_row_indices)
            end
            sort!(row_indices)
            unique!(row_indices)
            push!(supernodes, IndexSuperNode(column_indices, row_indices))
        end

        # updating the length scale 
        min_ℓ = min_ℓ / λ
    end
    return supernodes
end

# high-level driver routine for creating the supernodal ordering and sparisty pattern
# Methods using 1-maximin ordering
function ordering_and_sparsity_pattern(x::AbstractMatrix, ρ; init_distances=fill(typemax(eltype(x)), (size(x, 2))), lambda=1.5, alpha=1.0, Tree=KDTree)
    P, ℓ = maximin_ordering(x; init_distances, Tree)
    supernodes = supernodal_reverse_maximin_sparsity_pattern(x, P, ℓ, ρ; lambda, alpha, Tree)
    return P, ℓ, supernodes
end 


# This function was written by Chris Geoga (chrisgeoga.com), and any mistakes
# here are not due to Florian's initial implementation.
function supernodes_to_condix(n, sn)
  condix = [Int64[]] # first conditioning set is empty.
  # populate the conditioning set, giving some sizehints! as well.
  shint = maximum(length, getfield.(sn, :row_indices))
  for k in 2:n
    ck = Int64[]
    sizehint!(ck, shint)
    push!(condix, ck)
  end
  # loop over the supernodes and add to the relevant condix entry.
  for snj in sn
    (rowj, colj) = (snj.row_indices, snj.column_indices)
    for (rj, cj) in Iterators.product(rowj, colj)
      (rj < cj && !in(rj, condix[cj])) && push!(condix[cj], rj) 
    end
  end
  condix
end

