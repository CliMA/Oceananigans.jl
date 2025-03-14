using Oceananigans.Grids: total_length, topology

using OffsetArrays: OffsetArray


#####
##### Creating offset arrays for field data by dispatching on architecture.
#####

"""
Return a range of indices for a field located at either cell `Center`s or `Face`s along a
grid dimension which is `Periodic`, or cell `Center`s for a grid dimension which is `Bounded`.
The dimension has length `N` and `H` halo points.
"""
offset_indices(loc, topo, N, H=0) = 1 - H : N + H

"""
Return a range of indices for a field located at cell `Face`s along a grid dimension which
is `Bounded` and has length `N` and with halo points `H`.
"""
offset_indices(::Face, ::BoundedTopology, N, H=0) = 1 - H : N + H + 1

"""
Return a range of indices for a field along a 'reduced' dimension.
"""
offset_indices(::Nothing, topo, N, H=0) = 1:1

offset_indices(ℓ,         topo, N, H, ::Colon) = offset_indices(ℓ, topo, N, H)
offset_indices(ℓ,         topo, N, H, r::UnitRange) = r
offset_indices(::Nothing, topo, N, H, ::UnitRange) = 1:1

instantiate(T::Type) = T()
instantiate(t) = t

converted_offset(IntType, i) = convert(IntType, i)

# A unit range is converted to an integer offset corresponding
# to the first index of the range minus one.
function converted_offset(IntType, r::UnitRange) 
    i_start = convert(IntType, r[1])
    return i_start - IntType(1)
end

# OneTo ranges have offset 0!
converted_offset(IntType, ::Base.OneTo) = IntType(0)
converted_offset(IntType, t::Tuple) = Tuple(converted_offset(IntType, i) for i in t)

function find_minimum_precision(ii::Integer)
    maxInt8  = typemax(Int8)
    maxInt16 = typemax(Int16)
    maxInt32 = typemax(Int32)
    IntType = ii > maxInt8 ? (ii > maxInt16 ? (ii > maxInt32 ? Int64 : Int32) : Int16) : Int8
    return IntType
end

find_minimum_precision(ii::UnitRange) = find_minimum_precision(first(ii))
    
function find_minimum_precision(ii::Union{AbstractArray, Tuple, AbstractRange})
    IntTypes = Tuple(find_minimum_precision(i) for i in ii)
    IntType = Int64 ∈ IntTypes ? Int64 : (Int32 ∈ IntTypes ? Int32 : (Int16 ∈ IntTypes ? Int16 : Int8))
    return IntType
end

convert_offsets(ii) = converted_offset(find_minimum_precision(ii), ii)

# The type parameter for indices helps / encourages the compiler to fully type infer `offset_data`
function offset_data(underlying_data::A, loc, topo, N, H, indices::T=default_indices(length(loc))) where {A<:AbstractArray, T}
    loc  = map(instantiate, loc)
    topo = map(instantiate, topo)
    ii   = map(offset_indices, loc, topo, N, H, indices)
    # Add extra indices for arrays of higher dimension than loc, topo, etc.
    # Use the "`ntuple` trick" so the compiler can infer the type of `extra_ii`
    extra_ii = ntuple(Val(ndims(underlying_data)-length(ii))) do i
        Base.@_inline_meta
        axes(underlying_data, i+length(ii))
    end

    ii = (ii..., extra_ii...)
    ii = convert_offsets(ii)

    return OffsetArray(underlying_data, ii...)
end

"""
    offset_data(underlying_data, grid::AbstractGrid, loc, indices=default_indices(length(loc)))

Return an `OffsetArray` that maps to `underlying_data` in memory, with offset indices
appropriate for the `data` of a field on a `grid` of `size(grid)` and located at `loc`.
"""
offset_data(underlying_data::AbstractArray, grid::AbstractGrid, loc, indices=default_indices(length(loc))) =
    offset_data(underlying_data, loc, topology(grid), size(grid), halo_size(grid), indices)

"""
    new_data(FT, arch, loc, topo, sz, halo_sz, indices)

Return an `OffsetArray` of zeros of float type `FT` on `arch`itecture,
with indices corresponding to a field on a `grid` of `size(grid)` and located at `loc`.
"""
function new_data(FT::DataType, arch, loc, topo, sz, halo_sz, indices=default_indices(length(loc)))
    Tsz = total_size(loc, topo, sz, halo_sz, indices)
    underlying_data = zeros(arch, FT, Tsz...)
    indices = validate_indices(indices, loc, topo, sz, halo_sz)
    return offset_data(underlying_data, loc, topo, sz, halo_sz, indices)
end

new_data(FT::DataType, grid::AbstractGrid, loc, indices=default_indices(length(loc))) =
    new_data(FT, architecture(grid), loc, topology(grid), size(grid), halo_size(grid), indices)

new_data(grid::AbstractGrid, loc, indices=default_indices) = new_data(eltype(grid), grid, loc, indices)

