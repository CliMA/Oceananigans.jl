using OffsetArrays: OffsetArray
using Oceananigans.Utils
using Oceananigans.Grids: architecture
using KernelAbstractions.Extras.LoopInfo: @unroll

import Base

#####
##### General halo filling functions
#####

fill_halo_regions!(::Nothing, args...) = nothing
fill_halo_regions!(::NamedTuple{(), Tuple{}}, args...) = nothing

"""
    fill_halo_regions!(fields::Union{Tuple, NamedTuple}, arch, args...)

Fill halo regions for each field in the tuple `fields` according to their boundary
conditions, possibly recursing into `fields` if it is a nested tuple-of-tuples.
"""
# Some fields have `nothing` boundary conditions, such as `FunctionField` and `ZeroField`.
fill_halo_regions!(c::OffsetArray, ::Nothing, args...; kwargs...) = nothing

for dir in (:west, :east, :south, :north, :bottom, :top)
    extract_bc = Symbol(:extract_, dir, :_bc)
    @eval begin
        @inline $extract_bc(bc) = bc.$dir
        @inline $extract_bc(bc::Tuple) = map($extract_bc, bc)
    end
end

# For inhomogeneous BC we extract the _last_ one 
# example 
# `bc.west <: DCBC`
# `bc.east <: PBC`
# `extract_west_or_east_bc(bc) == bc.west`
# NOTE that `isless` follows order of execution, 
# so `max(bcs...)` returns the last BC to execute

  extract_west_or_east_bc(bc) = max(bc.west,   bc.east)
extract_south_or_north_bc(bc) = max(bc.south,  bc.north)
 extract_bottom_or_top_bc(bc) = max(bc.bottom, bc.top)

  extract_west_or_east_bc(bc::Tuple) =   map(extract_west_or_east_bc, bc)
extract_south_or_north_bc(bc::Tuple) = map(extract_south_or_north_bc, bc)
 extract_bottom_or_top_bc(bc::Tuple) =  map(extract_bottom_or_top_bc, bc)
 
# Finally, the true fill_halo!
const MaybeTupledData = Union{OffsetArray, NTuple{<:Any, OffsetArray}}

"Fill halo regions in ``x``, ``y``, and ``z`` for a given field's data."
function fill_halo_regions!(c::MaybeTupledData, boundary_conditions, indices, loc, grid, args...; kwargs...)

    arch = architecture(grid)

    halo_tuple  = permute_boundary_conditions(boundary_conditions)

    # Fill halo in the three permuted directions (1, 2, and 3), making sure dependencies are fulfilled
    for task in 1:3
        fill_halo_event!(task, halo_tuple, c, indices, loc, arch, grid, args...; kwargs...)
    end

    return nothing
end

function fill_halo_event!(task, halo_tuple, c, indices, loc, arch, grid, args...; kwargs...)
    fill_halo!  = halo_tuple[1][task]
    bc_left     = halo_tuple[2][task]
    bc_right    = halo_tuple[3][task]

    # Calculate size and offset of the fill_halo kernel
    size   = fill_halo_size(c, fill_halo!, indices, bc_left, loc, grid)
    offset = fill_halo_offset(size, fill_halo!, indices)

    fill_halo!(c, bc_left, bc_right, size, offset, loc, arch, grid, args...; kwargs...)
    return
end

function permute_boundary_conditions(boundary_conditions)

    fill_halos! = [
        fill_west_and_east_halo!,
        fill_south_and_north_halo!,
        fill_bottom_and_top_halo!,
    ]

    boundary_conditions_array = [
        extract_west_or_east_bc(boundary_conditions),
        extract_south_or_north_bc(boundary_conditions),
        extract_bottom_or_top_bc(boundary_conditions)
    ]

    boundary_conditions_array_left = [
        extract_west_bc(boundary_conditions),
        extract_south_bc(boundary_conditions),
        extract_bottom_bc(boundary_conditions)
    ]

    boundary_conditions_array_right = [
        extract_east_bc(boundary_conditions),
        extract_north_bc(boundary_conditions),
        extract_top_bc(boundary_conditions),
    ]

    perm = sortperm(boundary_conditions_array, lt=fill_first)
    fill_halos! = fill_halos![perm]
    boundary_conditions_array_left  = boundary_conditions_array_left[perm]
    boundary_conditions_array_right = boundary_conditions_array_right[perm]

    return (fill_halos!, boundary_conditions_array_left, boundary_conditions_array_right)
end

#####
##### Halo filling order
#####

const PBCT  = Union{PBC,  NTuple{<:Any, <:PBC}}
const MCBCT = Union{MCBC, NTuple{<:Any, <:MCBC}}
const DCBCT = Union{DCBC, NTuple{<:Any, <:DCBC}}

# Distributed halos have to be filled for last in case of 
# buffered communication. Hence, we always fill them last

# The reasoning for filling Periodic after Flux, Value, Gradient 
# Periodic fills also corners while Flux, Value, Gradient do not

# Order of halo filling
# 1) Flux, Value, Gradient (TODO: remove these BC and apply them as fluxes)
# 2) Periodic (PBCT)
# 3) Shared Communication (MCBCT)
# 4) Distributed Communication (DCBCT)

# We define "greater than" `>` and "lower than", for boundary conditions
# following the rules outlined in `fill_first`
# i.e. if `bc1 > bc2` then `bc2` precedes `bc1` in filling order
@inline Base.isless(bc1::BoundaryCondition, bc2::BoundaryCondition) = fill_first(bc1, bc2)

# fallback for `Nothing` BC.
@inline Base.isless(::Nothing,           ::Nothing) = true
@inline Base.isless(::BoundaryCondition, ::Nothing) = false
@inline Base.isless(::Nothing, ::BoundaryCondition) = true
@inline Base.isless(::BoundaryCondition, ::Missing) = false
@inline Base.isless(::Missing, ::BoundaryCondition) = true

fill_first(bc1::DCBCT, bc2)        = false
fill_first(bc1::PBCT,  bc2::DCBCT) = true
fill_first(bc1::DCBCT, bc2::PBCT)  = false
fill_first(bc1::MCBCT, bc2::DCBCT) = true
fill_first(bc1::DCBCT, bc2::MCBCT) = false
fill_first(bc1, bc2::DCBCT)        = true
fill_first(bc1::DCBCT, bc2::DCBCT) = true
fill_first(bc1::PBCT,  bc2)        = false
fill_first(bc1::MCBCT, bc2)        = false
fill_first(bc1::PBCT,  bc2::MCBCT) = true
fill_first(bc1::MCBCT, bc2::PBCT)  = false
fill_first(bc1, bc2::PBCT)         = true
fill_first(bc1, bc2::MCBCT)        = true
fill_first(bc1::PBCT,  bc2::PBCT)  = true
fill_first(bc1::MCBCT, bc2::MCBCT) = true
fill_first(bc1, bc2)               = true

#####
##### General fill_halo! kernels
#####

@kernel function _fill_west_and_east_halo!(c, west_bc, east_bc, offset, loc, grid, args...)
    j, k = @index(Global, NTuple)
    j′ = j + offset[1]
    k′ = k + offset[2]
    _fill_west_halo!(j′, k′, grid, c, west_bc, loc, args...)
    _fill_east_halo!(j′, k′, grid, c, east_bc, loc, args...)
end

@kernel function _fill_south_and_north_halo!(c, south_bc, north_bc, offset, loc, grid, args...)
    i, k = @index(Global, NTuple)
    i′ = i + offset[1]
    k′ = k + offset[2]
    _fill_south_halo!(i′, k′, grid, c, south_bc, loc, args...)
    _fill_north_halo!(i′, k′, grid, c, north_bc, loc, args...)
end

@kernel function _fill_bottom_and_top_halo!(c, bottom_bc, top_bc, offset, loc, grid, args...)
    i, j = @index(Global, NTuple)
    i′ = i + offset[1]
    j′ = j + offset[2]
    _fill_bottom_halo!(i′, j′, grid, c, bottom_bc, loc, args...)
       _fill_top_halo!(i′, j′, grid, c, top_bc,    loc, args...)
end

#####
##### Tuple fill_halo! kernels
#####

import Oceananigans.Utils: @constprop

@kernel function _fill_west_and_east_halo!(c::NTuple, west_bc, east_bc, offset, loc, grid, args...)
    j, k = @index(Global, NTuple)
    ntuple(Val(length(west_bc))) do n
        Base.@_inline_meta
        @constprop(:aggressive) # TODO constprop failure on `loc[n]`
        @inbounds begin
            _fill_west_halo!(j, k, grid, c[n], west_bc[n], loc[n], args...)
            _fill_east_halo!(j, k, grid, c[n], east_bc[n], loc[n], args...)
        end
    end
end

@kernel function _fill_south_and_north_halo!(c::NTuple, south_bc, north_bc, offset, loc, grid, args...)
    i, k = @index(Global, NTuple)
    ntuple(Val(length(south_bc))) do n
        Base.@_inline_meta
        @constprop(:aggressive) # TODO constprop failure on `loc[n]`
        @inbounds begin
            _fill_south_halo!(i, k, grid, c[n], south_bc[n], loc[n], args...)
            _fill_north_halo!(i, k, grid, c[n], north_bc[n], loc[n], args...)
        end
    end
end

@kernel function _fill_bottom_and_top_halo!(c::NTuple, bottom_bc, top_bc, offset, loc, grid, args...)
    i, j = @index(Global, NTuple)
    ntuple(Val(length(bottom_bc))) do n
        Base.@_inline_meta
        @constprop(:aggressive) # TODO constprop failure on `loc[n]`
        @inbounds begin
            _fill_bottom_halo!(i, j, grid, c[n], bottom_bc[n], loc[n], args...)
               _fill_top_halo!(i, j, grid, c[n], top_bc[n],    loc[n], args...)
        end
    end
end

fill_west_and_east_halo!(c, west_bc, east_bc, size, offset, loc, arch, grid, args...; kwargs...) =
    launch!(arch, grid, size, _fill_west_and_east_halo!, c, west_bc, east_bc, offset, loc, grid, args...; kwargs...)

fill_south_and_north_halo!(c, south_bc, north_bc, size, offset, loc, arch, grid, args...; kwargs...) =
    launch!(arch, grid, size, _fill_south_and_north_halo!, c, south_bc, north_bc, offset, loc, grid, args...; kwargs...)

fill_bottom_and_top_halo!(c, bottom_bc, top_bc, size, offset, loc, arch, grid, args...; kwargs...) =
    launch!(arch, grid, size, _fill_bottom_and_top_halo!, c, bottom_bc, top_bc, offset, loc, grid, args...; kwargs...)

#####
##### Calculate kernel size and offset for Windowed and Sliced Fields
#####

const WEB = typeof(fill_west_and_east_halo!)
const SNB = typeof(fill_south_and_north_halo!)
const TBB = typeof(fill_bottom_and_top_halo!)

# Tupled halo filling _only_ deals with full fields!
@inline fill_halo_size(::Tuple, ::WEB, args...) = :yz
@inline fill_halo_size(::Tuple, ::SNB, args...) = :xz
@inline fill_halo_size(::Tuple, ::TBB, args...) = :xy

# If indices are colon, fill the whole boundary plane!
@inline fill_halo_size(::OffsetArray, ::WEB, ::Tuple{<:Any, <:Colon, <:Colon}, args...) = :yz
@inline fill_halo_size(::OffsetArray, ::SNB, ::Tuple{<:Colon, <:Any, <:Colon}, args...) = :xz
@inline fill_halo_size(::OffsetArray, ::TBB, ::Tuple{<:Colon, <:Colon, <:Any}, args...) = :xy

# If the index is a Colon and the location is _NOT_ a `Nothing` (i.e. not a `ReducedField`), 
# then fill the whole boundary, otherwise fill the size of the corresponding array
@inline whole_halo(idx, loc)           = false
@inline whole_halo(idx,     ::Nothing) = false
@inline whole_halo(::Colon, ::Nothing) = false
@inline whole_halo(::Colon,       loc) = true

# Calculate kernel size
@inline fill_halo_size(c::OffsetArray, ::WEB, idx, bc, loc, grid) =
    @inbounds (ifelse(whole_halo(idx[2], loc[2]), size(grid, 2), size(c, 2)), ifelse(whole_halo(idx[3], loc[3]), size(grid, 3), size(c, 3)))
@inline fill_halo_size(c::OffsetArray, ::SNB, idx, bc, loc, grid) =
    @inbounds (ifelse(whole_halo(idx[1], loc[1]), size(grid, 1), size(c, 1)), ifelse(whole_halo(idx[3], loc[3]), size(grid, 3), size(c, 3)))
@inline fill_halo_size(c::OffsetArray, ::TBB, idx, bc, loc, grid) =
    @inbounds (ifelse(whole_halo(idx[1], loc[1]), size(grid, 1), size(c, 1)), ifelse(whole_halo(idx[2], loc[2]), size(grid, 2), size(c, 2)))

# Remember that Periodic BCs also fill halo points!
@inline fill_halo_size(c::OffsetArray, ::WEB, idx, ::PBC, args...) = @inbounds size(c)[[2, 3]]
@inline fill_halo_size(c::OffsetArray, ::SNB, idx, ::PBC, args...) = @inbounds size(c)[[1, 3]]
@inline fill_halo_size(c::OffsetArray, ::TBB, idx, ::PBC, args...) = @inbounds size(c)[[1, 2]]
@inline fill_halo_size(c::OffsetArray, ::WEB, ::Tuple{<:Any, <:Colon, <:Colon}, ::PBC, args...) = @inbounds size(c)[[2, 3]]
@inline fill_halo_size(c::OffsetArray, ::SNB, ::Tuple{<:Colon, <:Any, <:Colon}, ::PBC, args...) = @inbounds size(c)[[1, 3]]
@inline fill_halo_size(c::OffsetArray, ::TBB, ::Tuple{<:Colon, <:Colon, <:Any}, ::PBC, args...) = @inbounds size(c)[[1, 2]]

# The offsets are non-zero only if the indices are not Colon
@inline fill_halo_offset(::Symbol, args...)    = (0, 0)
@inline fill_halo_offset(::Tuple, ::WEB, idx)  = (idx[2] == Colon() ? 0 : first(idx[2])-1, idx[3] == Colon() ? 0 : first(idx[3])-1)
@inline fill_halo_offset(::Tuple, ::SNB, idx)  = (idx[1] == Colon() ? 0 : first(idx[1])-1, idx[3] == Colon() ? 0 : first(idx[3])-1)
@inline fill_halo_offset(::Tuple, ::TBB, idx)  = (idx[1] == Colon() ? 0 : first(idx[1])-1, idx[2] == Colon() ? 0 : first(idx[2])-1)
