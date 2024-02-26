using OffsetArrays: OffsetArray
using Oceananigans.Utils
using Oceananigans.Grids: architecture
using KernelAbstractions.Extras.LoopInfo: @unroll

import Base

#####
##### General halo filling functions
#####

fill_halo_regions!(::Nothing, args...; kwargs...) = nothing
fill_halo_regions!(::NamedTuple{(), Tuple{}}, args...; kwargs...) = nothing

"""
    fill_halo_regions!(fields::Union{Tuple, NamedTuple}, arch, args...)

Fill halo regions for each field in the tuple `fields` according to their boundary
conditions, possibly recursing into `fields` if it is a nested tuple-of-tuples.
"""
# Some fields have `nothing` boundary conditions, such as `FunctionField` and `ZeroField`.
fill_halo_regions!(c::OffsetArray, ::Nothing, args...; kwargs...) = nothing

# Returns the boundary conditions a specific side for `FieldBoundaryConditions` inputs and
# a tuple of boundary conditions for `NTuple{N, <:FieldBoundaryConditions}` inputs
for dir in (:west, :east, :south, :north, :bottom, :top)
    extract_side_bc = Symbol(:extract_, dir, :_bc)
    @eval begin
        @inline $extract_side_bc(bc) = bc.$dir
        @inline $extract_side_bc(bc::Tuple) = map($extract_side_bc, bc)
    end
end
#
@inline extract_bc(bc, ::Val{:west})   = tuple(extract_west_bc(bc))
@inline extract_bc(bc, ::Val{:east})   = tuple(extract_east_bc(bc))
@inline extract_bc(bc, ::Val{:south})  = tuple(extract_south_bc(bc))
@inline extract_bc(bc, ::Val{:north})  = tuple(extract_north_bc(bc))
@inline extract_bc(bc, ::Val{:bottom}) = tuple(extract_bottom_bc(bc))
@inline extract_bc(bc, ::Val{:top})    = tuple(extract_top_bc(bc))

@inline extract_bc(bc, ::Val{:west_and_east})   = (extract_west_bc(bc), extract_east_bc(bc))
@inline extract_bc(bc, ::Val{:south_and_north}) = (extract_south_bc(bc), extract_north_bc(bc))
@inline extract_bc(bc, ::Val{:bottom_and_top})  = (extract_bottom_bc(bc), extract_top_bc(bc))

# Finally, the true fill_halo!
const MaybeTupledData = Union{OffsetArray, NTuple{<:Any, OffsetArray}}

"Fill halo regions in ``x``, ``y``, and ``z`` for a given field's data."
function fill_halo_regions!(c::MaybeTupledData, boundary_conditions, indices, loc, grid, args...; kwargs...)

    arch = architecture(grid)

    fill_halos!, bcs = permute_boundary_conditions(boundary_conditions)
<<<<<<< HEAD
    number_of_tasks = length(fill_halos!)
=======
    number_of_tasks  = length(fill_halos!)
>>>>>>> origin/ss-glw/time-bcs

    # Fill halo in the three permuted directions (1, 2, and 3), making sure dependencies are fulfilled
    for task = 1:number_of_tasks
        fill_halo_event!(c, fill_halos![task], bcs[task], indices, loc, arch, grid, args...; kwargs...)
    end

    return nothing
end

function fill_halo_event!(c, fill_halos!, bcs, indices, loc, arch, grid, args...; kwargs...)

    # Calculate size and offset of the fill_halo kernel
    # We assume that the kernel size is the same for west and east boundaries, 
    # south and north boundaries, and bottom and top boundaries
    size   = fill_halo_size(c, fill_halos!, indices, bcs[1], loc, grid)
    offset = fill_halo_offset(size, fill_halos!, indices)

    fill_halos!(c, bcs..., size, offset, loc, arch, grid, args...; kwargs...)

    return nothing
end

# In case of a DistributedCommunication paired with a 
# Flux, Value or Gradient boundary condition, we split the direction in two single-sided
# fill_halo! events (see issue #3342)
# `permute_boundary_conditions` returns a 2-tuple containing the ordered operations to execute in 
# position [1] and the associated boundary conditions in position [2]
function permute_boundary_conditions(boundary_conditions)

    split_x_halo_filling = split_halo_filling(extract_west_bc(boundary_conditions),  extract_east_bc(boundary_conditions))
    split_y_halo_filling = split_halo_filling(extract_south_bc(boundary_conditions), extract_north_bc(boundary_conditions))

    west_bc  = extract_west_bc(boundary_conditions)
    east_bc  = extract_east_bc(boundary_conditions)
    south_bc = extract_south_bc(boundary_conditions)
    north_bc = extract_north_bc(boundary_conditions)
    
    if split_x_halo_filling
        if split_y_halo_filling
            fill_halos! = [fill_west_halo!, fill_east_halo!, fill_south_halo!, fill_north_halo!, fill_bottom_and_top_halo!]
            sides       = [:west, :east, :south, :north, :bottom_and_top]
            bcs_array   = [west_bc, east_bc, south_bc, north_bc, extract_bottom_bc(boundary_conditions)]
        else
            fill_halos! = [fill_west_halo!, fill_east_halo!, fill_south_and_north_halo!, fill_bottom_and_top_halo!]
            sides       = [:west, :east, :south_and_north, :bottom_and_top]
            bcs_array   = [west_bc, east_bc, south_bc, extract_bottom_bc(boundary_conditions)]
        end
    else
        if split_y_halo_filling
            fill_halos! = [fill_west_and_east_halo!, fill_south_halo!, fill_north_halo!, fill_bottom_and_top_halo!]
            sides       = [:west_and_east, :south, :north, :bottom_and_top]
            bcs_array   = [west_bc, south_bc, north_bc, extract_bottom_bc(boundary_conditions)]
        else
            fill_halos! = [fill_west_and_east_halo!, fill_south_and_north_halo!, fill_bottom_and_top_halo!]
            sides       = [:west_and_east, :south_and_north, :bottom_and_top]
            bcs_array   = [west_bc, south_bc, extract_bottom_bc(boundary_conditions)]
        end
    end

    perm = sortperm(bcs_array, lt=fill_first)
    fill_halos! = fill_halos![perm]
    sides = sides[perm]

    boundary_conditions = Tuple(extract_bc(boundary_conditions, Val(side)) for side in sides)

    return fill_halos!, boundary_conditions
end

# Split direction in two distinct fill_halo! events in case of a communication boundary condition 
# (distributed DCBC), paired with a Flux, Value or Gradient boundary condition
split_halo_filling(bcs1, bcs2)     = false
split_halo_filling(::DCBC, ::DCBC) = false
split_halo_filling(bcs1, ::DCBC)   = true
split_halo_filling(::DCBC, bcs2)   = true

# TODO: support heterogeneous distributed-shared communication
# split_halo_filling(::MCBC, ::DCBC) = false
# split_halo_filling(::DCBC, ::MCBC) = false
# split_halo_filling(::MCBC, ::MCBC) = false
# split_halo_filling(bcs1, ::MCBC)   = true
# split_halo_filling(::MCBC, bcs2)   = true

#####
##### Halo filling order
#####

const PBCT  = Union{PBC,  NTuple{<:Any, <:PBC}}
const MCBCT = Union{MCBC, NTuple{<:Any, <:MCBC}}
const DCBCT = Union{DCBC, NTuple{<:Any, <:DCBC}}

# Distributed halos have to be filled last to allow the 
# possibility of asynchronous communication: 
# If other halos are filled after we initiate the distributed communication, 
# (but before communication is completed) the halos will be overwritten. 
# For this reason we always want to perform local halo filling first and then 
# initiate communication

# Periodic is handled after Flux, Value, Gradient because
# Periodic fills also corners while Flux, Value, Gradient do not
# TODO: remove this ordering requirement (see issue https://github.com/CliMA/Oceananigans.jl/issues/3342)

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
##### Double-sided fill_halo! kernels
#####

@kernel function _fill_west_and_east_halo!(c, west_bc, east_bc, loc, grid, args) 
    j, k = @index(Global, NTuple)
    _fill_west_halo!(j, k, grid, c, west_bc, loc, args...)
    _fill_east_halo!(j, k, grid, c, east_bc, loc, args...)
end

@kernel function _fill_south_and_north_halo!(c, south_bc, north_bc, loc, grid, args)
    i, k = @index(Global, NTuple)
    _fill_south_halo!(i, k, grid, c, south_bc, loc, args...)
    _fill_north_halo!(i, k, grid, c, north_bc, loc, args...)
end

@kernel function _fill_bottom_and_top_halo!(c, bottom_bc, top_bc, loc, grid, args)
    i, j = @index(Global, NTuple)
    _fill_bottom_halo!(i, j, grid, c, bottom_bc, loc, args...)
       _fill_top_halo!(i, j, grid, c, top_bc,    loc, args...)
end

#####
##### Single-sided fill_halo! kernels
#####

@kernel function _fill_only_west_halo!(c, bc, loc, grid, args)
    j, k = @index(Global, NTuple)
    _fill_west_halo!(j, k, grid, c, bc, loc, args...)
end

@kernel function _fill_only_south_halo!(c, bc, loc, grid, args)
    i, k = @index(Global, NTuple)
    _fill_south_halo!(i, k, grid, c, bc, loc, args...)
end

@kernel function _fill_only_bottom_halo!(c, bc, loc, grid, args)
    i, j = @index(Global, NTuple)
    _fill_bottom_halo!(i, j, grid, c, bc, loc, args...)
end

@kernel function _fill_only_east_halo!(c, bc, loc, grid, args)
    j, k = @index(Global, NTuple)
    _fill_east_halo!(j, k, grid, c, bc, loc, args...)
end

@kernel function _fill_only_north_halo!(c, bc, loc, grid, args)
    i, k = @index(Global, NTuple)
    _fill_north_halo!(i, k, grid, c, bc, loc, args...)
end

@kernel function _fill_only_top_halo!(c, bc, loc, grid, args)
    i, j = @index(Global, NTuple)
    _fill_top_halo!(i, j, grid, c, bc, loc, args...)
end

#####
##### Tupled double-sided fill_halo! kernels
#####

# Note, we do not need tupled single-sided fill_halo! kernels since `DCBC` do not 
# support tupled halo filling
import Oceananigans.Utils: @constprop

@kernel function _fill_west_and_east_halo!(c::NTuple, west_bc, east_bc, loc, grid, args)
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

@kernel function _fill_south_and_north_halo!(c::NTuple, south_bc, north_bc, loc, grid, args) 
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

@kernel function _fill_bottom_and_top_halo!(c::NTuple, bottom_bc, top_bc, loc, grid, args) 
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

#####
##### Kernel launchers for single-sided fill_halos
#####

fill_west_halo!(c, bc, size, offset, loc, arch, grid, args...; kwargs...) = 
    launch!(arch, grid, KernelParameters(size, offset),
            _fill_only_west_halo!, c, bc, loc, grid, Tuple(args); kwargs...)

fill_east_halo!(c, bc, size, offset, loc, arch, grid, args...; kwargs...) = 
    launch!(arch, grid, KernelParameters(size, offset),
            _fill_only_east_halo!, c, bc, loc, grid, Tuple(args); kwargs...)

fill_south_halo!(c, bc, size, offset, loc, arch, grid, args...; kwargs...) = 
    launch!(arch, grid, KernelParameters(size, offset),
            _fill_only_south_halo!, c, bc, loc, grid, Tuple(args); kwargs...)

fill_north_halo!(c, bc, size, offset, loc, arch, grid, args...; kwargs...) = 
    launch!(arch, grid, KernelParameters(size, offset),
            _fill_only_north_halo!, c, bc, loc, grid, Tuple(args); kwargs...)

fill_bottom_halo!(c, bc, size, offset, loc, arch, grid, args...; kwargs...) = 
    launch!(arch, grid, KernelParameters(size, offset),
            _fill_only_bottom_halo!, c, bc, loc, grid, Tuple(args); kwargs...)

fill_top_halo!(c, bc, size, offset, loc, arch, grid, args...; kwargs...) = 
    launch!(arch, grid, KernelParameters(size, offset),
            _fill_only_top_halo!, c, bc, loc, grid, Tuple(args); kwargs...)

#####
##### Kernel launchers for double-sided fill_halos
#####

function fill_west_and_east_halo!(c, west_bc, east_bc, size, offset, loc, arch, grid, args...; kwargs...)
    return launch!(arch, grid, KernelParameters(size, offset),
                   _fill_west_and_east_halo!, c, west_bc, east_bc, loc, grid, Tuple(args); kwargs...)
end

function fill_south_and_north_halo!(c, south_bc, north_bc, size, offset, loc, arch, grid, args...; kwargs...)
    return launch!(arch, grid, KernelParameters(size, offset),
                   _fill_south_and_north_halo!, c, south_bc, north_bc, loc, grid, Tuple(args); kwargs...)
end

function fill_bottom_and_top_halo!(c, bottom_bc, top_bc, size, offset, loc, arch, grid, args...; kwargs...)
    return launch!(arch, grid, KernelParameters(size, offset),
                   _fill_bottom_and_top_halo!, c, bottom_bc, top_bc, loc, grid, Tuple(args); kwargs...)
end

#####
##### Calculate kernel size and offset for Windowed and Sliced Fields
#####

const WEB = Union{typeof(fill_west_and_east_halo!), typeof(fill_west_halo!), typeof(fill_east_halo!)}
const SNB = Union{typeof(fill_south_and_north_halo!), typeof(fill_south_halo!), typeof(fill_north_halo!)}
const TBB = Union{typeof(fill_bottom_and_top_halo!), typeof(fill_bottom_halo!), typeof(fill_top_halo!)}

# Tupled halo filling _only_ deals with full fields!
@inline fill_halo_size(::Tuple, ::WEB, args...) = :yz
@inline fill_halo_size(::Tuple, ::SNB, args...) = :xz
@inline fill_halo_size(::Tuple, ::TBB, args...) = :xy

# If indices are colon, and locations are _not_ Nothing, fill the whole boundary plane!
# If locations are _Nothing_, then the kwarg `reduced_dimensions` will allow the size `:xz`
# to be correctly interpreted inside `launch!`.
@inline fill_halo_size(::OffsetArray, ::WEB, ::Tuple{<:Any, <:Colon, <:Colon}, args...) = :yz
@inline fill_halo_size(::OffsetArray, ::SNB, ::Tuple{<:Colon, <:Any, <:Colon}, args...) = :xz
@inline fill_halo_size(::OffsetArray, ::TBB, ::Tuple{<:Colon, <:Colon, <:Any}, args...) = :xy

# If the index is a Colon and the location is _NOT_ a `Nothing` (i.e. not a `ReducedField`), 
# then fill the whole boundary, otherwise fill the size of the corresponding array
@inline whole_halo(idx, loc)           = false
@inline whole_halo(idx,     ::Nothing) = false
@inline whole_halo(::Colon, ::Nothing) = false
@inline whole_halo(::Colon,       loc) = true

# Calculate kernel size for windowed fields. This code is only called when
# one or more of the elements of `idx` is not Colon in the two direction perpendicular
# to the halo region and `bc` is not `PeriodicBoundaryCondition`.
@inline function fill_halo_size(c::OffsetArray, ::WEB, idx, bc, loc, grid)
    @inbounds begin
        whole_y_halo = whole_halo(idx[2], loc[2])
        whole_z_halo = whole_halo(idx[3], loc[3])
    end

    _, Ny, Nz = size(grid)
    _, Cy, Cz = size(c)

    Sy = ifelse(whole_y_halo, Ny, Cy)
    Sz = ifelse(whole_z_halo, Nz, Cz)

    return (Sy, Sz)
end

@inline function fill_halo_size(c::OffsetArray, ::SNB, idx, bc, loc, grid)
    @inbounds begin
        whole_x_halo = whole_halo(idx[1], loc[1])
        whole_z_halo = whole_halo(idx[3], loc[3])
    end

    Nx, _, Nz = size(grid)
    Cx, _, Cz = size(c)

    Sx = ifelse(whole_x_halo, Nx, Cx)
    Sz = ifelse(whole_z_halo, Nz, Cz)

    return (Sx, Sz)
end
                      
@inline function fill_halo_size(c::OffsetArray, ::TBB, idx, bc, loc, grid)
    @inbounds begin
        whole_x_halo = whole_halo(idx[1], loc[1])
        whole_y_halo = whole_halo(idx[2], loc[2])
    end

    Nx, Ny, _ = size(grid)
    Cx, Cy, _ = size(c)

    Sx = ifelse(whole_x_halo, Nx, Cx)
    Sy = ifelse(whole_y_halo, Ny, Cy)

    return (Sx, Sy)
end

# Remember that Periodic BCs also fill halo points!
@inline fill_halo_size(c::OffsetArray, ::WEB, idx, ::PBC, args...) = tuple(size(c, 2), size(c, 3))
@inline fill_halo_size(c::OffsetArray, ::SNB, idx, ::PBC, args...) = tuple(size(c, 1), size(c, 3))
@inline fill_halo_size(c::OffsetArray, ::TBB, idx, ::PBC, args...) = tuple(size(c, 1), size(c, 2))

@inline function fill_halo_size(c::OffsetArray, ::WEB, ::Tuple{<:Any, <:Colon, <:Colon}, ::PBC, args...)
    _, Cy, Cz = size(c)
    return (Cy, Cz)
end

@inline function fill_halo_size(c::OffsetArray, ::SNB, ::Tuple{<:Colon, <:Any, <:Colon}, ::PBC, args...)
    Cx, _, Cz = size(c)
    return (Cx, Cz)
end

@inline function fill_halo_size(c::OffsetArray, ::TBB, ::Tuple{<:Colon, <:Colon, <:Any}, ::PBC, args...)
    Cx, Cy, _ = size(c)
    return (Cx, Cy)
end

# The offsets are non-zero only if the indices are not Colon
@inline fill_halo_offset(::Symbol, args...)   = (0, 0)
@inline fill_halo_offset(::Tuple, ::WEB, idx) = (idx[2] == Colon() ? 0 : first(idx[2])-1, idx[3] == Colon() ? 0 : first(idx[3])-1)
@inline fill_halo_offset(::Tuple, ::SNB, idx) = (idx[1] == Colon() ? 0 : first(idx[1])-1, idx[3] == Colon() ? 0 : first(idx[3])-1)
@inline fill_halo_offset(::Tuple, ::TBB, idx) = (idx[1] == Colon() ? 0 : first(idx[1])-1, idx[2] == Colon() ? 0 : first(idx[2])-1)
