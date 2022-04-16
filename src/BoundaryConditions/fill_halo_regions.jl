using OffsetArrays: OffsetArray
using Oceananigans.Architectures: device_event
using Oceananigans.Grids: architecture
using KernelAbstractions.Extras.LoopInfo: @unroll

#####
##### General halo filling functions
#####

fill_halo_regions!(::Nothing, args...) = nothing

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
        @inline $extract_bc(bc::Tuple) = $extract_bc.(bc)
    end
end

# Finally, the true fill_halo!
const MaybeTupledData = Union{OffsetArray, NTuple{<:Any, OffsetArray}}

"Fill halo regions in ``x``, ``y``, and ``z`` for a given field's data."
function fill_halo_regions!(c::MaybeTupledData, boundary_conditions, loc::Tuple, grid, args...; kw...)

    arch = architecture(grid)

    fill_halos! = [
        fill_west_and_east_halo!,
        fill_south_and_north_halo!,
        fill_bottom_and_top_halo!,
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

    perm = sortperm(boundary_conditions_array_left, lt=fill_first)
    fill_halos! = fill_halos![perm]
    boundary_conditions_array_left  = boundary_conditions_array_left[perm]
    boundary_conditions_array_right = boundary_conditions_array_right[perm]
   
    for task = 1:3
        barrier    = device_event(arch)
        fill_halo! = fill_halos![task]
        bc_left    = boundary_conditions_array_left[task]
        bc_right   = boundary_conditions_array_right[task]
        events     = fill_halo!(c, bc_left, bc_right, loc, arch, barrier, grid, args...; kw...)
       
        wait(device(arch), events)
    end

    return nothing
end

@inline validate_event(::Nothing) = NoneEvent()
@inline validate_event(event)     = event

#####
##### Halo filling order
#####

const PBCT = Union{PBC, NTuple{<:Any, <:PBC}}

fill_first(bc1::PBCT, bc2)       = false
fill_first(bc1, bc2::PBCT)       = true
fill_first(bc1::PBCT, bc2::PBCT) = true
fill_first(bc1, bc2)             = true

#####
##### General fill_halo! kernels
#####

@kernel function _fill_west_and_east_halo!(c, west_bc, east_bc, loc, grid, args...)
    j, k = @index(Global, NTuple)
    _fill_west_halo!(j, k, grid, c, west_bc, loc, args...)
    _fill_east_halo!(j, k, grid, c, east_bc, loc, args...)
end

@kernel function _fill_south_and_north_halo!(c, south_bc, north_bc, loc, grid, args...)
    i, k = @index(Global, NTuple)
    _fill_south_halo!(i, k, grid, c, south_bc, loc, args...)
    _fill_north_halo!(i, k, grid, c, north_bc, loc, args...)
end

@kernel function _fill_bottom_and_top_halo!(c, bottom_bc, top_bc, loc, grid, args...)
    i, j = @index(Global, NTuple)
    _fill_bottom_halo!(i, j, grid, c, bottom_bc, loc, args...)
       _fill_top_halo!(i, j, grid, c, top_bc,    loc, args...)
end

#####
##### Tuple fill_halo! kernels
#####

@kernel function _fill_west_and_east_halo!(c::NTuple, west_bc, east_bc, loc, grid, args...)
    j, k = @index(Global, NTuple)
    ntuple(Val(length(west_bc))) do n
        Base.@_inline_meta
        @inbounds begin
            _fill_west_halo!(j, k, grid, c[n], west_bc[n], loc[n], args...)
            _fill_east_halo!(j, k, grid, c[n], east_bc[n], loc[n], args...)
        end
    end
end

@kernel function _fill_south_and_north_halo!(c::NTuple, south_bc, north_bc, loc, grid, args...)
    i, k = @index(Global, NTuple)
    ntuple(Val(length(south_bc))) do n
        Base.@_inline_meta
        @inbounds begin
            _fill_south_halo!(i, k, grid, c[n], south_bc[n], loc[n], args...)
            _fill_north_halo!(i, k, grid, c[n], north_bc[n], loc[n], args...)
        end
    end
end

@kernel function _fill_bottom_and_top_halo!(c::NTuple, bottom_bc, top_bc, loc, grid, args...)
    i, j = @index(Global, NTuple)
    ntuple(Val(length(bottom_bc))) do n
        Base.@_inline_meta
        @inbounds begin
            _fill_bottom_halo!(i, j, grid, c[n], bottom_bc[n], loc[n], args...)
               _fill_top_halo!(i, j, grid, c[n], top_bc[n],    loc[n], args...)
        end
    end
end

fill_west_and_east_halo!(c, west_bc, east_bc, loc, arch, dep, grid, args...; kwargs...) =
    launch!(arch, grid, :yz, _fill_west_and_east_halo!, c, west_bc, east_bc, loc, grid, args...; dependencies=dep, kwargs...)

fill_south_and_north_halo!(c, south_bc, north_bc, loc, arch, dep, grid, args...; kwargs...) = 
    launch!(arch, grid, :xz, _fill_south_and_north_halo!, c, south_bc, north_bc, loc, grid, args...; dependencies=dep, kwargs...)

fill_bottom_and_top_halo!(c, bottom_bc, top_bc, loc, arch, dep, grid, args...; kwargs...) =
    launch!(arch, grid, :xy, _fill_bottom_and_top_halo!, c, bottom_bc, top_bc, loc, grid, args...; dependencies=dep, kwargs...)
