using OffsetArrays: OffsetArray
using Oceananigans.Utils
using Oceananigans.Architectures: device_event
using Oceananigans.Grids: architecture
using KernelAbstractions.Extras.LoopInfo: @unroll

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
        @inline $extract_bc(bc::Tuple) = $extract_bc.(bc)
    end
end

# Finally, the true fill_halo!
const MaybeTupledData = Union{OffsetArray, NTuple{<:Any, OffsetArray}}

"Fill halo regions in ``x``, ``y``, and ``z`` for a given field's data."
function fill_halo_regions!(c::MaybeTupledData, boundary_conditions, indices, loc, grid, args...; kwargs...)

    arch = architecture(grid)

    halo_tuple = permute_boundary_conditions(boundary_conditions)
   
    for task = 1:3
        barrier = device_event(arch)
        fill_halo_event!(task, halo_tuple, c, indices, loc, arch, barrier, grid, args...; kwargs...)
    end

    return nothing
end

function fill_halo_event!(task, halo_tuple, c, indices, loc, arch, barrier, grid, args...; kwargs...)
    fill_halo!  = halo_tuple[1][task]
    bc_left     = halo_tuple[2][task]
    bc_right    = halo_tuple[3][task]

    size   = fill_halo_size(c, fill_halo!, indices, bc_left, grid)

    offset = fill_halo_offset(size, fill_halo!, indices)

    event      = fill_halo!(c, bc_left, bc_right, size, offset, loc, arch, barrier, grid, args...; kwargs...)
    wait(device(arch), event)
end

function permute_boundary_conditions(boundary_conditions)

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

    return (fill_halos!, boundary_conditions_array_left, boundary_conditions_array_right)
end

@inline validate_event(::Nothing) = NoneEvent()
@inline validate_event(event)     = event

#####
##### Halo filling order
#####

const PBCT = Union{PBC, NTuple{<:Any, <:PBC}}
const CBCT = Union{CBC, NTuple{<:Any, <:CBC}}

fill_first(bc1::PBCT, bc2)       = false
fill_first(bc1::CBCT, bc2)       = false
fill_first(bc1::PBCT, bc2::CBCT) = false
fill_first(bc1::CBCT, bc2::PBCT) = true
fill_first(bc1, bc2::PBCT)       = true
fill_first(bc1, bc2::CBCT)       = true
fill_first(bc1::PBCT, bc2::PBCT) = true
fill_first(bc1::CBCT, bc2::CBCT) = true
fill_first(bc1, bc2)             = true

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

@kernel function _fill_west_and_east_halo!(c::NTuple, west_bc, east_bc, offset, loc, grid, args...)
    j, k = @index(Global, NTuple)
    ntuple(Val(length(west_bc))) do n
        Base.@_inline_meta
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
        @inbounds begin
            _fill_bottom_halo!(i, j, grid, c[n], bottom_bc[n], loc[n], args...)
               _fill_top_halo!(i, j, grid, c[n], top_bc[n],    loc[n], args...)
        end
    end
end

fill_west_and_east_halo!(c, west_bc, east_bc, size, offset, loc, arch, dep, grid, args...; kwargs...) =
    launch!(arch, grid, size, _fill_west_and_east_halo!, c, west_bc, east_bc, offset, loc, grid, args...; dependencies=dep, kwargs...)

fill_south_and_north_halo!(c, south_bc, north_bc, size, offset, loc, arch, dep, grid, args...; kwargs...) = 
    launch!(arch, grid, size, _fill_south_and_north_halo!, c, south_bc, north_bc, offset, loc, grid, args...; dependencies=dep, kwargs...)

fill_bottom_and_top_halo!(c, bottom_bc, top_bc, size, offset, loc, arch, dep, grid, args...; kwargs...) =
    launch!(arch, grid, size, _fill_bottom_and_top_halo!, c, bottom_bc, top_bc, offset, loc, grid, args...; dependencies=dep, kwargs...)

#####
##### Pass kernel size and offset for Windowed and Sliced Fields
#####

const YZFullIndex = Tuple{<:Any, <:Colon, <:Colon}
const XZFullIndex = Tuple{<:Colon, <:Any, <:Colon}
const XYFullIndex = Tuple{<:Colon, <:Colon, <:Any}

const WEB = typeof(fill_west_and_east_halo!)
const SNB = typeof(fill_south_and_north_halo!)
const TBB = typeof(fill_bottom_and_top_halo!)

# In case of a tuple we are _always_ dealing with full fields!
fill_halo_size(::Tuple, ::WEB, idx) = :yz
fill_halo_size(::Tuple, ::SNB, idx) = :xz
fill_halo_size(::Tuple, ::TBB, idx) = :xy

# If indices are colon, just fill the whole boundary!
fill_halo_size(::OffsetArray, ::WEB, ::YZFullIndex) = :yz
fill_halo_size(::OffsetArray, ::SNB, ::XZFullIndex) = :xz
fill_halo_size(::OffsetArray, ::TBB, ::XYFullIndex) = :xy

# If they are not... we have to calculate the size!
fill_halo_size(c::OffsetArray, ::WEB, idx, bc, grid) = (idx[2] == Colon() ? size(grid, 2) : size(c, 2), idx[3] == Colon() ? size(grid, 3) : size(c, 3))
fill_halo_size(c::OffsetArray, ::SNB, idx, bc, grid) = (idx[1] == Colon() ? size(grid, 1) : size(c, 1), idx[3] == Colon() ? size(grid, 3) : size(c, 3))
fill_halo_size(c::OffsetArray, ::TBB, idx, bc, grid) = (idx[1] == Colon() ? size(grid, 1) : size(c, 1), idx[2] == Colon() ? size(grid, 2) : size(c, 2))

# Remember that Periodic BC have to fill also the halo points always!
fill_halo_size(c::OffsetArray, ::WEB, idx, ::PBC, grid) = size(c)[[2, 3]]
fill_halo_size(c::OffsetArray, ::SNB, idx, ::PBC, grid) = size(c)[[1, 3]]
fill_halo_size(c::OffsetArray, ::TBB, idx, ::PBC, grid) = size(c)[[1, 2]]

fill_halo_offset(::Symbol, args...)    = (0, 0)
fill_halo_offset(::Tuple, ::WEB, idx)  = (idx[2] == Colon() ? 0 : first(idx[2])-1, idx[3] == Colon() ? 0 : first(idx[3])-1)
fill_halo_offset(::Tuple, ::SNB, idx)  = (idx[1] == Colon() ? 0 : first(idx[1])-1, idx[3] == Colon() ? 0 : first(idx[3])-1)
fill_halo_offset(::Tuple, ::TBB, idx)  = (idx[1] == Colon() ? 0 : first(idx[1])-1, idx[2] == Colon() ? 0 : first(idx[2])-1)
