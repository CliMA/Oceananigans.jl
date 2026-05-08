using Oceananigans.Advection: AbstractAdvectionScheme, fixed_order_scheme
using Oceananigans.ImmersedBoundaries
using Oceananigans.Grids: active_cell, halo_size, topology, XFlatGrid, YFlatGrid, RightConnected, LeftConnected
using Oceananigans.Architectures: cpu_architecture
import Oceananigans.Architectures as AC
using Oceananigans.DistributedComputations: AsynchronousDistributed
using Oceananigans.Fields: Field, interior
using Oceananigans.Utils: @apply_regionally, contiguousrange
using KernelAbstractions: @kernel, @index

# Generic fallback for any grids that aren't specifically supported
@inline function generate_condition_maps(grid, advection; kwargs...)
    return NamedTuple{Tuple(keys(advection))}(ntuple(_ -> nothing, length(keys(advection))))
end

# Currently maintaining this union until condition mapping works on all
# types of grids
const SupportedUnderlyingGrids = Union{LatitudeLongitudeGrid,
                                       RectilinearGrid,
                                       OrthogonalSphericalShellGrid}

const SupportedGrids = Union{SupportedUnderlyingGrids,
                             ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:SupportedUnderlyingGrids}}

@inline function core_workspec(grid)
    architecture(grid) isa AsynchronousDistributed && return distributed_region_kernel_parameters(grid)
    return :xyz
end

struct InteriorBoundarySet{I, B}
    interior :: I
    boundary :: B
end

@inline function generate_condition_maps(grid::SupportedGrids, advection;
                                         condition_momentum_advection=false,
                                         condition_tracer_advection=false)

    workspec = core_workspec(grid)
    condition_maps = Dict()

    for key in keys(advection)
        condition = key == :momentum ? condition_momentum_advection : condition_tracer_advection
        condition_maps[key] = condition ? compute_advection_conditioned_map(advection[key], grid, workspec) : nothing
    end

    return (; condition_maps...)
end

compute_advection_conditioned_map(scheme::Nothing, grid,  workspec) = nothing
compute_advection_conditioned_map(scheme::Nothing, grid,  ::NamedTuple) = nothing
compute_advection_conditioned_map(scheme,          grid, ::Nothing) = nothing
compute_advection_conditioned_map(::Nothing,       grid, ::Nothing) = nothing

function compute_advection_conditioned_map(scheme, grid, workspec)
    # Field is true if the max scheme can be used for the advection
    max_scheme_field = Field{Center, Center, Center}(grid, Bool)
    fill!(max_scheme_field, false)
    launch!(architecture(grid), grid, workspec, _condition_map!, max_scheme_field, grid, scheme)

    interior_indices, boundary_indices = split_indices(max_scheme_field, grid, workspec)
    isempty(boundary_indices) && return nothing
    return InteriorBoundarySet(interior_indices, boundary_indices)
end

function compute_advection_conditioned_map(scheme, grid, regions::NamedTuple{names}) where names
    vals = Tuple(compute_advection_conditioned_map(scheme, grid, regions[name]) for name in names)
    return NamedTuple{names}(vals)
end

@kernel function _condition_map!(max_scheme_field, ibg, scheme)
    i, j, k = @index(Global, NTuple)
    @inbounds max_scheme_field[i, j, k] = convert(Bool, check_interior_xyz(i, j, k, ibg, scheme))
end

split_indices(field, grid, ::Symbol) = split_indices_in_ranges(field, grid, 1:size(grid, 1), 1:size(grid, 2), 1:size(grid, 3))

function split_indices(field, grid, kp::KernelParameters)
    irange, jrange, krange = contiguousrange(kp)
    return split_indices_in_ranges(field, grid, irange, jrange, krange)
end

function split_indices_in_ranges(field, grid, irange, jrange, krange)
    N = maximum(size(grid))
    IT = N > typemax(UInt8) ? (N > typemax(UInt16) ? (N > typemax(UInt32) ? UInt64 : UInt32) : UInt16) : UInt8
    IndicesType = Tuple{IT, IT, IT}
    cpu_arch    = cpu_architecture(architecture(grid))
    cpu_grid    = AC.on_architecture(cpu_arch, grid)
    values      = AC.on_architecture(cpu_arch, interior(field))
    map1 = IndicesType[]
    map2 = IndicesType[]
    for k in krange, j in jrange, i in irange
        active_cell(i, j, k, cpu_grid) || continue
        index = (IT(i), IT(j), IT(k))
        if values[i, j, k]
            push!(map1, index)
        else
            push!(map2, index)
        end
    end
    map1 = AC.on_architecture(architecture(grid), map1)
    map2 = AC.on_architecture(architecture(grid), map2)
    return (map1, map2)
end

function check_interior_xyz(i, j, k, ibg, scheme)
    return (check_interior_x(i, j, k, ibg, scheme)
         && check_interior_y(i, j, k, ibg, scheme)
         && check_interior_z(i, j, k, ibg, scheme))
end

# The condition map is built at (Center, Center, Center). To account for velocities
# read at faces (i-1) and (j-1), we conservatively build the map starting from `-1`
function check_interior_x(i, j, k, ibg, ::AbstractAdvectionScheme{N}) where N
    interior = true
    buffer   = N + 1
    if (i - buffer - 1) < 0 || (i + buffer + 1) > ibg.Nx
      return false
    end
    for di in -buffer-1:buffer+1
        interior = interior && active_cell(i + di, j, k, ibg)
    end
    return interior
end

function check_interior_y(i, j, k, ibg, ::AbstractAdvectionScheme{N}) where N
    interior = true
    buffer   = N + 1
    if (j - buffer - 1) < 0 || (j + buffer + 1) > ibg.Ny
      return false
    end
    for dj in -buffer-1:buffer+1
        interior = interior && active_cell(i, j + dj, k, ibg)
    end
    return interior
end

function check_interior_z(i, j, k, ibg, ::AbstractAdvectionScheme{N}) where N
    interior = true
    buffer   = N + 1
    if (k - buffer - 1) < 0 || (k + buffer + 1) > ibg.Nz
      return false
    end
    for dk in -buffer:buffer
        interior = interior && active_cell(i, j, k + dk, ibg)
    end
    return interior
end

#####
##### Tendency launching kernels with a split map
##### For a generic model to opt-in in this feature the requirements are:
##### - possess a model.condition_maps field constructed using the (generic) structure outlined above
##### - have tendency kernels with signature `kernel!(G, grid, advection, args...)` where advection is the first argument
##### - use maybe_launch_split_tendency_kernels! instead of `launch!`
#####

@inline lookup_condition_map(::Nothing, region)               = nothing
@inline lookup_condition_map(cm::InteriorBoundarySet, region) = cm
@inline lookup_condition_map(cm::NamedTuple, region)          = cm[region]

@inline function maybe_launch_split_tendency_kernels!(arch, kernel_parameters, kernel!,
                                                      tendency, grid, advection, args,
                                                      active_cells_map, condition_map::InteriorBoundarySet)
    fixed_advection = fixed_order_scheme(advection)
    launch!(arch, grid, kernel_parameters, kernel!, tendency, grid,
            (fixed_advection, args...); active_cells_map=condition_map.interior)
    launch!(arch, grid, kernel_parameters, kernel!, tendency, grid,
            (advection, args...); active_cells_map=condition_map.boundary)
    return nothing
end

@inline function maybe_launch_split_tendency_kernels!(arch, kernel_parameters, kernel!,
                                                      tendency, grid, advection, args,
                                                      active_cells_map, ::Nothing)
    launch!(arch, grid, kernel_parameters, kernel!, tendency, grid,
            (advection, args...); active_cells_map)
    return nothing
end
