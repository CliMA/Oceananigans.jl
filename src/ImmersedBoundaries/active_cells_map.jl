using Oceananigans
using Oceananigans.Grids: AbstractGrid

using KernelAbstractions: @kernel, @index

import Oceananigans.Utils: active_cells_work_layout, 
                           use_only_active_interior_cells

const ActiveCellsIBG = ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:AbstractArray}

struct InteriorMap end
struct SurfaceMap end

@inline use_only_active_interior_cells(grid::ActiveCellsIBG) = InteriorMap()

@inline use_only_active_surface_cells(grid::AbstractGrid)   = nothing
@inline use_only_active_surface_cells(grid::ActiveCellsIBG) = SurfaceMap()

@inline active_cells_work_layout(size, ::InteriorMap, grid::ActiveCellsIBG) = min(length(grid.active_cells_interior), 256), length(grid.active_cells_interior)
@inline active_cells_work_layout(size, ::SurfaceMap,  grid::ActiveCellsIBG) = min(length(grid.active_cells_surface),  256), length(grid.active_cells_surface)

@inline active_linear_index_to_interior_tuple(idx, grid::ActiveCellsIBG) = Base.map(Int, grid.active_cells_interior[idx])
@inline  active_linear_index_to_surface_tuple(idx, grid::ActiveCellsIBG) = Base.map(Int, grid.active_cells_surface[idx])

function ImmersedBoundaryGrid(grid, ib, active_cells_map::Bool) 

    ibg = ImmersedBoundaryGrid(grid, ib)
    TX, TY, TZ = topology(ibg)
    
    # Create the cells map on the CPU, then switch it to the GPU
    if active_cells_map 
        map_interior = active_cells_map_interior(ibg)
        map_interior = arch_array(architecture(ibg), map_interior)

        map_surface  = nothing
        # map_surface = active_cells_map_surface(ibg)
        # map_surface = arch_array(architecture(ibg), map_surface)
    else
        map_surface  = nothing
        map_interior = nothing
    end

    return ImmersedBoundaryGrid{TX, TY, TZ}(ibg.underlying_grid, 
                                            ibg.immersed_boundary, 
                                            map_interior)
end

@inline active_cell(i, j, k, ibg) = !immersed_cell(i, j, k, ibg)
@inline active_column(i, j, k, grid, column) = column[i, j, k] != 0

function compute_active_cells_interior(ibg)
    is_immersed_operation = KernelFunctionOperation{Center, Center, Center}(active_cell, ibg)
    active_cells_field = Field{Center, Center, Center}(ibg, Bool)
    set!(active_cells_field, is_immersed_operation)
    return active_cells_field
end

function compute_active_cells_surface(ibg)
    one_field = ConditionalOperation{Center, Center, Center}(OneField(Int), identity, ibg, NotImmersed(truefunc), 0.0)
    column    = sum(one_field, dims = 3)
    is_immersed_column = KernelFunctionOperation{Center, Center, Nothing}(active_column, ibg, computed_dependencies = (column, ))
    active_cells_field = Field{Center, Center, Nothing}(ibg, Bool)
    set!(active_cells_field, is_immersed_column)
    return active_cells_field
end

const MAXUInt8  = 2^8  - 1
const MAXUInt16 = 2^16 - 1
const MAXUInt32 = 2^32 - 1

function active_cells_map_interior(ibg)
    active_cells_field = compute_active_cells_interior(ibg)
    
    N = maximum(size(ibg))
    IntType = N > MAXUInt8 ? (N > MAXUInt16 ? (N > MAXUInt32 ? UInt64 : UInt32) : UInt16) : UInt8
   
    IndicesType = Tuple{IntType, IntType, IntType}

    # Cannot findall on the entire field because we incur on OOM errors
    active_indices = IndicesType[]
    active_indices = findall_active_indices!(active_indices, active_cells_field, ibg, IndicesType)

    return active_indices
end

function findall_active_indices!(active_indices, active_cells_field, ibg, IndicesType)
    
    for k in 1:size(ibg, 3)
        interior_indices = findall(arch_array(CPU(), interior(active_cells_field, :, :, k:k)))
        interior_indices = convert_interior_indices(interior_indices, k, IndicesType)
        active_indices = vcat(active_indices, interior_indices)
        GC.gc()
    end

    return active_indices
end

function convert_interior_indices(interior_indices, k, IndicesType)
    interior_indices =   getproperty.(interior_indices, :I) 
    interior_indices = add_3rd_index.(interior_indices, k) |> Array{IndicesType}
    return interior_indices
end

@inline add_3rd_index(t::Tuple, k) = (t[1], t[2], k) 

function active_cells_map_surface(ibg)
    active_cells_field = compute_active_cells_surface(ibg)
    interior_cells     = arch_array(CPU(), interior(active_cells_field, :, :, 1))
  
    full_indices = findall(interior_cells)

    Nx, Ny, Nz = size(ibg)
    # Reduce the size of the active_cells_map (originally a tuple of Int64)
    N = max(Nx, Ny)
    IntType = N > MAXUInt8 ? (N > MAXUInt16 ? (N > MAXUInt32 ? UInt64 : UInt32) : UInt16) : UInt8
    smaller_indices = getproperty.(full_indices, Ref(:I)) .|> Tuple{IntType, IntType}
    
    return smaller_indices
end

# using Oceananigans.TurbulenceClosures: Riᶜᶜᶠ, _compute_ri_based_diffusivities!, FlavorOfRBVD
# import Oceananigans.TurbulenceClosures: compute_ri_number!, compute_ri_based_diffusivities!

# @kernel function compute_ri_number!(diffusivities, offs, grid::ActiveCellsIBG, closure::FlavorOfRBVD,
#     velocities, tracers, buoyancy, tracer_bcs, clock)
#     idx = @index(Global, Linear)
#     i, j, k = active_linear_index_to_interior_tuple(idx, grid)

#     @inbounds diffusivities.Ri[i, j, k] = Riᶜᶜᶠ(i, j, k, grid, velocities, buoyancy, tracers)
# end

# @kernel function compute_ri_based_diffusivities!(diffusivities, offs, grid::ActiveCellsIBG, closure::FlavorOfRBVD,
#                 velocities, tracers, buoyancy, tracer_bcs, clock)

#     idx = @index(Global, Linear)
#     i, j, k = active_linear_index_to_interior_tuple(idx, grid)
            
#     _compute_ri_based_diffusivities!(i, j, k, diffusivities, grid, closure,
#      velocities, tracers, buoyancy, tracer_bcs, clock)
# end
