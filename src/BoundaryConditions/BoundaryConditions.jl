module BoundaryConditions

export
    BCType, Flux, Gradient, Value, Open,
    BoundaryCondition, getbc, setbc!,
    PeriodicBoundaryCondition, OpenBoundaryCondition, NoFluxBoundaryCondition, CommunicationBoundaryCondition,
    FluxBoundaryCondition, ValueBoundaryCondition, GradientBoundaryCondition,
    validate_boundary_condition_topology, validate_boundary_condition_architecture,
    FieldBoundaryConditions,
    fill_halo_regions!

using CUDA
using KernelAbstractions: @index, @kernel, MultiEvent, NoneEvent

using Oceananigans.Architectures: CPU, GPU, device
using Oceananigans.Utils: work_layout, launch!
using Oceananigans.Grids

using Oceananigans.Grids: AbstractGrid, flip, XBoundedGrid, YBoundedGrid, ZBoundedGrid

include("boundary_condition_classifications.jl")
include("boundary_condition.jl")
include("discrete_boundary_function.jl")
include("continuous_boundary_function.jl")
include("field_boundary_conditions.jl")
include("show_boundary_conditions.jl")

include("fill_halo_regions.jl")
include("fill_halo_regions_open.jl")
include("fill_halo_regions_periodic.jl")

#####
##### Interface for calculating boundary fluxes
#####
##### NOTE SIGN CONVENTION...
#####

@inline   west_flux(i, j, k, grid, bc::FBC, loc, c, closure, K, id, args...) = + getbc(bc, j, k, grid, args...)
@inline   east_flux(i, j, k, grid, bc::FBC, loc, c, closure, K, id, args...) = + getbc(bc, j, k, grid, args...)
@inline  south_flux(i, j, k, grid, bc::FBC, loc, c, closure, K, id, args...) = + getbc(bc, i, k, grid, args...)
@inline  north_flux(i, j, k, grid, bc::FBC, loc, c, closure, K, id, args...) = + getbc(bc, i, k, grid, args...)
@inline bottom_flux(i, j, k, grid, bc::FBC, loc, c, closure, K, id, args...) = + getbc(bc, i, j, grid, args...)
@inline    top_flux(i, j, k, grid, bc::FBC, loc, c, closure, K, id, args...) = + getbc(bc, i, j, grid, args...)

@inline   west_flux(i, j, k, grid, args...) = zero(grid)
@inline   east_flux(i, j, k, grid, args...) = zero(grid)
@inline  south_flux(i, j, k, grid, args...) = zero(grid)
@inline  north_flux(i, j, k, grid, args...) = zero(grid)
@inline bottom_flux(i, j, k, grid, args...) = zero(grid)
@inline    top_flux(i, j, k, grid, args...) = zero(grid)

# Interface for "applying" boundary conditions

@inline apply_x_bcs!(Gc, dep, args...) = dep
@inline apply_y_bcs!(Gc, dep, args...) = dep
@inline apply_z_bcs!(Gc, dep, args...) = dep

@inline apply_x_bcs!(Gc, grid::XBoundedGrid, dep, c, ::ZFBC, ::ZFBC, args...) = dep
@inline apply_y_bcs!(Gc, grid::YBoundedGrid, dep, c, ::ZFBC, ::ZFBC, args...) = dep
@inline apply_z_bcs!(Gc, grid::ZBoundedGrid, dep, c, ::ZFBC, ::ZFBC, args...) = dep

@inline apply_x_bcs!(Gc, grid::XBoundedGrid, dep, c, ::Nothing, ::Nothing, args...) = dep
@inline apply_y_bcs!(Gc, grid::YBoundedGrid, dep, c, ::Nothing, ::Nothing, args...) = dep
@inline apply_z_bcs!(Gc, grid::ZBoundedGrid, dep, c, ::Nothing, ::Nothing, args...) = dep

end # module
