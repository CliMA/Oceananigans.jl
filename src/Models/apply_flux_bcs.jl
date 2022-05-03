using Oceananigans.Grids

using Oceananigans: instantiated_location, prognostic_fields
using Oceananigans.Architectures: AbstractArchitecture, architecture, device
using Oceananigans.BoundaryConditions: west_flux, east_flux, south_flux, north_flux, bottom_flux, top_flux
using Oceananigans.BoundaryConditions: ZFBC
using Oceananigans.Fields: Field
using Oceananigans.Grids: AbstractGrid, flip, XBoundedGrid, YBoundedGrid, ZBoundedGrid
using Oceananigans.Operators: idxᴿ, idxᴸ, Ax, Ay, Az, volume
using Oceananigans.Utils: work_layout, launch!

using KernelAbstractions: @kernel, @index, MultiEvent

import Oceananigans.BoundaryConditions: apply_x_bcs!, apply_y_bcs!, apply_z_bcs!

# Unpack
# args has to be (closure, diffusivity_fields, id, clock, model_fields) --- see below.
apply_x_bcs!(Gc::Field, dep, c::Field, args...) = apply_x_bcs!(Gc, Gc.grid, dep, c, c.boundary_conditions.west,   c.boundary_conditions.east,  args...)
apply_y_bcs!(Gc::Field, dep, c::Field, args...) = apply_y_bcs!(Gc, Gc.grid, dep, c, c.boundary_conditions.south,  c.boundary_conditions.north, args...)
apply_z_bcs!(Gc::Field, dep, c::Field, args...) = apply_z_bcs!(Gc, Gc.grid, dep, c, c.boundary_conditions.bottom, c.boundary_conditions.top,   args...)

@inline apply_x_bcs!(::Nothing, dep, args...) = dep
@inline apply_y_bcs!(::Nothing, dep, args...) = dep
@inline apply_z_bcs!(::Nothing, dep, args...) = dep

"""
    apply_x_bcs!(Gc, grid::XBoundedGrid, dependencies, c, west_bc, east_bc, args...)

Apply flux boundary conditions to a field `c` by adding the associated flux divergence to
the source term `Gc` at the west and east boundary.
"""
apply_x_bcs!(Gc, grid::XBoundedGrid, dependencies, c, west_bc, east_bc, args...) =
    launch!(architecture(grid), grid, :yz, _apply_x_bcs!, Gc, grid, west_bc, east_bc, instantiated_location(c), c, args...; dependencies)

"""
    apply_y_bcs!(Gc, grid::YBoundedGrid, dependencies, c, south_bc, north_bc, args...)

Apply flux boundary conditions to a field `c` by adding the associated flux divergence to
the source term `Gc` at the south and north boundary.
"""
apply_y_bcs!(Gc, grid::YBoundedGrid, dependencies, c, south_bc, north_bc, args...) =
    launch!(architecture(grid), grid, :xz, _apply_y_bcs!, Gc, grid, south_bc, north_bc, instantiated_location(c), c, args...; dependencies)

"""
    apply_z_bcs!(Gc, grid::ZBoundedGrid, c, bottom_bc, top_bc, dep, args...)

Apply flux boundary conditions to a field `c` by adding the associated flux divergence to
the source term `Gc` at the top and bottom boundary.
"""
apply_z_bcs!(Gc, grid::ZBoundedGrid, dependencies, c, bottom_bc, top_bc, args...) =
    launch!(architecture(grid), grid, :xy, _apply_z_bcs!, Gc, grid, bottom_bc, top_bc, instantiated_location(c), c, args...; dependencies)

"""
    _apply_x_bcs!(Gc, grid, west_bc, east_bc, args...)

Apply a west and/or east boundary condition to variable `c`.
"""
@kernel function _apply_x_bcs!(Gc, grid, west_bc, east_bc, loc, c, closure, K, id, clock, fields)
    j, k = @index(Global, NTuple)

    Nx = grid.Nx
    LX, LY, LZ = loc

    # West flux across i = 1
    qᵂ = west_flux(1, j, k, grid, west_bc, loc, c, closure, K, id, clock, fields)
    Axᵂ =       Ax(1, j, k, grid, LX, LY, LZ)
    Vᵂ  =   volume(1, j, k, grid, LX, LY, LZ)

    # Flux across i = Nx
    qᴱ = east_flux(Nx+1, j, k, grid, east_bc, loc, c, closure, K, id, clock, fields)
    Axᴱ =       Ax(Nx+1, j, k, grid, LX, LY, LZ)
    Vᵂ  =   volume(Nx,   j, k, grid, LX, LY, LZ)

    @inbounds Gc[1,  j, k] += qᵂ * Axᵂ / Vᵂ
    @inbounds Gc[Nx, j, k] -= qᴱ * Axᴱ / Vᴱ 
end

"""
    _apply_y_bcs!(Gc, grid, south_bc, north_bc, args...)

Apply a south and/or north boundary condition to variable `c`.
"""
@kernel function _apply_y_bcs!(Gc, grid, south_bc, north_bc, loc, c, closure, K, id, clock, fields)
    i, k = @index(Global, NTuple)

    Ny = grid.Ny
    LX, LY, LZ = loc

    # Flux across j = 1
    qˢ  = south_flux(i, 1, k, grid, south_bc, loc, c, closure, K, id, clock, fields)
    Ayˢ =         Ay(i, 1, k, grid, LX, LY, LZ)
    Vˢ  =     volume(i, 1, k, grid, LX, LY, LZ)

    # Flux across j = Ny
    qᴺ  = north_flux(i, Ny+1, k, grid, north_bc, loc, c, closure, K, id, clock, fields)
    Ayᴺ =         Ay(i, Ny+1, k, grid, LX, LY, LZ)
    Vᴺ  =     volume(i, Ny,   k, grid, LX, LY, LZ)

    @inbounds Gc[i, 1,  k] += qˢ * Ayˢ / Vˢ
    @inbounds Gc[i, Ny, k] -= qᴺ * Ayᴺ / Vᴺ 
end

"""
    _apply_z_bcs!(Gc, grid, bottom_bc, top_bc, args...)

Apply a top and/or bottom boundary condition to variable `c`.
"""
@kernel function _apply_z_bcs!(Gc, grid, bottom_bc, top_bc, loc, c, closure, K, id, clock, fields)
    i, j = @index(Global, NTuple)

    Nz = grid.Nz
    LX, LY, LZ = loc

    # Flux across southern domain boundary
    qᴮ  = bottom_flux(i, j, 1, grid, bottom_bc, loc, c, closure, K, id, clock, fields)
    Azᴮ =          Az(i, j, 1, grid, LX, LY, LZ)
    Vᴮ  =      volume(i, j, 1, grid, LX, LY, LZ)

    # Flux across northern domain boundary
    qᵀ  = top_flux(i, j, Nz+1, grid, top_bc, loc, c, closure, K, id, clock, fields)
    Azᵀ =       Ay(i, j, Nz+1, grid, LX, LY, LZ)
    Vᵀ  =   volume(i, j, Nz,   grid, LX, LY, LZ)

    @inbounds Gc[i, j, 1]  += qᴮ * Azᴮ / Vᴮ
    @inbounds Gc[i, j, Nz] -= qᵀ * Azᵀ / Vᵀ 
end

#####
##### Boundary tendency contributions
#####

""" Apply boundary conditions by adding flux divergences to the right-hand-side. """
function calculate_boundary_tendency_contributions!(model)
    Gⁿ = model.timestepper.Gⁿ
    K = model.diffusivity_fields
    grid = model.grid
    arch = model.architecture
    velocities = model.velocities
    tracers = model.tracers
    clock = model.clock
    model_fields = merge(model.velocities, model.tracers)
    closure = model.closure
    barrier = device_event(arch)

    events = []

    Φ = prognostic_fields(model)
    Nfields = length(Φ)
    ids = [q > 3 ? Val(q) : nothing for q = 1:Nfields]
    events = [barrier for _ = 1:Nfields]

    if grid isa XBoundedGrid
        for q = 1:Nfields
            events[q] = apply_x_bcs!(Gⁿ[q], barrier, Φ[q], closure, K, ids[q], clock, model_fields)
        end
    end

    if grid isa YBoundedGrid
        for q = 1:Nfields
            events[q] = apply_y_bcs!(Gⁿ[q], barrier, Φ[q], closure, K, ids[q], clock, model_fields)
        end
    end

    if grid isa ZBoundedGrid
        for q = 1:Nfields
            events[q] = apply_z_bcs!(Gⁿ[q], barrier, Φ[q], closure, K, ids[q], clock, model_fields)
        end
    end

    wait(device(arch), MultiEvent(Tuple(events)))

    return nothing
end

