using Oceananigans.BoundaryConditions: BoundaryCondition, Open, PerturbationAdvection, FlatExtrapolation
using Oceananigans.AbstractOperations: Average
using Oceananigans.Fields: Field

const OBC  = BoundaryCondition{<:Open} # OpenBoundaryCondition (with no matching scheme)

# Left boundary averages for normal velocity components
west_average(u)   = Field(Average(view(u, 1, :, :), dims=(2, 3)))[]
south_average(v)  = Field(Average(view(v, :, 1, :), dims=(1, 3)))[]
bottom_average(w) = Field(Average(view(w, :, :, 1), dims=(1, 2)))[]

# Right boundary averages for normal velocity components
east_average(u)   = Field(Average(view(u, u.grid.Nx + 1, :, :), dims=(2, 3)))[]
north_average(v)  = Field(Average(view(v, :, v.grid.Ny + 1, :), dims=(1, 3)))[]
top_average(w)    = Field(Average(view(w, :, :, w.grid.Nz + 1), dims=(1, 2)))[]

"""
    initialize_boundary_mass_fluxes(velocities::NamedTuple)

Initialize boundary mass fluxes for boundaries with OpenBoundaryConditions,
returning a NamedTuple of boundary fluxes.
"""
function initialize_boundary_mass_fluxes(velocities::NamedTuple)

    u_bcs = velocities.u.boundary_conditions
    v_bcs = velocities.v.boundary_conditions
    w_bcs = velocities.w.boundary_conditions

    boundary_fluxes = NamedTuple()

    # Check west boundary (u velocity)
    if u_bcs.west isa OBC
        west_flux = west_average(velocities.u)
        boundary_fluxes = merge(boundary_fluxes, (west = west_flux,))
    end

    # Check east boundary (u velocity)
    if u_bcs.east isa OBC
        east_flux = east_average(velocities.u)
        boundary_fluxes = merge(boundary_fluxes, (east = east_flux,))
    end

    # Check south boundary (v velocity)
    if v_bcs.south isa OBC
        south_flux = south_average(velocities.v)
        boundary_fluxes = merge(boundary_fluxes, (south = south_flux,))
    end

    # Check north boundary (v velocity)
    if v_bcs.north isa OBC
        north_flux = north_average(velocities.v)
        boundary_fluxes = merge(boundary_fluxes, (north = north_flux,))
    end

    # Check bottom boundary (w velocity)
    if w_bcs.bottom isa OBC
        bottom_flux = bottom_average(velocities.w)
        boundary_fluxes = merge(boundary_fluxes, (bottom = bottom_flux,))
    end

    # Check top boundary (w velocity)
    if w_bcs.top isa OBC
        top_flux = top_average(velocities.w)
        boundary_fluxes = merge(boundary_fluxes, (top = top_flux,))
    end

    return boundary_fluxes
end