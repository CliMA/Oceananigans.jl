using Oceananigans.BoundaryConditions: BoundaryCondition, Open, PerturbationAdvection, FlatExtrapolation
using Oceananigans.AbstractOperations: Integral
using Oceananigans.Fields: Field, interior, XFaceField, YFaceField, ZFaceField
using Statistics: mean, filter
using CUDA: @allowscalar

const OBC  = BoundaryCondition{<:Open} # OpenBoundaryCondition
const MatchingScheme = Union{FlatExtrapolation, PerturbationAdvection}
const ROBC = BoundaryCondition{<:Open{<:MatchingScheme}} # Radiation OpenBoundaryCondition
const FOBC = BoundaryCondition{<:Open{<:Nothing}} # "Fixed-velocity" OpenBoundaryCondition (with no matching scheme)

# Left boundary integrals for normal velocity components
@inline west_integral(u)   = Field(Integral(view(u, 1, :, :), dims=(2, 3)))
@inline south_integral(v)  = Field(Integral(view(v, :, 1, :), dims=(1, 3)))
@inline bottom_integral(w) = Field(Integral(view(w, :, :, 1), dims=(1, 2)))

# Right boundary integrals for normal velocity components
@inline east_integral(u)   = Field(Integral(view(u, u.grid.Nx + 1, :, :), dims=(2, 3)))
@inline north_integral(v)  = Field(Integral(view(v, :, v.grid.Ny + 1, :), dims=(1, 3)))
@inline top_integral(w)    = Field(Integral(view(w, :, :, w.grid.Nz + 1), dims=(1, 2)))

function get_west_area(grid)
    f = XFaceField(grid); set!(f, 1) # Create an XFaceField with all ones
    ∫f = Field(Integral(f, dims=(2, 3))) # Integrate over y and z dimensions
    @allowscalar return ∫f[1, 1, 1]
end

function get_east_area(grid)
    f = XFaceField(grid); set!(f, 1) # Create an XFaceField with all ones
    ∫f = Field(Integral(f, dims=(2, 3))) # Integrate over y and z dimensions
    @allowscalar return ∫f[grid.Nx+1, 1, 1]
end

function get_south_area(grid)
    f = YFaceField(grid); set!(f, 1) # Create a YFaceField with all ones
    ∫f = Field(Integral(f, dims=(1, 3))) # Integrate over x and z dimensions
    @allowscalar return ∫f[1, 1, 1]
end

function get_north_area(grid)
    f = YFaceField(grid); set!(f, 1) # Create a YFaceField with all ones
    ∫f = Field(Integral(f, dims=(1, 3))) # Integrate over x and z dimensions
    @allowscalar return ∫f[1, grid.Ny+1, 1]
end

function get_bottom_area(grid)
    f = ZFaceField(grid); set!(f, 1) # Create a ZFaceField with all ones
    ∫f = Field(Integral(f, dims=(1, 2))) # Integrate over x and y dimensions
    @allowscalar return ∫f[1, 1, 1]
end

function get_top_area(grid)
    f = ZFaceField(grid); set!(f, 1) # Create a ZFaceField with all ones
    ∫f = Field(Integral(f, dims=(1, 2))) # Integrate over x and y dimensions
    @allowscalar return ∫f[1, 1, grid.Nz+1]
end

function get_boundary_mass_flux(bc, boundary_flux_field)
    if bc isa FOBC
        if bc.condition isa Number # If the BC is a fixed velocity, the flux is the fixed velocity
            return bc.condition
        elseif bc.condition isa Nothing # If the BC is no-inflow, the flux is zero
            return 0
        else
            return boundary_flux_field
        end
    else
        return boundary_flux_field
    end
end

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
        west_area = get_west_area(velocities.u.grid)
        west_flux = get_boundary_mass_flux(u_bcs.west, Field(west_integral(velocities.u) / west_area))
        boundary_fluxes = merge(boundary_fluxes, (; west = west_flux))
    end

    # Check east boundary (u velocity)
    if u_bcs.east isa OBC
        east_area = get_east_area(velocities.u.grid)
        east_flux = get_boundary_mass_flux(u_bcs.east, Field(east_integral(velocities.u) / east_area))
        boundary_fluxes = merge(boundary_fluxes, (; east = east_flux))
    end

    # Check south boundary (v velocity)
    if v_bcs.south isa OBC
        south_area = get_south_area(velocities.v.grid)
        south_flux = get_boundary_mass_flux(v_bcs.south, Field(south_integral(velocities.v) / south_area))
        boundary_fluxes = merge(boundary_fluxes, (; south = south_flux))
    end

    # Check north boundary (v velocity)
    if v_bcs.north isa OBC
        north_area = get_north_area(velocities.v.grid)
        north_flux = get_boundary_mass_flux(v_bcs.north, Field(north_integral(velocities.v) / north_area))
        boundary_fluxes = merge(boundary_fluxes, (; north = north_flux))
    end

    # Check bottom boundary (w velocity)
    if w_bcs.bottom isa OBC
        bottom_area = get_bottom_area(velocities.w.grid)
        bottom_flux = get_boundary_mass_flux(w_bcs.bottom, Field(bottom_integral(velocities.w) / bottom_area))
        boundary_fluxes = merge(boundary_fluxes, (; bottom = bottom_flux))
    end

    # Check top boundary (w velocity)
    if w_bcs.top isa OBC
        top_area = get_top_area(velocities.w.grid)
        top_flux = get_boundary_mass_flux(w_bcs.top, Field(top_integral(velocities.w) / top_area))
        boundary_fluxes = merge(boundary_fluxes, (; top = top_flux))
    end

    return boundary_fluxes
end

update_open_boundary_mass_fluxes(model) = map(compute!, model.boundary_mass_fluxes)

function open_boundary_mass_fluxes(model)

    update_open_boundary_mass_fluxes(model)

    u_bcs = model.velocities.u.boundary_conditions
    v_bcs = model.velocities.v.boundary_conditions
    w_bcs = model.velocities.w.boundary_conditions

    # Collect left and right ROBC boundary conditions into separate lists
    left_ROBCs = Symbol[]
    right_ROBCs = Symbol[]

    # Initialize fluxes to zero
    left_flux = zero(model.grid)
    right_flux = zero(model.grid)

    # Calculate flux through left boundaries
    if u_bcs.west isa OBC
        left_flux += compute!(model.boundary_mass_fluxes.west)[]
        u_bcs.west isa ROBC && push!(left_ROBCs, :west)
    end
    if v_bcs.south isa OBC
        left_flux += compute!(model.boundary_mass_fluxes.south)[]
        v_bcs.south isa ROBC && push!(left_ROBCs, :south)
    end
    if w_bcs.bottom isa OBC
        left_flux += compute!(model.boundary_mass_fluxes.bottom)[]
        w_bcs.bottom isa ROBC && push!(left_ROBCs, :bottom)
    end

    # Calculate flux through right boundaries
    if u_bcs.east isa OBC
        right_flux += compute!(model.boundary_mass_fluxes.east)[]
        u_bcs.east isa ROBC && push!(right_ROBCs, :east)
    end
    if v_bcs.north isa OBC
        right_flux += compute!(model.boundary_mass_fluxes.north)[]
        v_bcs.north isa ROBC && push!(right_ROBCs, :north)
    end
    if w_bcs.top isa OBC
        right_flux += compute!(model.boundary_mass_fluxes.top)[]
        w_bcs.top isa ROBC && push!(right_ROBCs, :top)
    end

    # Calculate total flux (positive means net inflow)
    total_flux = left_flux - right_flux
    return total_flux, left_ROBCs, right_ROBCs
end

"""
enforce_open_boundary_mass_conservation!(model::NonhydrostaticModel)

Correct boundary mass fluxes for perturbation advection boundary conditions to ensure
zero net mass flux through each boundary.
"""
function enforce_open_boundary_mass_conservation!(model)
    velocities = model.velocities
    grid = model.grid

    total_flux, left_ROBCs, right_ROBCs = open_boundary_mass_fluxes(model)

    # Calculate flux correction per boundary
    extra_flux_per_boundary = total_flux / (length(left_ROBCs) + length(right_ROBCs))

    # Subtract extra flux from left boundaries to reduce inflow
    for bc in left_ROBCs
        if bc == :west
            velocities.u[1, :, :] = velocities.u[1, :, :] .- extra_flux_per_boundary
        elseif bc == :south
            velocities.v[:, 1, :] = velocities.v[:, 1, :] .- extra_flux_per_boundary
        elseif bc == :bottom
            velocities.w[:, :, 1] = velocities.w[:, :, 1] .- extra_flux_per_boundary
        end
    end

    # Add extra flux to right boundaries to increase outflow
    for bc in right_ROBCs
        if bc == :east
            velocities.u[grid.Nx + 1, :, :] = velocities.u[grid.Nx + 1, :, :] .+ extra_flux_per_boundary
        elseif bc == :north
            velocities.v[:, grid.Ny + 1, :] = velocities.v[:, grid.Ny + 1, :] .+ extra_flux_per_boundary
        elseif bc == :top
            velocities.w[:, :, grid.Nz + 1] = velocities.w[:, :, grid.Nz + 1] .+ extra_flux_per_boundary
        end
    end
end
