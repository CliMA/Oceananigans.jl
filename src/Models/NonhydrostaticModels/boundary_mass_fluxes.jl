using Oceananigans.BoundaryConditions: BoundaryCondition, Open, PerturbationAdvection, FlatExtrapolation
using Oceananigans.AbstractOperations: Integral
using Oceananigans.Fields: Field, interior, XFaceField, YFaceField, ZFaceField
using Statistics: mean, filter
using CUDA: @allowscalar

const OBC  = BoundaryCondition{<:Open} # OpenBoundaryCondition
const MatchingScheme = Union{FlatExtrapolation, PerturbationAdvection}
const ROBC = BoundaryCondition{<:Open{<:MatchingScheme}} # Radiation OpenBoundaryCondition
const IOBC = BoundaryCondition{<:Open{<:Nothing}} # "Imposed-velocity" OpenBoundaryCondition (with no matching scheme)
const FIOBC = BoundaryCondition{<:Open{<:Nothing}, <:Number} # "Fixed-imposed-velocity" OpenBoundaryCondition
const ZIOBC = BoundaryCondition{<:Open{<:Nothing}, <:Nothing} # "Zero-imposed-velocity" OpenBoundaryCondition (no-inflow)

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
    if bc isa IOBC
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
    right_ROBCs = Symbol[]
    left_ROBCs = Symbol[]

    # Check west boundary (u velocity)
    if u_bcs.west isa OBC
        west_area = get_west_area(velocities.u.grid)
        west_flux = get_boundary_mass_flux(u_bcs.west, west_integral(velocities.u))
        boundary_fluxes = merge(boundary_fluxes, (; west = west_flux, west_area))
        u_bcs.west isa ROBC && push!(left_ROBCs, :west)
    end

    # Check east boundary (u velocity)
    if u_bcs.east isa OBC
        east_area = get_east_area(velocities.u.grid)
        east_flux = get_boundary_mass_flux(u_bcs.east, east_integral(velocities.u))
        boundary_fluxes = merge(boundary_fluxes, (; east = east_flux, east_area))
        u_bcs.east isa ROBC && push!(right_ROBCs, :east)
    end

    # Check south boundary (v velocity)
    if v_bcs.south isa OBC
        south_area = get_south_area(velocities.v.grid)
        south_flux = get_boundary_mass_flux(v_bcs.south, south_integral(velocities.v))
        boundary_fluxes = merge(boundary_fluxes, (; south = south_flux, south_area))
        v_bcs.south isa ROBC && push!(left_ROBCs, :south)
    end

    # Check north boundary (v velocity)
    if v_bcs.north isa OBC
        north_area = get_north_area(velocities.v.grid)
        north_flux = get_boundary_mass_flux(v_bcs.north, north_integral(velocities.v))
        boundary_fluxes = merge(boundary_fluxes, (; north = north_flux, north_area))
        v_bcs.north isa ROBC && push!(right_ROBCs, :north)
    end

    # Check bottom boundary (w velocity)
    if w_bcs.bottom isa OBC
        bottom_area = get_bottom_area(velocities.w.grid)
        bottom_flux = get_boundary_mass_flux(w_bcs.bottom, bottom_integral(velocities.w))
        boundary_fluxes = merge(boundary_fluxes, (; bottom = bottom_flux, bottom_area))
        w_bcs.bottom isa ROBC && push!(left_ROBCs, :bottom)
    end

    # Check top boundary (w velocity)
    if w_bcs.top isa OBC
        top_area = get_top_area(velocities.w.grid)
        top_flux = get_boundary_mass_flux(w_bcs.top, top_integral(velocities.w))
        boundary_fluxes = merge(boundary_fluxes, (; top = top_flux, top_area))
        w_bcs.top isa ROBC && push!(right_ROBCs, :top)
    end

    boundary_fluxes = merge(boundary_fluxes, (; left_ROBCs, right_ROBCs))
    return boundary_fluxes
end

update_open_boundary_mass_fluxes!(model) = map(compute!, model.boundary_mass_fluxes)

open_boundary_mass_flux(model, bc::OBC, ::Val{:west}, u) = model.boundary_mass_fluxes.west[] / model.boundary_mass_fluxes.west_area
open_boundary_mass_flux(model, bc::OBC, ::Val{:east}, u) = model.boundary_mass_fluxes.east[] / model.boundary_mass_fluxes.east_area
open_boundary_mass_flux(model, bc::OBC, ::Val{:south}, v) = model.boundary_mass_fluxes.south[] / model.boundary_mass_fluxes.south_area
open_boundary_mass_flux(model, bc::OBC, ::Val{:north}, v) = model.boundary_mass_fluxes.north[] / model.boundary_mass_fluxes.north_area
open_boundary_mass_flux(model, bc::OBC, ::Val{:bottom}, w) = model.boundary_mass_fluxes.bottom[] / model.boundary_mass_fluxes.bottom_area
open_boundary_mass_flux(model, bc::OBC, ::Val{:top}, w) = model.boundary_mass_fluxes.top[] / model.boundary_mass_fluxes.top_area

open_boundary_mass_flux(model, bc, side, velocity) = zero(model.grid)

function open_boundary_mass_inflow(model)
    update_open_boundary_mass_fluxes!(model)

    u, v, w = model.velocities
    u_bcs = u.boundary_conditions
    v_bcs = v.boundary_conditions
    w_bcs = w.boundary_conditions

    total_flux = zero(model.grid)

    # Add flux through left boundaries
    total_flux += open_boundary_mass_flux(model, u_bcs.west, Val(:west), u)
    total_flux += open_boundary_mass_flux(model, v_bcs.south, Val(:south), v)
    total_flux += open_boundary_mass_flux(model, w_bcs.bottom, Val(:bottom), w)

    # Subtract flux through right boundaries
    total_flux -= open_boundary_mass_flux(model, u_bcs.east, Val(:east), u)
    total_flux -= open_boundary_mass_flux(model, v_bcs.north, Val(:north), v)
    total_flux -= open_boundary_mass_flux(model, w_bcs.top, Val(:top), w)

    return total_flux
end


correct_left_boundary_mass_flux!(u, bc::ROBC, ::Val{:west}, extra_flux_per_boundary) = u[1, :, :] = u[1, :, :] .- extra_flux_per_boundary
correct_left_boundary_mass_flux!(v, bc::ROBC, ::Val{:south}, extra_flux_per_boundary) = v[:, 1, :] = v[:, 1, :] .- extra_flux_per_boundary
correct_left_boundary_mass_flux!(w, bc::ROBC, ::Val{:bottom}, extra_flux_per_boundary) = w[:, :, 1] = w[:, :, 1] .- extra_flux_per_boundary
correct_left_boundary_mass_flux!(u, bc, side, extra_flux_per_boundary) = nothing

correct_right_boundary_mass_flux!(u, bc::ROBC, ::Val{:east}, extra_flux_per_boundary) = u[u.grid.Nx + 1, :, :] = u[u.grid.Nx + 1, :, :] .+ extra_flux_per_boundary
correct_right_boundary_mass_flux!(v, bc::ROBC, ::Val{:north}, extra_flux_per_boundary) = v[:, v.grid.Ny + 1, :] = v[:, v.grid.Ny + 1, :] .+ extra_flux_per_boundary
correct_right_boundary_mass_flux!(w, bc::ROBC, ::Val{:top}, extra_flux_per_boundary) = w[:, :, w.grid.Nz + 1] = w[:, :, w.grid.Nz + 1] .+ extra_flux_per_boundary
correct_right_boundary_mass_flux!(u, bc, side, extra_flux_per_boundary) = nothing

"""
enforce_open_boundary_mass_conservation!(model::NonhydrostaticModel)

Correct boundary mass fluxes for perturbation advection boundary conditions to ensure
zero net mass flux through each boundary.
"""
function enforce_open_boundary_mass_conservation!(model)
    u, v, w = model.velocities
    grid = model.grid

    total_mass_inflow = open_boundary_mass_inflow(model)

    # Calculate flux correction per boundary
    extra_flux_per_boundary = total_mass_inflow / (length(model.boundary_mass_fluxes.left_ROBCs) + length(model.boundary_mass_fluxes.right_ROBCs))

    # Subtract extra flux from left boundaries to reduce inflow
    correct_left_boundary_mass_flux!(u, u.boundary_conditions.west, Val(:west), extra_flux_per_boundary)
    correct_left_boundary_mass_flux!(v, v.boundary_conditions.south, Val(:south), extra_flux_per_boundary)
    correct_left_boundary_mass_flux!(w, w.boundary_conditions.bottom, Val(:bottom), extra_flux_per_boundary)

    correct_right_boundary_mass_flux!(u, u.boundary_conditions.east, Val(:east), extra_flux_per_boundary)
    correct_right_boundary_mass_flux!(v, v.boundary_conditions.north, Val(:north), extra_flux_per_boundary)
    correct_right_boundary_mass_flux!(w, w.boundary_conditions.top, Val(:top), extra_flux_per_boundary)
end
