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


initialize_boundary_mass_flux(u, bc::OBC, ::Val{:west})   = (; west = west_integral(u), west_area = get_west_area(u.grid))
initialize_boundary_mass_flux(u, bc::OBC, ::Val{:east})   = (; east = east_integral(u), east_area = get_east_area(u.grid))
initialize_boundary_mass_flux(v, bc::OBC, ::Val{:south})  = (; south = south_integral(v), south_area = get_south_area(v.grid))
initialize_boundary_mass_flux(v, bc::OBC, ::Val{:north})  = (; north = north_integral(v), north_area = get_north_area(v.grid))
initialize_boundary_mass_flux(w, bc::OBC, ::Val{:bottom}) = (; bottom = bottom_integral(w), bottom_area = get_bottom_area(w.grid))
initialize_boundary_mass_flux(w, bc::OBC, ::Val{:top})    = (; top = top_integral(w), top_area = get_top_area(w.grid))

initialize_boundary_mass_flux(u, bc::ZIOBC, ::Val{:west})   = NamedTuple()
initialize_boundary_mass_flux(u, bc::ZIOBC, ::Val{:east})   = NamedTuple()
initialize_boundary_mass_flux(v, bc::ZIOBC, ::Val{:south})  = NamedTuple()
initialize_boundary_mass_flux(v, bc::ZIOBC, ::Val{:north})  = NamedTuple()
initialize_boundary_mass_flux(w, bc::ZIOBC, ::Val{:bottom}) = NamedTuple()
initialize_boundary_mass_flux(w, bc::ZIOBC, ::Val{:top})    = NamedTuple()

initialize_boundary_mass_flux(u, bc::FIOBC, ::Val{:west})   = (; west = bc.condition * get_west_area(u.grid), west_area = get_west_area(u.grid))
initialize_boundary_mass_flux(u, bc::FIOBC, ::Val{:east})   = (; east = bc.condition * get_east_area(u.grid), east_area = get_east_area(u.grid))
initialize_boundary_mass_flux(v, bc::FIOBC, ::Val{:south})  = (; south = bc.condition * get_south_area(v.grid), south_area = get_south_area(v.grid))
initialize_boundary_mass_flux(v, bc::FIOBC, ::Val{:north})  = (; north = bc.condition * get_north_area(v.grid), north_area = get_north_area(v.grid))
initialize_boundary_mass_flux(w, bc::FIOBC, ::Val{:bottom}) = (; bottom = bc.condition * get_bottom_area(w.grid), bottom_area = get_bottom_area(w.grid))
initialize_boundary_mass_flux(w, bc::FIOBC, ::Val{:top})    = (; top = bc.condition * get_top_area(w.grid), top_area = get_top_area(w.grid))

initialize_boundary_mass_flux(velocity, ::Nothing, side) = NamedTuple()
initialize_boundary_mass_flux(velocity, bc, side) = NamedTuple()

"""
    initialize_boundary_mass_fluxes(velocities::NamedTuple)

Initialize boundary mass fluxes for boundaries with OpenBoundaryConditions,
returning a NamedTuple of boundary fluxes.
"""
function initialize_boundary_mass_fluxes(velocities::NamedTuple)

    u, v, w = velocities
    u_bcs = velocities.u.boundary_conditions
    v_bcs = velocities.v.boundary_conditions
    w_bcs = velocities.w.boundary_conditions

    boundary_fluxes = NamedTuple()
    right_matching_scheme_boundaries = Symbol[]
    left_matching_scheme_boundaries = Symbol[]
    total_area_matching_scheme_boundaries = zero(eltype(u))

    # Check west boundary (u velocity)
    west_flux_and_area = initialize_boundary_mass_flux(u, u_bcs.west, Val(:west))
    boundary_fluxes = merge(boundary_fluxes, west_flux_and_area)
    if u_bcs.west isa ROBC
        push!(left_matching_scheme_boundaries, :west)
        total_area_matching_scheme_boundaries += boundary_fluxes.west_area
    end

    # Check east boundary (u velocity)
    east_flux_and_area = initialize_boundary_mass_flux(u, u_bcs.east, Val(:east))
    boundary_fluxes = merge(boundary_fluxes, east_flux_and_area)
    if u_bcs.east isa ROBC
        push!(right_matching_scheme_boundaries, :east)
        total_area_matching_scheme_boundaries += boundary_fluxes.east_area
    end

    # Check south boundary (v velocity)
    south_flux_and_area = initialize_boundary_mass_flux(v, v_bcs.south, Val(:south))
    boundary_fluxes = merge(boundary_fluxes, south_flux_and_area)
    if v_bcs.south isa ROBC
        push!(left_matching_scheme_boundaries, :south)
        total_area_matching_scheme_boundaries += boundary_fluxes.south_area
    end

    # Check north boundary (v velocity)
    north_flux_and_area = initialize_boundary_mass_flux(v, v_bcs.north, Val(:north))
    boundary_fluxes = merge(boundary_fluxes, north_flux_and_area)
    if v_bcs.north isa ROBC
        push!(right_matching_scheme_boundaries, :north)
        total_area_matching_scheme_boundaries += boundary_fluxes.north_area
    end

    # Check bottom boundary (w velocity)
    bottom_flux_and_area = initialize_boundary_mass_flux(w, w_bcs.bottom, Val(:bottom))
    boundary_fluxes = merge(boundary_fluxes, bottom_flux_and_area)
    if w_bcs.bottom isa ROBC
        push!(left_matching_scheme_boundaries, :bottom)
        total_area_matching_scheme_boundaries += boundary_fluxes.bottom_area
    end

    # Check top boundary (w velocity)
    top_flux_and_area = initialize_boundary_mass_flux(w, w_bcs.top, Val(:top))
    boundary_fluxes = merge(boundary_fluxes, top_flux_and_area)
    if w_bcs.top isa ROBC
        push!(right_matching_scheme_boundaries, :top)
        total_area_matching_scheme_boundaries += boundary_fluxes.top_area
    end

    boundary_fluxes = merge(boundary_fluxes, (; left_matching_scheme_boundaries,
                                                right_matching_scheme_boundaries,
                                                total_area_matching_scheme_boundaries))
    return boundary_fluxes
end

update_open_boundary_mass_fluxes!(model) = map(compute!, model.boundary_mass_fluxes)

open_boundary_mass_flux(model, bc::OBC, ::Val{:west}, u) = @allowscalar model.boundary_mass_fluxes.west[]
open_boundary_mass_flux(model, bc::OBC, ::Val{:east}, u) = @allowscalar model.boundary_mass_fluxes.east[]
open_boundary_mass_flux(model, bc::OBC, ::Val{:south}, v) = @allowscalar model.boundary_mass_fluxes.south[]
open_boundary_mass_flux(model, bc::OBC, ::Val{:north}, v) = @allowscalar model.boundary_mass_fluxes.north[]
open_boundary_mass_flux(model, bc::OBC, ::Val{:bottom}, w) = @allowscalar model.boundary_mass_fluxes.bottom[] 
open_boundary_mass_flux(model, bc::OBC, ::Val{:top}, w) = @allowscalar model.boundary_mass_fluxes.top[]

open_boundary_mass_flux(model, bc::ZIOBC, ::Val{:west}, u) = zero(model.grid)
open_boundary_mass_flux(model, bc::ZIOBC, ::Val{:east}, u) = zero(model.grid)
open_boundary_mass_flux(model, bc::ZIOBC, ::Val{:south}, v) = zero(model.grid)
open_boundary_mass_flux(model, bc::ZIOBC, ::Val{:north}, v) = zero(model.grid)
open_boundary_mass_flux(model, bc::ZIOBC, ::Val{:bottom}, w) = zero(model.grid)
open_boundary_mass_flux(model, bc::ZIOBC, ::Val{:top}, w) = zero(model.grid)

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

correct_left_boundary_mass_flux!(u, bc::ROBC, ::Val{:west},   average_mass_inflow) = interior(u, 1, :, :) .-= average_mass_inflow
correct_left_boundary_mass_flux!(v, bc::ROBC, ::Val{:south},  average_mass_inflow) = interior(v, :, 1, :) .-= average_mass_inflow
correct_left_boundary_mass_flux!(w, bc::ROBC, ::Val{:bottom}, average_mass_inflow) = interior(w, :, :, 1) .-= average_mass_inflow
correct_left_boundary_mass_flux!(velocity, bc, side, average_mass_inflow) = nothing

correct_right_boundary_mass_flux!(u, bc::ROBC, ::Val{:east},  average_mass_inflow) = interior(u, u.grid.Nx + 1, :, :) .+= average_mass_inflow
correct_right_boundary_mass_flux!(v, bc::ROBC, ::Val{:north}, average_mass_inflow) = interior(v, :, v.grid.Ny + 1, :) .+= average_mass_inflow
correct_right_boundary_mass_flux!(w, bc::ROBC, ::Val{:top},   average_mass_inflow) = interior(w, :, :, w.grid.Nz + 1) .+= average_mass_inflow
correct_right_boundary_mass_flux!(velocity, bc, side, average_mass_inflow) = nothing

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
    average_mass_inflow = total_mass_inflow / model.boundary_mass_fluxes.total_area_matching_scheme_boundaries

    # Subtract extra flux from left boundaries to reduce inflow
    correct_left_boundary_mass_flux!(u, u.boundary_conditions.west, Val(:west), average_mass_inflow)
    correct_left_boundary_mass_flux!(v, v.boundary_conditions.south, Val(:south), average_mass_inflow)
    correct_left_boundary_mass_flux!(w, w.boundary_conditions.bottom, Val(:bottom), average_mass_inflow)

    correct_right_boundary_mass_flux!(u, u.boundary_conditions.east, Val(:east), average_mass_inflow)
    correct_right_boundary_mass_flux!(v, v.boundary_conditions.north, Val(:north), average_mass_inflow)
    correct_right_boundary_mass_flux!(w, w.boundary_conditions.top, Val(:top), average_mass_inflow)
end
