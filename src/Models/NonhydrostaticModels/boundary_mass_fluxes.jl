using Oceananigans.BoundaryConditions: BoundaryCondition, Open
using Oceananigans.AbstractOperations: Integral, Ax, Ay, Az, grid_metric_operation
using Oceananigans.Fields: Field, interior
using GPUArraysCore: @allowscalar

#####
##### Boundary location types for dispatch
#####

struct WestBoundary end
struct EastBoundary end
struct SouthBoundary end
struct NorthBoundary end
struct BottomBoundary end
struct TopBoundary end

const west   = WestBoundary()
const east   = EastBoundary()
const south  = SouthBoundary()
const north  = NorthBoundary()
const bottom = BottomBoundary()
const top    = TopBoundary()

# Boundary properties for dispatch
normal_direction(::WestBoundary)   = +1
normal_direction(::EastBoundary)   = -1
normal_direction(::SouthBoundary)  = +1
normal_direction(::NorthBoundary)  = -1
normal_direction(::BottomBoundary) = +1
normal_direction(::TopBoundary)    = -1

#####
##### Boundary condition type aliases
#####

const OBC   = BoundaryCondition{<:Open}                          # OpenBoundaryCondition
const IOBC  = BoundaryCondition{<:Open{<:Nothing}}               # "Imposed-velocity" OpenBoundaryCondition (with no scheme)
const FIOBC = BoundaryCondition{<:Open{<:Nothing}, <:Number}     # "Fixed-imposed-velocity" OpenBoundaryCondition
const ZIOBC = BoundaryCondition{<:Open{<:Nothing}, <:Nothing}    # "Zero-imposed-velocity" OpenBoundaryCondition (no-inflow)

#####
##### BoundaryMassFlux and BoundaryMassFluxes structs
#####

"""
    BoundaryMassFlux{F, A}

Holds the mass flux field (or scalar value) and area for a single boundary.
"""
struct BoundaryMassFlux{BC, F, A, BND}
    boundary_condition :: BC
    flux :: F
    area :: A
    boundary :: BND
end

const RadiatingBoundaryMassFlux = BoundaryMassFlux{<:OBC}
const NonRadiatingBoundaryMassFlux = BoundaryMassFlux{<:IOBC}

"""
    BoundaryMassFluxes

Container for mass flux information at all open boundaries.
Each boundary field (`west`, `east`, etc.) is either a `BoundaryMassFlux` or `nothing`.
The `radiating_boundary_area` is the sum of areas of radiating boundaries (with advection schemes).
"""
struct BoundaryMassFluxes{W, E, S, N, B, T, A}
    west   :: W
    east   :: E
    south  :: S
    north  :: N
    bottom :: B
    top    :: T
    radiating_boundary_area :: A
end

#####
##### Boundary area computation
#####

function boundary_area(::WestBoundary, grid)
    dA = grid_metric_operation((Face, Center, Center), Ax, grid)
    ∫dA = sum(dA, dims=(2, 3))
    return @allowscalar ∫dA[1, 1, 1]
end

function boundary_area(::EastBoundary, grid)
    dA = grid_metric_operation((Face, Center, Center), Ax, grid)
    ∫dA = sum(dA, dims=(2, 3))
    return @allowscalar ∫dA[grid.Nx+1, 1, 1]
end

function boundary_area(::SouthBoundary, grid)
    dA = grid_metric_operation((Center, Face, Center), Ay, grid)
    ∫dA = sum(dA, dims=(1, 3))
    return @allowscalar ∫dA[1, 1, 1]
end

function boundary_area(::NorthBoundary, grid)
    dA = grid_metric_operation((Center, Face, Center), Ay, grid)
    ∫dA = sum(dA, dims=(1, 3))
    return @allowscalar ∫dA[1, grid.Ny+1, 1]
end

function boundary_area(::BottomBoundary, grid)
    dA = grid_metric_operation((Center, Center, Face), Az, grid)
    ∫dA = sum(dA, dims=(1, 2))
    return @allowscalar ∫dA[1, 1, 1]
end

function boundary_area(::TopBoundary, grid)
    dA = grid_metric_operation((Center, Center, Face), Az, grid)
    ∫dA = sum(dA, dims=(1, 2))
    return @allowscalar ∫dA[1, 1, grid.Nz+1]
end

#####
##### Boundary mass flux field computation
#####

boundary_mass_flux_field(::WestBoundary, u)   = Field(Integral(view(u, 1, :, :), dims=(2, 3)))
boundary_mass_flux_field(::EastBoundary, u)   = Field(Integral(view(u, u.grid.Nx + 1, :, :), dims=(2, 3)))
boundary_mass_flux_field(::SouthBoundary, v)  = Field(Integral(view(v, :, 1, :), dims=(1, 3)))
boundary_mass_flux_field(::NorthBoundary, v)  = Field(Integral(view(v, :, v.grid.Ny + 1, :), dims=(1, 3)))
boundary_mass_flux_field(::BottomBoundary, w) = Field(Integral(view(w, :, :, 1), dims=(1, 2)))
boundary_mass_flux_field(::TopBoundary, w)    = Field(Integral(view(w, :, :, w.grid.Nz + 1), dims=(1, 2)))

#####
##### Initialize individual boundary mass flux
#####

radiating_boundary_condition(::IOBC)    = false
radiating_boundary_condition(::OBC)     = true
radiating_boundary_condition(bc)        = false

# OBC with advection scheme: create Field for dynamic flux computation
function initialize_boundary_mass_flux(velocity, bc::OBC, boundary)
    A = boundary_area(boundary, velocity.grid)
    flux = boundary_mass_flux_field(boundary, velocity)
    return BoundaryMassFlux(bc, flux, A, boundary)
end

# FIOBC (fixed imposed velocity): flux is constant = velocity × area
function initialize_boundary_mass_flux(velocity, bc::FIOBC, boundary)
    A = boundary_area(boundary, velocity.grid)
    flux = bc.condition * A
    return BoundaryMassFlux(bc, flux, A, boundary)
end

# ZIOBC (zero imposed velocity): no flux needed
initialize_boundary_mass_flux(velocity, bc::ZIOBC, boundary) = nothing
initialize_boundary_mass_flux(velocity, bc, boundary) = nothing

#####
##### Initialize all boundary mass fluxes
#####

"""
    initialize_boundary_mass_fluxes(velocities::NamedTuple)

Initialize boundary mass fluxes for boundaries with OpenBoundaryConditions,
returning a `BoundaryMassFluxes` object.
"""
function initialize_boundary_mass_fluxes(velocities::NamedTuple)
    u, v, w = velocities
    u_bcs = u.boundary_conditions
    v_bcs = v.boundary_conditions
    w_bcs = w.boundary_conditions

    # Initialize each boundary
    west_bmf   = initialize_boundary_mass_flux(u, u_bcs.west,   west)
    east_bmf   = initialize_boundary_mass_flux(u, u_bcs.east,   east)
    south_bmf  = initialize_boundary_mass_flux(v, v_bcs.south,  south)
    north_bmf  = initialize_boundary_mass_flux(v, v_bcs.north,  north)
    bottom_bmf = initialize_boundary_mass_flux(w, w_bcs.bottom, bottom)
    top_bmf    = initialize_boundary_mass_flux(w, w_bcs.top,    top)

    # Return nothing if no open boundaries
    all_bmfs = (west_bmf, east_bmf, south_bmf, north_bmf, bottom_bmf, top_bmf)
    all(isnothing, all_bmfs) && return nothing

    # Compute total radiating boundary area
    radiating_boundary_area = zero(eltype(u))
    for bmf in all_bmfs
        radiating_boundary_area += radiating_area(bmf)
    end

    return BoundaryMassFluxes(all_bmfs..., radiating_boundary_area)
end

radiating_area(bmf::RadiatingBoundaryMassFlux) = bmf.area
radiating_area(::NonRadiatingBoundaryMassFlux) = 0
radiating_area(::Nothing) = 0

#####
##### Update and access boundary mass fluxes
#####

compute_mass_flux!(bmf::BoundaryMassFlux{<:Any, <:Field}) = compute!(bmf.flux)
compute_mass_flux!(bmf) = nothing

function update_open_boundary_mass_fluxes!(model)
    bmfs = model.boundary_mass_fluxes
    isnothing(bmfs) && return nothing

    for bmf in (bmfs.west, bmfs.east, bmfs.south, bmfs.north, bmfs.bottom, bmfs.top)
        compute_mass_flux!(bmf)
    end
    return nothing
end

# Get the scalar mass flux value from a BoundaryMassFlux
mass_flux(bmf::BoundaryMassFlux{<:Any, <:Field}) = @allowscalar bmf.flux[]
mass_flux(bmf::BoundaryMassFlux{<:Any, <:Number}) = bmf.flux
mass_flux(::Nothing) = 0

signed_mass_flux(bmf::BoundaryMassFlux) = normal_direction(bmf.boundary) * mass_flux(bmf)
signed_mass_flux(::Nothing) = 0

# Compute the total mass flux through all open boundaries.
# Positive values indicate net inflow (mass convergence),
# negative values indicate net outflow (mass divergence).
function total_open_boundary_mass_flux(model)
    update_open_boundary_mass_fluxes!(model)
    bmfs = model.boundary_mass_fluxes
    total_flux = zero(model.grid)
    for bmf in (bmfs.west, bmfs.east, bmfs.south, bmfs.north, bmfs.bottom, bmfs.top)
        total_flux += signed_mass_flux(bmf)
    end
    return total_flux
end

#####
##### Update radiating boundary fluxes for conservation
#####

# Update radiating (OBC with advection scheme) boundaries
update_radiating_boundary_flux!(velocity, ::OBC, ::WestBoundary,   Δu) = interior(velocity, 1, :, :) .-= Δu
update_radiating_boundary_flux!(velocity, ::OBC, ::EastBoundary,   Δu) = interior(velocity, velocity.grid.Nx + 1, :, :) .+= Δu
update_radiating_boundary_flux!(velocity, ::OBC, ::SouthBoundary,  Δu) = interior(velocity, :, 1, :) .-= Δu
update_radiating_boundary_flux!(velocity, ::OBC, ::NorthBoundary,  Δu) = interior(velocity, :, velocity.grid.Ny + 1, :) .+= Δu
update_radiating_boundary_flux!(velocity, ::OBC, ::BottomBoundary, Δu) = interior(velocity, :, :, 1) .-= Δu
update_radiating_boundary_flux!(velocity, ::OBC, ::TopBoundary,    Δu) = interior(velocity, :, :, velocity.grid.Nz + 1) .+= Δu

# No update for imposed velocity boundaries (IOBC is a subtype of OBC, so we need specific methods)
update_radiating_boundary_flux!(velocity, ::IOBC, ::WestBoundary,   Δu) = nothing
update_radiating_boundary_flux!(velocity, ::IOBC, ::EastBoundary,   Δu) = nothing
update_radiating_boundary_flux!(velocity, ::IOBC, ::SouthBoundary,  Δu) = nothing
update_radiating_boundary_flux!(velocity, ::IOBC, ::NorthBoundary,  Δu) = nothing
update_radiating_boundary_flux!(velocity, ::IOBC, ::BottomBoundary, Δu) = nothing
update_radiating_boundary_flux!(velocity, ::IOBC, ::TopBoundary,    Δu) = nothing

# Fallback for non-open boundaries
update_radiating_boundary_flux!(velocity, bc, boundary, Δu) = nothing

enforce_open_boundary_mass_conservation!(model, ::Nothing) = nothing

"""
    enforce_open_boundary_mass_conservation!(model, boundary_mass_fluxes)

Update velocities at radiating open boundaries to ensure zero net mass flux through the domain.
"""
function enforce_open_boundary_mass_conservation!(model, boundary_mass_fluxes)
    u, v, w = model.velocities

    ∮udA = total_open_boundary_mass_flux(model)
    A = boundary_mass_fluxes.radiating_boundary_area
    Δu = ∮udA / A

    # Apply corrections to all boundaries
    update_radiating_boundary_flux!(u, u.boundary_conditions.west,   west,   Δu)
    update_radiating_boundary_flux!(u, u.boundary_conditions.east,   east,   Δu)
    update_radiating_boundary_flux!(v, v.boundary_conditions.south,  south,  Δu)
    update_radiating_boundary_flux!(v, v.boundary_conditions.north,  north,  Δu)
    update_radiating_boundary_flux!(w, w.boundary_conditions.bottom, bottom, Δu)
    update_radiating_boundary_flux!(w, w.boundary_conditions.top,    top,    Δu)
end
