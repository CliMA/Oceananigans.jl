using Oceananigans
using Oceananigans.Units
using Oceananigans.Grids: topology

#####
##### Utilities
#####

using Oceananigans.Models: AbstractModel

import Oceananigans.TimeSteppers: time_step!, update_state!
import Oceananigans: fields

struct CoupledAtmosphereOceanModel{O, A, C, P} <: AbstractModel{Nothing}
    atmos :: A
    ocean :: O
    clock :: C
    air_sea_flux_parameters :: P
end

function CoupledAtmosphereOceanModel(atmos, ocean; air_sea_flux_parameters=(; cᴰ=2e-3, ρ_atmos=1, ρ_ocean=1024))
    clock = atmos.clock
    return CoupledAtmosphereOceanModel(atmos, ocean, clock, air_sea_flux_parameters)
end

fields(model::CoupledAtmosphereOceanModel) = fields(model.atmos) # convenience hack for now

function update_state!(coupled_model::CoupledAtmosphereOceanModel, update_atmos_ocean_state=true)
    atmos_model = coupled_model.atmos
    ocean_model = coupled_model.ocean

    if update_atmos_ocean_state
        update_state!(ocean_model)
        update_state!(atmos_model)
    end

    uo, vo, wo = ocean_model.velocities
    ua, va, wa = atmos_model.velocities
    atmos_grid = atmos_model.grid
    atmos_surface_flux_u = ua.boundary_conditions.bottom.condition
    atmos_surface_flux_v = va.boundary_conditions.bottom.condition
    ocean_surface_flux_u = uo.boundary_conditions.top.condition
    ocean_surface_flux_v = vo.boundary_conditions.top.condition

    # Much room for improvement here...
    cᴰ = coupled_model.air_sea_flux_parameters.cᴰ
    ρ_atmos = coupled_model.air_sea_flux_parameters.ρ_atmos
    ρ_ocean = coupled_model.air_sea_flux_parameters.ρ_ocean

    # Use broadcasting to compute bulk formula for surface wind stress
    topo = topology(atmos_grid)
    Nx, Ny, Nz = size(atmos_grid)
    Hx, Hy, Hz = atmos_grid.Hx, atmos_grid.Hy, atmos_grid.Hz
    ii = Hx+1:Hx+Nx
    jj = topo[2]() isa Flat ? 1 : Hy+1:Hy+Ny # hack because 2D can be fun
    k = atmos_grid.Hz+1 # surface atmospheric velocity
    ua₁ = view(parent(ua), ii, jj, k:k)
    va₁ = view(parent(va), ii, jj, k:k)

    @. atmos_surface_flux_u = - cᴰ * ua₁ * sqrt(ua₁^2 + va₁^2)
    @. atmos_surface_flux_v = - cᴰ * va₁ * sqrt(ua₁^2 + va₁^2)

    @. ocean_surface_flux_u = ρ_atmos / ρ_ocean * atmos_surface_flux_u
    @. ocean_surface_flux_v = ρ_atmos / ρ_ocean * atmos_surface_flux_v

    return nothing
end

function time_step!(coupled_model::CoupledAtmosphereOceanModel, Δt; euler=false)
    time_step!(coupled_model.ocean, Δt; euler)
    time_step!(coupled_model.atmos, Δt; euler)
    update_state!(coupled_model, false)
    return nothing
end

