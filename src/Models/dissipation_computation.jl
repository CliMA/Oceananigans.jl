using Oceananigans.Grids: architecture
using Oceananigans.Utils
using Oceananigans.Fields: Field, VelocityFields
using Oceananigans.Operators
using Oceananigans.BoundaryConditions
using Oceananigans.Advection: _advective_tracer_flux_x, _advective_tracer_flux_y, _advective_tracer_flux_z
using Oceananigans.Advection: horizontal_advection_U, horizontal_advection_V
using Oceananigans.Models.HydrostaticFreeSurfaceModels: vertical_scaling, previous_vertical_scaling

import Oceananigans.Utils: KernelParameters

const c = Center()
const f = Face()

struct DissipationComputation{P, A, S}
    production :: P
    advective_fluxes :: A
    previous_state :: S
end

function fluxes_field(grid)
    x = XFaceField(grid)
    y = YFaceField(grid)
    z = ZFaceField(grid)

    return (; x, y, z)
end

function DissipationComputation(model; tracers = tracer_names(model.tracers))
    
    if !(model.timestepper isa QuasiAdamsBashforth2Timestepper)
        throw(ArgumentError("DissipationComputation requires a QuasiAdamsBashforth2Timestepper"))
    end

    grid = model.grid
    P    = NamedTuple{tracers}(fluxes_field(grid) for tracer in tracers)
    Fⁿ   = NamedTuple{tracers}(fluxes_field(grid) for tracer in tracers)
    Fⁿ⁻¹ = NamedTuple{tracers}(fluxes_field(grid) for tracer in tracers)

    Uⁿ⁻¹ = VelocityFields(grid)
    Uⁿ   = VelocityFields(grid)
        
    cⁿ⁻¹ =  NamedTuple{tracers}(CenterField(grid) for tracer in tracers)

    previous_state = merge(cⁿ⁻¹, (; Uⁿ⁻¹, Uⁿ))
    advective_fluxes = (; Fⁿ, Fⁿ⁻¹)

    return DissipationComputation(P, advective_fluxes, previous_state)
end

# Function to call in a callback
function (dc::DissipationComputation)(simulation)
    # We first assemble values for Pⁿ⁻¹
    assemble_P_values!(simulation, dc)

    # Then we update the fluxes to be used in the next time step
    update_fluxes!(simulation, dc)

    return nothing
end

@inline function KernelParameters(f::Field)
    sz = size(f.data)
    of = f.data.offsets
    return KernelParameters(sz, of)
end

function update_fluxes!(simulation, dissipation_computation)
    model = simulation.model

    grid = model.grid
    arch = architecture(grid)

    params = KernelParameters(model.tracers[1])
    
    Uⁿ   = dissipation_computation.previous_state.Uⁿ
    Uⁿ⁻¹ = dissipation_computation.previous_state.Uⁿ⁻¹ 

    U = model.velocities

    launch!(architecture(grid), grid, params, _update_transport!, Uⁿ, Uⁿ⁻¹, grid, U)

    for tracer_name in keys(dissipation_computation.production)
        c    = model.tracers[tracer_name]
        cⁿ⁻¹ = dissipation_computation.previous_state.tracers[tracer_name]
        Fⁿ   = dissipation_computation.advective_fluxes.Fⁿ[tracer_fluxes]
        Fⁿ⁻¹ = dissipation_computation.advective_fluxes.Fⁿ⁻¹[tracer_name]
        A    = model.advection[tracer_name]

        launch!(arch, grid, params, _update_fluxes!, Fⁿ, Fⁿ⁻¹, cⁿ⁻¹, grid, A, U, c)
    end

    return nothing
end

@kenrel function _update_transport!(Uⁿ, Uⁿ⁻¹, grid, U)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        Uⁿ⁻¹.u[i, j, k] = Uⁿ.u[i, j, k]
        Uⁿ⁻¹.v[i, j, k] = Vⁿ.v[i, j, k]
        Uⁿ⁻¹.w[i, j, k] = Wⁿ.w[i, j, k]
          Uⁿ.u[i, j, k] = U.u[i, j, k] * Axᶠᶜᶜ(i, j, k, grid) 
          Uⁿ.v[i, j, k] = U.v[i, j, k] * Ayᶜᶠᶜ(i, j, k, grid) 
          Uⁿ.w[i, j, k] = U.w[i, j, k] * Azᶜᶜᶠ(i, j, k, grid) 
    end
end

@kernel function _update_fluxes!( Fⁿ, Fⁿ⁻¹, cⁿ⁻¹, grid, advection, U, c)
    i, j, k = @index(Global, NTuple)
    u, v, w = U

    @inbounds begin
        # Save previous advective fluxes
        Fⁿ⁻¹.x[i, j, k] = Fⁿ.x[i, j, k]
        Fⁿ⁻¹.y[i, j, k] = Fⁿ.y[i, j, k]
        Fⁿ⁻¹.z[i, j, k] = Fⁿ.z[i, j, k]
        
        cⁿ⁻¹[i, j, k] = c[i, j, k]

        # Calculate new advective fluxes
        Fⁿ.x[i, j, k] = _advective_tracer_flux_x(i, j, k, grid, advection, u, c) 
        Fⁿ.y[i, j, k] = _advective_tracer_flux_y(i, j, k, grid, advection, v, c) 
        Fⁿ.z[i, j, k] = _advective_tracer_flux_z(i, j, k, grid, advection, w, c) 
    end
end

function assemble_P_values!(simulation, dissipation_computation)
    model = simulation.model
    grid = model.grid
    arch = architecture(grid)

    χ = simulation.model.timestepper.χ

    # General velocities
    Uⁿ   = dissipation_computation.previous_state.Uⁿ
    Uⁿ⁻¹ = dissipation_computation.previous_state.Uⁿ⁻¹

    for tracer_names in keys(dissipation_computation.production)
        c = model.tracers[tracer_name]
        cⁿ⁻¹ = dissipation_computation.previous_state.tracers[tracer_name]
        P = dissipation_computation.production[tracer_name]
        Fⁿ = dissipation_computation.advective_fluxes.Fⁿ[tracer_name]
        Fⁿ⁻¹ = dissipation_computation.advective_fluxes.Fⁿ⁻¹[tracer_name]

        launch!(arch, grid, :xyz, _compute_dissipation!, 
                P, grid, χ, Fⁿ, Fⁿ⁻¹, Uⁿ, Uⁿ⁻¹, c, cⁿ⁻¹)

    return nothing
end

@inline c★(i, j, k, grid, cⁿ, cⁿ⁻¹) = @inbounds (cⁿ[i, j, k] + cⁿ⁻¹[i, j, k]) / 2
@inline c²(i, j, k, grid, c₁, c₂)   = @inbounds (c₁[i, j, k] * c₂[i, j, k])

@kernel function _compute_dissipation!(P,
                                       grid, χ, 
                                       Fⁿ, Fⁿ⁻¹, 
                                       Uⁿ, Uⁿ⁻¹, 
                                       cⁿ, cⁿ⁻¹)
    
    i, j, k = @index(Global, NTuple)


    δˣc★ = δxᶠᶜᶜ(i, j, k, grid, c★, cⁿ, cⁿ⁻¹)
    δˣc² = δxᶠᶜᶜ(i, j, k, grid, c², cⁿ, cⁿ⁻¹)

    δʸc★ = δyᶜᶠᶜ(i, j, k, grid, c★, cⁿ, cⁿ⁻¹)
    δʸc² = δyᶜᶠᶜ(i, j, k, grid, c², cⁿ, cⁿ⁻¹)

    δᶻc★ = δzᶜᶜᶠ(i, j, k, grid, c★, cⁿ, cⁿ⁻¹)
    δᶻc² = δzᶜᶜᶠ(i, j, k, grid, c², cⁿ, cⁿ⁻¹)

    @inbounds P.x[i, j, k] = compute_dissipation(i, j, k, grid, Uⁿ.u, Uⁿ⁻¹.u, χ, Fⁿ.x, Fⁿ⁻¹.x, δˣc★, δˣc²)
    @inbounds P.y[i, j, k] = compute_dissipation(i, j, k, grid, Uⁿ.v, Uⁿ⁻¹.v, χ, Fⁿ.y, Fⁿ⁻¹.y, δʸc★, δʸc²)
    @inbounds P.z[i, j, k] = compute_dissipation(i, j, k, grid, Uⁿ.w, Uⁿ⁻¹.w, χ, Fⁿ.z, Fⁿ⁻¹.z, δᶻc★, δᶻc²)
end

@inline function compute_dissipation(i, j, k, grid, Uⁿ⁻¹, Uⁿ⁻², χ, fⁿ⁻¹, fⁿ⁻², δc★, δc²)

    C₁ = convert(eltype(grid), 1.5 + χ)
    C₂ = convert(eltype(grid), 0.5 + χ)

    @inbounds begin
        𝒰ⁿ⁻¹ = C₁ * Uⁿ⁻¹[i, j, k] 
        𝒰ⁿ⁻² = C₂ * Uⁿ⁻²[i, j, k] 
        Fⁿ⁻¹ = C₁ * fⁿ⁻¹[i, j, k] 
        Fⁿ⁻² = C₂ * fⁿ⁻²[i, j, k] 
        𝒜 = Fⁿ⁻¹ - Fⁿ⁻²
        𝒟 = 𝒰ⁿ⁻¹ - 𝒰ⁿ⁻²
    end

    return 2 * δc★ * 𝒜 - δc² * 𝒟
end
