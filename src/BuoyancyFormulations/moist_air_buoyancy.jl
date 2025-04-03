struct MoistBuoyancy{FT} <: AbstractBuoyancyFormulation{Nothing}
    expansion_coefficient :: FT
    reference_potential_temperature :: FT
    gas_constant_ratio :: FT
end

function MoistBuoyancy(FT=Oceananigans.defaults.FloatType;
                       expansion_coefficient = 3.27e-2,
                       reference_potential_temperature = 0,
                       gas_constant_ratio = 1.61)

    return MoistBuoyancy{FT}(expansion_coefficient,
                             reference_potential_temperature,
                             gas_constant_ratio)
end

# const BuoyancyTracerFormulation = BuoyancyForce{<:BuoyancyTracer}

required_tracers(::MoistBuoyancy) = (:θ, :q)

@inline function buoyancy_perturbationᶜᶜᶜ(i, j, k, grid, mb::MoistBuoyancy, tracers)
    β = mb.expansion_coefficient
    θ₀ = mb.reference_potential_temperature
    ϵᵥ = mb.gas_constant_ratio
    δ = ϵᵥ - 1
    θ = @inbounds tracers.θ[i, j, k]
    q = @inbounds tracers.q[i, j, k]
    θᵥ = θ * (1 + δ * q)
    return β * (θᵥ - θ₀)
end

# @inline ∂x_b(i, j, k, grid, ::MoistBuoyancy, C) = ∂xᶠᶜᶜ(i, j, k, grid, C.b)
# @inline ∂y_b(i, j, k, grid, ::MoistBuoyancy, C) = ∂yᶜᶠᶜ(i, j, k, grid, C.b)
# @inline ∂z_b(i, j, k, grid, ::MoistBuoyancy, C) = ∂zᶜᶜᶠ(i, j, k, grid, C.b)

# @inline top_buoyancy_flux(i, j, grid, ::MoistBuoyancy, top_tracer_bcs, clock, fields) = getbc(top_tracer_bcs.b, i, j, grid, clock, fields)
# @inline bottom_buoyancy_flux(i, j, grid, ::MoistBuoyancy, bottom_tracer_bcs, clock, fields) = getbc(bottom_tracer_bcs.b, i, j, grid, clock, fields)
