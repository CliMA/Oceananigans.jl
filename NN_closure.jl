import Oceananigans.TurbulenceClosures:
        compute_diffusivities!,
        DiffusivityFields,
        viscosity, 
        diffusivity,
        diffusive_flux_x,
        diffusive_flux_y, 
        diffusive_flux_z,
        viscous_flux_ux,
        viscous_flux_uy,
        viscous_flux_uz,
        viscous_flux_vx,
        viscous_flux_vy,
        viscous_flux_vz,
        viscous_flux_wx,
        viscous_flux_wy,
        viscous_flux_wz

using Oceananigans.BuoyancyModels: ∂x_b, ∂y_b, ∂z_b 
using Oceananigans.Coriolis
using Oceananigans.Grids: φnode
using Lux, LuxCUDA
using JLD2
using ComponentArrays
using StaticArrays

using KernelAbstractions: @index, @kernel, @private

import Oceananigans.TurbulenceClosures: AbstractTurbulenceClosure, ExplicitTimeDiscretization

using Adapt 

include("./feature_scaling.jl")

@inline hack_sind(φ) = sin(φ * π / 180)

@inline fᶜᶜᵃ(i, j, k, grid, coriolis::HydrostaticSphericalCoriolis) = 2 * coriolis.rotation_rate * hack_sind(φnode(i, j, k, grid, Center(), Center(), Center()))
@inline fᶜᶜᵃ(i, j, k, grid, coriolis::FPlane)    = coriolis.f
@inline fᶜᶜᵃ(i, j, k, grid, coriolis::BetaPlane) = coriolis.f₀ + coriolis.β * ynode(i, j, k, grid, Center(), Center(), Center())

struct NN{M, P, S}
    model   :: M
    ps      :: P
    st      :: S
end

@inline (neural_network::NN)(input) = first(first(neural_network.model(input, neural_network.ps, neural_network.st)))
@inline tosarray(x::AbstractArray) = SArray{Tuple{size(x)...}}(x)

struct NNFluxClosure{A <: NN, S} <: AbstractTurbulenceClosure{ExplicitTimeDiscretization, 3}
    wT      :: A
    wS      :: A
    scaling :: S
end

Adapt.adapt_structure(to, nn :: NNFluxClosure) = 
    NNFluxClosure(Adapt.adapt(to, nn.wT), 
                  Adapt.adapt(to, nn.wS), 
                  Adapt.adapt(to, nn.scaling))

Adapt.adapt_structure(to, nn :: NN) = 
    NN(Adapt.adapt(to, nn.model), 
       Adapt.adapt(to, nn.ps), 
       Adapt.adapt(to, nn.st))

function NNFluxClosure(arch)
    dev = ifelse(arch == GPU(), gpu_device(), cpu_device())
    nn_model = Chain(Dense(11, 128, relu), Dense(128, 128, relu), Dense(128, 1))

    # scaling = jldopen("./NN_model2.jld2")["scaling"]
    scaling = (; ∂T∂z = ZeroMeanUnitVarianceScaling(-0.0006850967567052092, 0.019041912105983983),
                 ∂S∂z = ZeroMeanUnitVarianceScaling(-0.00042981832021978374, 0.0028927446724707905),
                 ∂ρ∂z = ZeroMeanUnitVarianceScaling(-0.0011311157767216616, 0.0008333035237211424),
                    f = ZeroMeanUnitVarianceScaling(-1.5e-5, 8.73212459828649e-5),
                   wb = ZeroMeanUnitVarianceScaling(6.539366623323223e-8, 1.827377562065243e-7),
                   wT = ZeroMeanUnitVarianceScaling(1.8169228278423015e-5, 0.00010721779595955453),
                   wS = ZeroMeanUnitVarianceScaling(-5.8185988680682135e-6, 1.7691239104281005e-5))

    # NNs = jldopen("./NN_model2.jld2")["NNs"]
    ps = jldopen("./NN_model2.jld2")["u"]
    sts = jldopen("./NN_model2.jld2")["sts"]

    ps_static = Lux.recursive_map(tosarray, ps)
    sts_static = Lux.recursive_map(tosarray, sts)

    wT_NN = NN(nn_model, ps.wT, sts.wT)
    wS_NN = NN(nn_model, ps.wS, sts.wS)

    return NNFluxClosure(wT_NN, wS_NN, scaling)
end

DiffusivityFields(grid, tracer_names, bcs, ::NNFluxClosure) = 
                (; wT = ZFaceField(grid),
                   wS = ZFaceField(grid))

function compute_diffusivities!(diffusivities, closure::NNFluxClosure, model; parameters = :xyz)
    arch = model.architecture
    grid = model.grid
    velocities = model.velocities
    tracers    = model.tracers
    buoyancy   = model.buoyancy
    coriolis   = model.coriolis    
    clock      = model.clock
    top_tracer_bcs = NamedTuple(c => tracers[c].boundary_conditions.top for c in propertynames(tracers))

    launch!(arch, grid, parameters,
            _compute_residual_fluxes!, diffusivities, grid, closure, tracers, velocities, buoyancy, coriolis, top_tracer_bcs, clock)

    return nothing
end

@kernel function _compute_residual_fluxes!(diffusivities, grid, closure, tracers, velocities, buoyancy, coriolis, top_tracer_bcs, clock)
    i, j, k = @index(Global, NTuple)

    # Find a way to extract the type FT
    nn_input = @private eltype(grid) 11

    scaling = closure.scaling

    nn_input[10] = Jᵇ = scaling.wb(top_buoyancy_flux(i, j, grid, buoyancy, top_tracer_bcs, clock, tracers))
    nn_input[11] = fᶜᶜ = scaling.f(fᶜᶜᵃ(i, j, k, grid, coriolis))

    nn_input[1] = ∂Tᵢ₋₁ = scaling.∂T∂z(∂zᶜᶜᶠ(i, j, k-1, grid, tracers.T))
    nn_input[2] = ∂Tᵢ = scaling.∂T∂z(∂zᶜᶜᶠ(i, j, k,   grid, tracers.T))
    nn_input[3] = ∂Tᵢ₊₁ = scaling.∂T∂z(∂zᶜᶜᶠ(i, j, k+1, grid, tracers.T))

    nn_input[4] = ∂Sᵢ₋₁ = scaling.∂S∂z(∂zᶜᶜᶠ(i, j, k-1, grid, tracers.S))
    nn_input[5] = ∂Sᵢ   = scaling.∂S∂z(∂zᶜᶜᶠ(i, j, k,   grid, tracers.S))
    nn_input[6] = ∂Sᵢ₊₁ = scaling.∂S∂z(∂zᶜᶜᶠ(i, j, k+1, grid, tracers.S))

    ρ₀ = buoyancy.model.equation_of_state.reference_density
    g  = buoyancy.model.gravitational_acceleration

    nn_input[7] = ∂σᵢ   = scaling.∂ρ∂z(ρ₀ * ∂z_b(i, j, k, grid, buoyancy, tracers) / g)
    nn_input[8] = ∂σᵢ₋₁ = scaling.∂ρ∂z(ρ₀ * ∂z_b(i, j, k, grid, buoyancy, tracers) / g)
    nn_input[9] = ∂σᵢ₊₁ = scaling.∂ρ∂z(ρ₀ * ∂z_b(i, j, k, grid, buoyancy, tracers) / g)

    @inbounds wT = inv(scaling.wT)(closure.wT(nn_input))
    @inbounds wS = inv(scaling.wS)(closure.wS(nn_input))

    @inbounds diffusivities.wT[i, j, k] = ifelse(k > grid.Nz - 2, 0, wT)
    @inbounds diffusivities.wS[i, j, k] = ifelse(k > grid.Nz - 2, 0, wS)
end

# Write here your constructor
# NNFluxClosure() = ... insert NN here ... (make sure it is on GPU if you need it on GPU!)

const NNC = NNFluxClosure
                                                         
#####
##### Abstract Smagorinsky functionality
#####

# Horizontal fluxes are zero!
@inline viscous_flux_wz( i, j, k, grid, clo::NNC, K, clk, fields, b) = zero(grid)
@inline viscous_flux_wx( i, j, k, grid, clo::NNC, K, clk, fields, b) = zero(grid)
@inline viscous_flux_wy( i, j, k, grid, clo::NNC, K, clk, fields, b) = zero(grid)
@inline viscous_flux_ux( i, j, k, grid, clo::NNC, K, clk, fields, b) = zero(grid)
@inline viscous_flux_vx( i, j, k, grid, clo::NNC, K, clk, fields, b) = zero(grid)
@inline viscous_flux_uy( i, j, k, grid, clo::NNC, K, clk, fields, b) = zero(grid)
@inline viscous_flux_vy( i, j, k, grid, clo::NNC, K, clk, fields, b) = zero(grid)
@inline diffusive_flux_x(i, j, k, grid, clo::NNC, K, ::Val{tracer_index}, c, clock, fields, buoyancy) where tracer_index = zero(grid)
@inline diffusive_flux_y(i, j, k, grid, clo::NNC, K, ::Val{tracer_index}, c, clock, fields, buoyancy) where tracer_index = zero(grid)

# Viscous fluxes are zero (for now)
@inline viscous_flux_uz(i, j, k, grid, clo::NNC, K, clk, fields, b) = zero(grid)
@inline viscous_flux_vz(i, j, k, grid, clo::NNC, K, clk, fields, b) = zero(grid)

# The only function extended by NNFluxClosure
@inline diffusive_flux_z(i, j, k, grid, clo::NNC, K, ::Val{1}, c, clock, fields, buoyancy) = @inbounds K.wT[i, j, k]
@inline diffusive_flux_z(i, j, k, grid, clo::NNC, K, ::Val{2}, c, clock, fields, buoyancy) = @inbounds K.wS[i, j, k]