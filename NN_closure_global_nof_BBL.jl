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
using Oceananigans.Grids: total_size
using Oceananigans.Utils: KernelParameters
using Oceananigans: architecture, on_architecture
using Lux, LuxCUDA
using JLD2
using ComponentArrays
using OffsetArrays

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

@inline (neural_network::NN)(input) = first(neural_network.model(input, neural_network.ps, neural_network.st))
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
    nn_path = "./NDE_FC_Qb_nof_BBL_trainFC24new_scalingtrain54new_2layer_64_relu_2Pr_model_temp.jld2"

    ps, sts, scaling_params, wT_model, wS_model = jldopen(nn_path, "r") do file
        ps = file["u"] |> dev |> f64
        sts = file["sts"] |> dev |> f64
        scaling_params = file["scaling"]
        wT_model = file["model"].wT
        wS_model = file["model"].wS
        return ps, sts, scaling_params, wT_model, wS_model
    end

    scaling = construct_zeromeanunitvariance_scaling(scaling_params)

    wT_NN = NN(wT_model, ps.wT, sts.wT)
    wS_NN = NN(wS_model, ps.wS, sts.wS)

    return NNFluxClosure(wT_NN, wS_NN, scaling)
end

function DiffusivityFields(grid, tracer_names, bcs, ::NNFluxClosure)
    arch = architecture(grid)
    wT = ZFaceField(grid)
    wS = ZFaceField(grid)
    ∂ρ²∂z² = ZFaceField(grid)
    BBL_index = Field((Center, Center, Nothing), grid, Int32)

    Nx_in, Ny_in, Nz_in = total_size(wT)
    ox_in, oy_in, oz_in = wT.data.offsets

    wrk_in = OffsetArray(zeros(10, Nx_in, Ny_in, Nz_in), 0, ox_in, oy_in, oz_in)
    wrk_in = on_architecture(arch, wrk_in)

    return (; wrk_in, wT, wS, ∂ρ²∂z², BBL_index)
end

function compute_diffusivities!(diffusivities, closure::NNFluxClosure, model; parameters = :xyz)
    arch = model.architecture
    grid = model.grid
    velocities = model.velocities
    tracers    = model.tracers
    buoyancy   = model.buoyancy
    coriolis   = model.coriolis    
    clock      = model.clock
    top_tracer_bcs = NamedTuple(c => tracers[c].boundary_conditions.top for c in propertynames(tracers))
    input = diffusivities.wrk_in
    wT = diffusivities.wT
    wS = diffusivities.wS
    ∂ρ²∂z² = diffusivities.∂ρ²∂z²
    BBL_index = diffusivities.BBL_index

    Nx_in, Ny_in, Nz_in = total_size(wT)
    ox_in, oy_in, oz_in = wT.data.offsets
    kp = KernelParameters((Nx_in, Ny_in, Nz_in), (ox_in, oy_in, oz_in))
    kp_2D = KernelParameters((Nx_in, Ny_in), (ox_in, oy_in))

    launch!(arch, grid, kp, 
            _populate_input!, input, ∂ρ²∂z², grid, closure, tracers, velocities, buoyancy, coriolis, top_tracer_bcs, clock)

    launch!(arch, grid, kp_2D, _find_base_boundary_layer!, ∂ρ²∂z², grid, BBL_index)

    wT.data.parent .= dropdims(closure.wT(input.parent), dims=1)
    wS.data.parent .= dropdims(closure.wS(input.parent), dims=1)

    launch!(arch, grid, kp, _adjust_nn_fluxes!, diffusivities, grid, closure, tracers, velocities, buoyancy, top_tracer_bcs, clock)
    return nothing
end

@kernel function _populate_input!(input, ∂ρ²∂z², grid, closure::NNFluxClosure, tracers, velocities, buoyancy, coriolis, top_tracer_bcs, clock)
    i, j, k = @index(Global, NTuple)

    scaling = closure.scaling

    ρ₀ = buoyancy.model.equation_of_state.reference_density
    g  = buoyancy.model.gravitational_acceleration

    @inbounds input[1, i, j, k] = ∂Tᵢ₋₁ = scaling.∂T∂z(∂zᶜᶜᶠ(i, j, k-1, grid, tracers.T))
    @inbounds input[2, i, j, k] = ∂Tᵢ   = scaling.∂T∂z(∂zᶜᶜᶠ(i, j, k,   grid, tracers.T))
    @inbounds input[3, i, j, k] = ∂Tᵢ₊₁ = scaling.∂T∂z(∂zᶜᶜᶠ(i, j, k+1, grid, tracers.T))

    @inbounds input[4, i, j, k] = ∂Sᵢ₋₁ = scaling.∂S∂z(∂zᶜᶜᶠ(i, j, k-1, grid, tracers.S))
    @inbounds input[5, i, j, k] = ∂Sᵢ   = scaling.∂S∂z(∂zᶜᶜᶠ(i, j, k,   grid, tracers.S))
    @inbounds input[6, i, j, k] = ∂Sᵢ₊₁ = scaling.∂S∂z(∂zᶜᶜᶠ(i, j, k+1, grid, tracers.S))

    @inbounds input[7, i, j, k] = ∂σᵢ₋₁ = scaling.∂ρ∂z(-ρ₀ * ∂z_b(i, j, k-1, grid, buoyancy, tracers) / g)
    @inbounds input[8, i, j, k] = ∂σᵢ   = scaling.∂ρ∂z(-ρ₀ * ∂z_b(i, j, k,   grid, buoyancy, tracers) / g)
    @inbounds input[9, i, j, k] = ∂σᵢ₊₁ = scaling.∂ρ∂z(-ρ₀ * ∂z_b(i, j, k+1, grid, buoyancy, tracers) / g)

    @inbounds input[10, i, j, k] = Jᵇ = scaling.wb(top_buoyancy_flux(i, j, grid, buoyancy, top_tracer_bcs, clock, merge(velocities, tracers)))

    @inbounds ∂ρ²∂z²[i, j, k] = abs(-ρ₀ * ∂zᶜᶜᶜ(i, j, k, grid, ∂z_b, buoyancy, tracers) / g)
end

@inline function find_field_max!(i, j, field, grid, h)
    kmax = grid.Nz
    @inbounds maxf = field[i, j, grid.Nz-1]

    @inbounds for k in grid.Nz-2:-1:2
        kmax = ifelse(field[i, j, k] > maxf, k, kmax)
        maxf = ifelse(field[i, j, k] > maxf, field[i, j, k], maxf)
    end

    @inbounds h[i, j, 1] = kmax
end

@kernel function _find_base_boundary_layer!(∂ρ²∂z², grid, h)
    i, j = @index(Global, NTuple)
    find_field_max!(i, j, ∂ρ²∂z², grid, h)

    @inbounds h[i, j, 1] = ifelse(h[i, j, 1] < 6, ifelse(h[i, j, 1] == 1, grid.Nz+1, 4), h[i, j, 1] - 2)
end

@kernel function _adjust_nn_fluxes!(diffusivities, grid, closure::NNFluxClosure, tracers, velocities, buoyancy, top_tracer_bcs, clock)
    i, j, k = @index(Global, NTuple)
    scaling = closure.scaling
    convecting = top_buoyancy_flux(i, j, grid, buoyancy, top_tracer_bcs, clock, merge(velocities, tracers)) > 0
    @inbounds above_base_boundary_layer = k > diffusivities.BBL_index[i, j, 1] & k <= grid.Nz - 1

    @inbounds diffusivities.wT[i, j, k] = ifelse(convecting & above_base_boundary_layer, scaling.wT.σ * diffusivities.wT[i, j, k], 0)
    @inbounds diffusivities.wS[i, j, k] = ifelse(convecting & above_base_boundary_layer, scaling.wS.σ * diffusivities.wS[i, j, k], 0)
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