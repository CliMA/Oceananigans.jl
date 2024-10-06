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
using SeawaterPolynomials.TEOS10

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
    nn_path = "./NDE_Qb_dt20min_nof_BBLkappazonelast41_wTwS_64simnew_2layer_128_relu_123seed_1.0e-5lr_localbaseclosure_2Pr_6simstableRi_model_temp.jld2"

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
    first_index = Field((Center, Center, Nothing), grid, Int32)
    last_index = Field((Center, Center, Nothing), grid, Int32)

    Nx_in, Ny_in, _ = size(wT)

    wrk_in = zeros(10, Nx_in, Ny_in, 5)
    wrk_in = on_architecture(arch, wrk_in)

    wrk_wT = zeros(Nx_in, Ny_in, 5)
    wrk_wS = zeros(Nx_in, Ny_in, 5)
    wrk_wT = on_architecture(arch, wrk_wT)
    wrk_wS = on_architecture(arch, wrk_wS)

    return (; wrk_in, wrk_wT, wrk_wS, wT, wS, first_index, last_index)
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

    κᶜ = model.diffusivity_fields[1].κᶜ
    κ₀ = model.closure[1].ν₀ / model.closure[1].Pr_shearₜ

    wrk_in = diffusivities.wrk_in
    wrk_wT = diffusivities.wrk_wT
    wrk_wS = diffusivities.wrk_wS
    wT = diffusivities.wT
    wS = diffusivities.wS

    first_index = diffusivities.first_index
    last_index = diffusivities.last_index

    Nx_in, Ny_in, Nz_in = total_size(wT)
    ox_in, oy_in, oz_in = wT.data.offsets
    kp = KernelParameters((Nx_in, Ny_in, Nz_in), (ox_in, oy_in, oz_in))
    kp_2D = KernelParameters((Nx_in, Ny_in), (ox_in, oy_in))

    kp_wrk = KernelParameters((Nx_in, Ny_in, 5), (0, 0, 0))

    launch!(arch, grid, kp_2D, _find_NN_active_region!, κᶜ, grid, κ₀, first_index, last_index)

    launch!(arch, grid, kp_wrk, 
            _populate_input!, wrk_in, first_index, last_index, grid, closure, tracers, velocities, buoyancy, top_tracer_bcs, clock)

    wrk_wT .= dropdims(closure.wT(wrk_in), dims=1)
    wrk_wS .= dropdims(closure.wS(wrk_in), dims=1)

    launch!(arch, grid, kp, _fill_adjust_nn_fluxes!, diffusivities, first_index, last_index, wrk_wT, wrk_wS, grid, closure, tracers, velocities, buoyancy, top_tracer_bcs, clock)
    return nothing
end

@kernel function _populate_input!(input, first_index, last_index, grid, closure::NNFluxClosure, tracers, velocities, buoyancy, top_tracer_bcs, clock)
    i, j, k = @index(Global, NTuple)

    scaling = closure.scaling

    ρ₀ = buoyancy.model.equation_of_state.reference_density
    g  = buoyancy.model.gravitational_acceleration
    eos = TEOS10.TEOS10EquationOfState()

    @inbounds quiescent = quiescent_condition(first_index[i, j, 1], last_index[i, j, 1])
    @inbounds k_tracer = first_index[i, j, 1] + k - 1

    T, S = tracers.T, tracers.S

    @inbounds input[1, i, j, k] = ∂Tᵢ₋₁ = scaling.∂T∂z(ifelse(quiescent, 0, ∂zᶜᶜᶠ(i, j, k_tracer-1, grid, T)))
    @inbounds input[2, i, j, k] = ∂Tᵢ   = scaling.∂T∂z(ifelse(quiescent, 0, ∂zᶜᶜᶠ(i, j, k_tracer,   grid, T)))
    @inbounds input[3, i, j, k] = ∂Tᵢ₊₁ = scaling.∂T∂z(ifelse(quiescent, 0, ∂zᶜᶜᶠ(i, j, k_tracer+1, grid, T)))

    @inbounds input[4, i, j, k] = ∂Sᵢ₋₁ = scaling.∂S∂z(ifelse(quiescent, 0, ∂zᶜᶜᶠ(i, j, k_tracer-1, grid, S)))
    @inbounds input[5, i, j, k] = ∂Sᵢ   = scaling.∂S∂z(ifelse(quiescent, 0, ∂zᶜᶜᶠ(i, j, k_tracer,   grid, S)))
    @inbounds input[6, i, j, k] = ∂Sᵢ₊₁ = scaling.∂S∂z(ifelse(quiescent, 0, ∂zᶜᶜᶠ(i, j, k_tracer+1, grid, S)))

    @inbounds input[7, i, j, k] = ∂σᵢ₋₁ = scaling.∂ρ∂z(ifelse(quiescent, 0, -ρ₀ * ∂z_b(i, j, k_tracer-1, grid, buoyancy, tracers) / g))
    @inbounds input[8, i, j, k] = ∂σᵢ   = scaling.∂ρ∂z(ifelse(quiescent, 0, -ρ₀ * ∂z_b(i, j, k_tracer,   grid, buoyancy, tracers) / g))
    @inbounds input[9, i, j, k] = ∂σᵢ₊₁ = scaling.∂ρ∂z(ifelse(quiescent, 0, -ρ₀ * ∂z_b(i, j, k_tracer+1, grid, buoyancy, tracers) / g))

    @inbounds input[10, i, j, k] = Jᵇ = scaling.wb(ifelse(quiescent, 0, top_buoyancy_flux(i, j, grid, buoyancy, top_tracer_bcs, clock, merge(velocities, tracers))))

end

@kernel function _find_NN_active_region!(κᶜ, grid, κ₀, first_index, last_index)
    i, j = @index(Global, NTuple)
    top_index = grid.Nz + 1

    # Find the last index of the background κᶜ
    kmax = 1
    @inbounds for k in 2:grid.Nz
        kmax = ifelse(κᶜ[i, j, k] ≈ κ₀, k, kmax)
    end

    @inbounds last_index[i, j, 1] = ifelse(kmax == top_index, grid.Nz, min(kmax + 1, grid.Nz))
    @inbounds first_index[i, j, 1] = ifelse(kmax == top_index, top_index, max(kmax - 3, 2))
end

@inline function quiescent_condition(lo, hi)
    return hi - lo != 4
end

@kernel function _fill_adjust_nn_fluxes!(diffusivities, first_index, last_index, wrk_wT, wrk_wS, grid, closure::NNFluxClosure, tracers, velocities, buoyancy, top_tracer_bcs, clock)
    i, j, k = @index(Global, NTuple)
    scaling = closure.scaling

    k_first = first_index[i, j, 1]
    k_last = last_index[i, j, 1]

    convecting = top_buoyancy_flux(i, j, grid, buoyancy, top_tracer_bcs, clock, merge(velocities, tracers)) > 0
    @inbounds quiescent = quiescent_condition(k_first, k_last)
    within_zone = (k >= k_first) & (k <= k_last)

    @inbounds k_wrk = clamp(k - k_first + 1, 1, 5)

    NN_active = convecting & !quiescent & within_zone

    @inbounds diffusivities.wT[i, j, k] = ifelse(NN_active, scaling.wT.σ * wrk_wT[i, j, k_wrk], 0)
    @inbounds diffusivities.wS[i, j, k] = ifelse(NN_active, scaling.wS.σ * wrk_wS[i, j, k_wrk], 0)
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