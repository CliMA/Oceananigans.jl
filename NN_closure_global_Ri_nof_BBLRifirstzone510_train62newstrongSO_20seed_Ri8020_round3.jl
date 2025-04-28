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

using Oceananigans.BuoyancyModels: ∂x_b, ∂y_b, ∂z_b, g_Earth
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

struct NNFluxClosure{A <: NN, S, G} <: AbstractTurbulenceClosure{ExplicitTimeDiscretization, 3}
    wT               :: A
    wS               :: A
    scaling          :: S
    grid_point_above :: G
    grid_point_below :: G
end

Adapt.adapt_structure(to, nn :: NNFluxClosure) = 
    NNFluxClosure(Adapt.adapt(to, nn.wT), 
                  Adapt.adapt(to, nn.wS), 
                  Adapt.adapt(to, nn.scaling),
                  Adapt.adapt(to, nn.grid_point_above),
                  Adapt.adapt(to, nn.grid_point_below))

Adapt.adapt_structure(to, nn :: NN) = 
    NN(Adapt.adapt(to, nn.model), 
       Adapt.adapt(to, nn.ps), 
       Adapt.adapt(to, nn.st))

function NNFluxClosure(arch)
    dev = ifelse(arch == GPU(), gpu_device(), cpu_device())
    # nn_path = "./NDE5_Qb_Ri_nof_BBLRifirst510_train62newstrongSO_scalingtrain62newstrongSO_validate30new_btrain56newstrongSO_3layer_128_relu_20seed_2Pr_ls10_model.jld2"
    nn_path = "./NDE5_Qb_Ri_nof_BBLRifirst510_train62newstrongSO_scalingtrain62newstrongSO_validate30new_btrain56newstrongSO_3layer_128_relu_20seed_2Pr_ls10_tm10_model_round3_4.jld2"

    ps, sts, scaling_params, wT_model, wS_model = jldopen(nn_path, "r") do file
        # ps = file["u_validation"] |> dev |> f64
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

    grid_point_above = 10
    grid_point_below = 5

    return NNFluxClosure(wT_NN, wS_NN, scaling, grid_point_above, grid_point_below)
end

function DiffusivityFields(grid, tracer_names, bcs, closure::NNFluxClosure)
    arch = architecture(grid)
    wT = ZFaceField(grid)
    wS = ZFaceField(grid)
    first_index = Field((Center, Center, Nothing), grid, Int32)
    last_index = Field((Center, Center, Nothing), grid, Int32)

    N_input = closure.wT.model.layers.layer_1.in_dims
    N_levels = closure.grid_point_above + closure.grid_point_below

    Nx_in, Ny_in, _ = size(wT)
    wrk_in = zeros(N_input, Nx_in, Ny_in, N_levels)
    wrk_in = on_architecture(arch, wrk_in)

    wrk_wT = zeros(Nx_in, Ny_in, 15)
    wrk_wS = zeros(Nx_in, Ny_in, 15)
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

    Riᶜ = model.closure[1].Riᶜ
    Ri = model.diffusivity_fields[1].Ri

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

    N_levels = closure.grid_point_above + closure.grid_point_below

    kp_wrk = KernelParameters((Nx_in, Ny_in, N_levels), (0, 0, 0))

    launch!(arch, grid, kp_2D, _find_NN_active_region!, Ri, grid, Riᶜ, first_index, last_index, closure)

    launch!(arch, grid, kp_wrk, 
            _populate_input!, wrk_in, first_index, last_index, grid, closure, tracers, velocities, buoyancy, top_tracer_bcs, Ri, clock)

    wrk_wT .= dropdims(closure.wT(wrk_in), dims=1)
    wrk_wS .= dropdims(closure.wS(wrk_in), dims=1)

    launch!(arch, grid, kp, _fill_adjust_nn_fluxes!, diffusivities, first_index, last_index, wrk_wT, wrk_wS, grid, closure, tracers, velocities, buoyancy, top_tracer_bcs, clock)
    return nothing
end

@kernel function _populate_input!(input, first_index, last_index, grid, closure::NNFluxClosure, tracers, velocities, buoyancy, top_tracer_bcs, Ri, clock)
    i, j, k = @index(Global, NTuple)

    scaling = closure.scaling

    # ρ₀ = buoyancy.model.equation_of_state.reference_density
    # g  = buoyancy.model.gravitational_acceleration

    ρ₀ = TEOS10EquationOfState().reference_density
    g = g_Earth

    @inbounds k_first = first_index[i, j, 1]
    @inbounds k_last = last_index[i, j, 1]

    quiescent = quiescent_condition(k_first, k_last)

    @inbounds k_tracer = clamp_k_interior(k_first + k - 1, grid)

    @inbounds k₋₂ = clamp_k_interior(k_tracer - 2, grid)
    @inbounds k₋₁ = clamp_k_interior(k_tracer - 1, grid)
    @inbounds k₀ = clamp_k_interior(k_tracer, grid)
    @inbounds k₊₁ = clamp_k_interior(k_tracer + 1, grid)
    @inbounds k₊₂ = clamp_k_interior(k_tracer + 2, grid)

    T, S = tracers.T, tracers.S

    @inbounds input[1, i, j, k] = ifelse(quiescent, 0, atan(Ri[i, j, k₋₂]))
    @inbounds input[2, i, j, k] = ifelse(quiescent, 0, atan(Ri[i, j, k₋₁]))
    @inbounds input[3, i, j, k] = ifelse(quiescent, 0, atan(Ri[i, j, k₀]))
    @inbounds input[4, i, j, k] = ifelse(quiescent, 0, atan(Ri[i, j, k₊₁]))
    @inbounds input[5, i, j, k] = ifelse(quiescent, 0, atan(Ri[i, j, k₊₂]))

    @inbounds input[6, i, j, k] = scaling.∂T∂z(ifelse(quiescent, 0, ∂zᶜᶜᶠ(i, j, k₋₂, grid, T)))
    @inbounds input[7, i, j, k] = scaling.∂T∂z(ifelse(quiescent, 0, ∂zᶜᶜᶠ(i, j, k₋₁, grid, T)))
    @inbounds input[8, i, j, k] = scaling.∂T∂z(ifelse(quiescent, 0, ∂zᶜᶜᶠ(i, j, k₀, grid, T)))
    @inbounds input[9, i, j, k] = scaling.∂T∂z(ifelse(quiescent, 0, ∂zᶜᶜᶠ(i, j, k₊₁, grid, T)))
    @inbounds input[10, i, j, k] = scaling.∂T∂z(ifelse(quiescent, 0, ∂zᶜᶜᶠ(i, j, k₊₂, grid, T)))

    @inbounds input[11, i, j, k] = scaling.∂S∂z(ifelse(quiescent, 0, ∂zᶜᶜᶠ(i, j, k₋₂, grid, S)))
    @inbounds input[12, i, j, k] = scaling.∂S∂z(ifelse(quiescent, 0, ∂zᶜᶜᶠ(i, j, k₋₁, grid, S)))
    @inbounds input[13, i, j, k] = scaling.∂S∂z(ifelse(quiescent, 0, ∂zᶜᶜᶠ(i, j, k₀, grid, S)))
    @inbounds input[14, i, j, k] = scaling.∂S∂z(ifelse(quiescent, 0, ∂zᶜᶜᶠ(i, j, k₊₁, grid, S)))
    @inbounds input[15, i, j, k] = scaling.∂S∂z(ifelse(quiescent, 0, ∂zᶜᶜᶠ(i, j, k₊₂, grid, S)))

    @inbounds input[16, i, j, k] = scaling.∂ρ∂z(ifelse(quiescent, 0, -ρ₀ * ∂z_b(i, j, k₋₂, grid, buoyancy, tracers) / g))
    @inbounds input[17, i, j, k] = scaling.∂ρ∂z(ifelse(quiescent, 0, -ρ₀ * ∂z_b(i, j, k₋₁, grid, buoyancy, tracers) / g))
    @inbounds input[18, i, j, k] = scaling.∂ρ∂z(ifelse(quiescent, 0, -ρ₀ * ∂z_b(i, j, k₀, grid, buoyancy, tracers) / g))
    @inbounds input[19, i, j, k] = scaling.∂ρ∂z(ifelse(quiescent, 0, -ρ₀ * ∂z_b(i, j, k₊₁, grid, buoyancy, tracers) / g))
    @inbounds input[20, i, j, k] = scaling.∂ρ∂z(ifelse(quiescent, 0, -ρ₀ * ∂z_b(i, j, k₊₂, grid, buoyancy, tracers) / g))

    @inbounds input[21, i, j, k] = scaling.wb(top_buoyancy_flux(i, j, grid, buoyancy, top_tracer_bcs, clock, merge(velocities, tracers)))

end

@kernel function _find_NN_active_region!(Ri, grid, Riᶜ, first_index, last_index, closure::NNFluxClosure)
    i, j = @index(Global, NTuple)
    top_index = grid.Nz + 1
    grid_point_above_kappa = closure.grid_point_above
    grid_point_below_kappa = closure.grid_point_below

    n_stable = 0
    n_unstable = 0

    # Find the first index of the background κᶜ
    kloc = grid.Nz+1

    # stability_threshold_cutoff = grid.Nz - 49

    # stability threshold cut off is reduced to avoid misdiagnosing the base of ML when mixed layer is not completely homogenous due to local entrainment-driven mixing
    stability_threshold_cutoff = grid.Nz - 24
    stability_threshold = 8/2
    @inbounds for k in grid.Nz:-1:2
        unstable = Ri[i, j, k] < Riᶜ

        # Count the number of stable and unstable points including and above the current point k
        n_stable = ifelse(unstable, n_stable, n_stable + 1)
        n_unstable = ifelse(unstable, n_unstable + 1, n_unstable)

        kloc = ifelse(unstable, ifelse(n_unstable >= n_stable, ifelse(k > stability_threshold_cutoff, k, ifelse(n_unstable/n_stable >= stability_threshold, k, kloc)), kloc), kloc)
    end

    background_κ_index = kloc - 1
    nonbackground_κ_index = background_κ_index + 1

    @inbounds last_index[i, j, 1] = ifelse(nonbackground_κ_index == top_index, top_index - 1, clamp_k_interior(background_κ_index + grid_point_above_kappa, grid))
    @inbounds first_index[i, j, 1] = ifelse(nonbackground_κ_index == top_index, top_index, clamp_k_interior(background_κ_index - grid_point_below_kappa + 1, grid))
end

@inline function quiescent_condition(lo, hi)
    return hi < lo
end

@inline function within_zone_condition(k, lo, hi)
    return (k >= lo) & (k <= hi)
end

@inline function clamp_k_interior(k, grid)
    kmax = grid.Nz
    kmin = 2

    return clamp(k, kmin, kmax)
end

@kernel function _fill_adjust_nn_fluxes!(diffusivities, first_index, last_index, wrk_wT, wrk_wS, grid, closure::NNFluxClosure, tracers, velocities, buoyancy, top_tracer_bcs, clock)
    i, j, k = @index(Global, NTuple)
    scaling = closure.scaling

    @inbounds k_first = first_index[i, j, 1]
    @inbounds k_last = last_index[i, j, 1]

    convecting = top_buoyancy_flux(i, j, grid, buoyancy, top_tracer_bcs, clock, merge(velocities, tracers)) > 0
    quiescent = quiescent_condition(k_first, k_last)
    within_zone = within_zone_condition(k, k_first, k_last)

    N_levels = closure.grid_point_above + closure.grid_point_below
    @inbounds k_wrk = clamp(k - k_first + 1, 1, N_levels)

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