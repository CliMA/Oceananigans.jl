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
using Oceananigans.Grids: total_size, constructor_arguments
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

struct NNFluxClosure{A <: NN, S, G, R} <: AbstractTurbulenceClosure{ExplicitTimeDiscretization, 3}
    wT               :: A
    wS               :: A
    scaling          :: S
    grid_point_above :: G
    grid_point_below :: G
    resolution       :: R
end

Adapt.adapt_structure(to, nn :: NNFluxClosure) = 
    NNFluxClosure(Adapt.adapt(to, nn.wT), 
                  Adapt.adapt(to, nn.wS), 
                  Adapt.adapt(to, nn.scaling),
                  Adapt.adapt(to, nn.grid_point_above),
                  Adapt.adapt(to, nn.grid_point_below),
                  Adapt.adapt(to, nn.resolution))

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
    resolution = 8

    return NNFluxClosure(wT_NN, wS_NN, scaling, grid_point_above, grid_point_below, resolution)
end

function compute_interpolation_weights!(lower_grid_points, interpolation_weights, source_grid, target_grid)
    Nz_target = target_grid.Nz
    Nz_source = source_grid.Nz

    zs_target = target_grid.zᵃᵃᶠ[1:Nz_target+1]
    zs_source = source_grid.zᵃᵃᶠ[1:Nz_source+1]

    k_lower = 1
    for k in eachindex(lower_grid_points)
        z_target = zs_target[k]
        if zs_target[k] > zs_source[k_lower + 1]
            k_lower += 1
        end
        
        lower_grid_points[k] = k_lower

        z_lower = zs_source[k_lower]
        z_upper = zs_source[k_lower + 1]

        interpolation_weights[k, 1] = (z_upper - z_target) / (z_upper - z_lower)
        interpolation_weights[k, 2] = (z_target - z_lower) / (z_upper - z_lower)
    end
end

function interpolate_value(v₁, v₂, w₁, w₂)
    return v₁ * w₁ + v₂ * w₂
end

function DiffusivityFields(grid, tracer_names, bcs, closure::NNFluxClosure)
    arch = architecture(grid)
    wT = ZFaceField(grid)
    wS = ZFaceField(grid)

    first_index = Field((Center, Center, Nothing), grid, Int)
    last_index = Field((Center, Center, Nothing), grid, Int)

    N_input = closure.wT.model.layers.layer_1.in_dims
    N_levels = closure.grid_point_above + closure.grid_point_below

    Nx_in, Ny_in, _ = size(wT)
    wrk_in = zeros(N_input, Nx_in, Ny_in, N_levels)
    wrk_in = on_architecture(arch, wrk_in)

    wrk_wT = zeros(Nx_in, Ny_in, N_levels)
    wrk_wS = zeros(Nx_in, Ny_in, N_levels)
    wrk_wT = on_architecture(arch, wrk_wT)
    wrk_wS = on_architecture(arch, wrk_wS)

    z_top = grid.zᵃᵃᶠ[grid.Nz + 1]

    Nz_wrk_grid = Int(ceil(grid.Lz / closure.resolution))
    z_top_wrk = z_top
    z_bottom_wrk = z_top - closure.resolution * Nz_wrk_grid

    grid_args, grid_kwargs = constructor_arguments(grid)
    grid_kwargs[:z] = (z_bottom_wrk, z_top_wrk)
    grid_kwargs[:size] = (grid_kwargs[:size][1:end-1]..., Nz_wrk_grid)
    arch = grid_args[:architecture]
    FT = grid_args[:number_type]

    if grid isa RectilinearGrid
        wrk_grid = RectilinearGrid(arch, FT; grid_kwargs...)
    elseif grid isa OrthogonalSphericalShellGrid
        wrk_grid = OrthogonalSphericalShellGrid(arch, FT; grid_kwargs...)
    elseif grid isa LatitudeLongitudeGrid
        wrk_grid = LatitudeLongitudeGrid(arch, FT; grid_kwargs...)
    else
        error("Unsupported grid type: $(typeof(grid))")
    end

    lower_grid_point_main = zeros(Int, Nz_wrk_grid + 1)
    interpolation_weights_main = zeros(Nz_wrk_grid + 1, 2)
    
    lower_grid_point_wrk = zeros(Int, grid.Nz + 1)
    interpolation_weights_wrk = zeros(grid.Nz + 1, 2)
    
    compute_interpolation_weights!(lower_grid_point_main, interpolation_weights_main, grid, wrk_grid)
    compute_interpolation_weights!(lower_grid_point_wrk, interpolation_weights_wrk, wrk_grid, grid)
    
    lower_grid_point_main = on_architecture(arch, lower_grid_point_main)
    lower_grid_point_wrk = on_architecture(arch, lower_grid_point_wrk)
    interpolation_weights_main = on_architecture(arch, interpolation_weights_main)
    interpolation_weights_wrk = on_architecture(arch, interpolation_weights_wrk)

    main_to_wrk = (; lower_grid_point = lower_grid_point_main, weights = interpolation_weights_main)
    wrk_to_main = (; lower_grid_point = lower_grid_point_wrk, weights = interpolation_weights_wrk)

    return (; wrk_in, wrk_wT, wrk_wS, wT, wS, first_index, last_index, wrk_grid, main_to_wrk, wrk_to_main)
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

    launch!(arch, grid, kp_2D, _find_NN_active_region!, Ri, grid, diffusivities.wrk_grid, Riᶜ, first_index, last_index, closure)

    launch!(arch, grid, kp_wrk, 
            _populate_input!, wrk_in, first_index, last_index, grid, diffusivities.wrk_grid, diffusivities.main_to_wrk, closure, tracers, velocities, buoyancy, top_tracer_bcs, Ri, clock)

    wrk_wT .= dropdims(closure.wT(wrk_in), dims=1)
    wrk_wS .= dropdims(closure.wS(wrk_in), dims=1)

    launch!(arch, grid, kp, _fill_adjust_nn_fluxes!, diffusivities, first_index, last_index, wrk_wT, wrk_wS, grid, diffusivities.wrk_grid, diffusivities.wrk_to_main, closure, tracers, velocities, buoyancy, top_tracer_bcs, clock)
    return nothing
end

@inline function clamp_k_interior(k, grid)
    kmax = grid.Nz
    kmin = 2

    return clamp(k, kmin, kmax)
end

@kernel function _find_NN_active_region!(Ri, main_grid, wrk_grid, Riᶜ, first_index, last_index, closure::NNFluxClosure)
    i, j = @index(Global, NTuple)
    grid_point_above_kappa = closure.grid_point_above
    grid_point_below_kappa = closure.grid_point_below
    resolution = closure.resolution

    distance_above_kappa = grid_point_above_kappa * resolution
    distance_below_kappa = grid_point_below_kappa * resolution

    n_stable = 0
    n_unstable = 0

    # Find the first index of the background κᶜ
    kloc = main_grid.Nz+1

    # stability_threshold_cutoff = grid.Nz - 49

    # stability threshold cut off is reduced to avoid misdiagnosing the base of ML when mixed layer is not completely homogenous due to local entrainment-driven mixing
    stability_threshold_cutoff = main_grid.Nz - 24
    stability_threshold = 8/2
    @inbounds for k in main_grid.Nz:-1:2
        unstable = Ri[i, j, k] < Riᶜ

        # Count the number of stable and unstable points including and above the current point k
        n_stable = ifelse(unstable, n_stable, n_stable + 1)
        n_unstable = ifelse(unstable, n_unstable + 1, n_unstable)

        kloc = ifelse(unstable, ifelse(n_unstable >= n_stable, ifelse(k > stability_threshold_cutoff, k, ifelse(n_unstable/n_stable >= stability_threshold, k, kloc)), kloc), kloc)
    end

    background_κ_index = kloc - 1

    @inbounds background_kappa_z = main_grid.zᵃᵃᶠ[background_κ_index]
    @inbounds z_top = main_grid.zᵃᵃᶠ[main_grid.Nz + 1]

    background_κ_index_wrk = wrk_grid.Nz + 1 - Int(round((z_top - background_kappa_z) / closure.resolution))
    nonbackground_κ_index_wrk = background_κ_index_wrk + 1

    top_index = wrk_grid.Nz + 1
    @inbounds last_index[i, j, 1] = ifelse(nonbackground_κ_index_wrk == top_index, top_index - 1, clamp_k_interior(background_κ_index_wrk + grid_point_above_kappa, wrk_grid))
    @inbounds first_index[i, j, 1] = ifelse(nonbackground_κ_index_wrk == top_index, top_index, clamp_k_interior(background_κ_index_wrk - grid_point_below_kappa + 1, wrk_grid))
end

@kernel function _populate_input!(input, first_index, last_index, main_grid, wrk_grid, main_to_wrk, closure::NNFluxClosure, tracers, velocities, buoyancy, top_tracer_bcs, Ri, clock)
    i, j, k = @index(Global, NTuple)

    scaling = closure.scaling
    T, S = tracers.T, tracers.S

    ρ₀ = buoyancy.model.equation_of_state.reference_density
    g = buoyancy.model.gravitational_acceleration

    @inbounds k_first = first_index[i, j, 1]
    @inbounds k_last = last_index[i, j, 1]

    quiescent = quiescent_condition(k_first, k_last)

    @inbounds k_tracer = clamp_k_interior(k_first + k - 1, wrk_grid)

    @inbounds k₋₂ = clamp_k_interior(k_tracer - 2, wrk_grid)
    @inbounds k₋₁ = clamp_k_interior(k_tracer - 1, wrk_grid)
    @inbounds k₀ = clamp_k_interior(k_tracer, wrk_grid)
    @inbounds k₊₁ = clamp_k_interior(k_tracer + 1, wrk_grid)
    @inbounds k₊₂ = clamp_k_interior(k_tracer + 2, wrk_grid)

    @inbounds k_lower₋₂ = main_to_wrk.lower_grid_point[k₋₂]
    @inbounds k_lower₋₁ = main_to_wrk.lower_grid_point[k₋₁]
    @inbounds k_lower₀ = main_to_wrk.lower_grid_point[k₀]
    @inbounds k_lower₊₁ = main_to_wrk.lower_grid_point[k₊₁]
    @inbounds k_lower₊₂ = main_to_wrk.lower_grid_point[k₊₂]

    @inbounds w_lower₋₂ = main_to_wrk.weights[k₋₂, 1]
    @inbounds w_lower₋₁ = main_to_wrk.weights[k₋₁, 1]
    @inbounds w_lower₀ = main_to_wrk.weights[k₀, 1]
    @inbounds w_lower₊₁ = main_to_wrk.weights[k₊₁, 1]
    @inbounds w_lower₊₂ = main_to_wrk.weights[k₊₂, 1]

    @inbounds w_upper₋₂ = main_to_wrk.weights[k₋₂, 2]
    @inbounds w_upper₋₁ = main_to_wrk.weights[k₋₁, 2]
    @inbounds w_upper₀ = main_to_wrk.weights[k₀, 2]
    @inbounds w_upper₊₁ = main_to_wrk.weights[k₊₁, 2]
    @inbounds w_upper₊₂ = main_to_wrk.weights[k₊₂, 2]

    @inbounds input[1, i, j, k] = ifelse(quiescent, 0, interpolate_value(atan(Ri[i, j, k_lower₋₂]), atan(Ri[i, j, k_lower₋₂ + 1]), w_lower₋₂, w_upper₋₂))
    @inbounds input[2, i, j, k] = ifelse(quiescent, 0, interpolate_value(atan(Ri[i, j, k_lower₋₁]), atan(Ri[i, j, k_lower₋₁ + 1]), w_lower₋₁, w_upper₋₁))
    @inbounds input[3, i, j, k] = ifelse(quiescent, 0, interpolate_value(atan(Ri[i, j, k_lower₀]), atan(Ri[i, j, k_lower₀ + 1]), w_lower₀, w_upper₀))
    @inbounds input[4, i, j, k] = ifelse(quiescent, 0, interpolate_value(atan(Ri[i, j, k_lower₊₁]), atan(Ri[i, j, k_lower₊₁ + 1]), w_lower₊₁, w_upper₊₁))
    @inbounds input[5, i, j, k] = ifelse(quiescent, 0, interpolate_value(atan(Ri[i, j, k_lower₊₂]), atan(Ri[i, j, k_lower₊₂ + 1]), w_lower₊₂, w_upper₊₂))

    @inbounds input[6, i, j, k] = ifelse(quiescent, 0, interpolate_value(scaling.∂T∂z(∂zᶜᶜᶠ(i, j, k_lower₋₂, main_grid, T)), scaling.∂T∂z(∂zᶜᶜᶠ(i, j, k_lower₋₂ + 1, main_grid, T)), w_lower₋₂, w_upper₋₂))
    @inbounds input[7, i, j, k] = ifelse(quiescent, 0, interpolate_value(scaling.∂T∂z(∂zᶜᶜᶠ(i, j, k_lower₋₁, main_grid, T)), scaling.∂T∂z(∂zᶜᶜᶠ(i, j, k_lower₋₁ + 1, main_grid, T)), w_lower₋₁, w_upper₋₁))
    @inbounds input[8, i, j, k] = ifelse(quiescent, 0, interpolate_value(scaling.∂T∂z(∂zᶜᶜᶠ(i, j, k_lower₀, main_grid, T)), scaling.∂T∂z(∂zᶜᶜᶠ(i, j, k_lower₀ + 1, main_grid, T)), w_lower₀, w_upper₀))
    @inbounds input[9, i, j, k] = ifelse(quiescent, 0, interpolate_value(scaling.∂T∂z(∂zᶜᶜᶠ(i, j, k_lower₊₁, main_grid, T)), scaling.∂T∂z(∂zᶜᶜᶠ(i, j, k_lower₊₁ + 1, main_grid, T)), w_lower₊₁, w_upper₊₁))
    @inbounds input[10, i, j, k] = ifelse(quiescent, 0, interpolate_value(scaling.∂T∂z(∂zᶜᶜᶠ(i, j, k_lower₊₂, main_grid, T)), scaling.∂T∂z(∂zᶜᶜᶠ(i, j, k_lower₊₂ + 1, main_grid, T)), w_lower₊₂, w_upper₊₂))

    @inbounds input[11, i, j, k] = ifelse(quiescent, 0, interpolate_value(scaling.∂S∂z(∂zᶜᶜᶠ(i, j, k_lower₋₂, main_grid, S)), scaling.∂S∂z(∂zᶜᶜᶠ(i, j, k_lower₋₂ + 1, main_grid, S)), w_lower₋₂, w_upper₋₂))
    @inbounds input[12, i, j, k] = ifelse(quiescent, 0, interpolate_value(scaling.∂S∂z(∂zᶜᶜᶠ(i, j, k_lower₋₁, main_grid, S)), scaling.∂S∂z(∂zᶜᶜᶠ(i, j, k_lower₋₁ + 1, main_grid, S)), w_lower₋₁, w_upper₋₁))
    @inbounds input[13, i, j, k] = ifelse(quiescent, 0, interpolate_value(scaling.∂S∂z(∂zᶜᶜᶠ(i, j, k_lower₀, main_grid, S)), scaling.∂S∂z(∂zᶜᶜᶠ(i, j, k_lower₀ + 1, main_grid, S)), w_lower₀, w_upper₀))
    @inbounds input[14, i, j, k] = ifelse(quiescent, 0, interpolate_value(scaling.∂S∂z(∂zᶜᶜᶠ(i, j, k_lower₊₁, main_grid, S)), scaling.∂S∂z(∂zᶜᶜᶠ(i, j, k_lower₊₁ + 1, main_grid, S)), w_lower₊₁, w_upper₊₁))
    @inbounds input[15, i, j, k] = ifelse(quiescent, 0, interpolate_value(scaling.∂S∂z(∂zᶜᶜᶠ(i, j, k_lower₊₂, main_grid, S)), scaling.∂S∂z(∂zᶜᶜᶠ(i, j, k_lower₊₂ + 1, main_grid, S)), w_lower₊₂, w_upper₊₂))

    @inbounds input[16, i, j, k] = ifelse(quiescent, 0, interpolate_value(scaling.∂ρ∂z(-ρ₀ * ∂z_b(i, j, k_lower₋₂, main_grid, buoyancy, tracers) / g), scaling.∂ρ∂z(-ρ₀ * ∂z_b(i, j, k_lower₋₂ + 1, main_grid, buoyancy, tracers) / g), w_lower₋₂, w_upper₋₂))
    @inbounds input[17, i, j, k] = ifelse(quiescent, 0, interpolate_value(scaling.∂ρ∂z(-ρ₀ * ∂z_b(i, j, k_lower₋₁, main_grid, buoyancy, tracers) / g), scaling.∂ρ∂z(-ρ₀ * ∂z_b(i, j, k_lower₋₁ + 1, main_grid, buoyancy, tracers) / g), w_lower₋₁, w_upper₋₁))
    @inbounds input[18, i, j, k] = ifelse(quiescent, 0, interpolate_value(scaling.∂ρ∂z(-ρ₀ * ∂z_b(i, j, k_lower₀, main_grid, buoyancy, tracers) / g), scaling.∂ρ∂z(-ρ₀ * ∂z_b(i, j, k_lower₀ + 1, main_grid, buoyancy, tracers) / g), w_lower₀, w_upper₀))
    @inbounds input[19, i, j, k] = ifelse(quiescent, 0, interpolate_value(scaling.∂ρ∂z(-ρ₀ * ∂z_b(i, j, k_lower₊₁, main_grid, buoyancy, tracers) / g), scaling.∂ρ∂z(-ρ₀ * ∂z_b(i, j, k_lower₊₁ + 1, main_grid, buoyancy, tracers) / g), w_lower₊₁, w_upper₊₁))
    @inbounds input[20, i, j, k] = ifelse(quiescent, 0, interpolate_value(scaling.∂ρ∂z(-ρ₀ * ∂z_b(i, j, k_lower₊₂, main_grid, buoyancy, tracers) / g), scaling.∂ρ∂z(-ρ₀ * ∂z_b(i, j, k_lower₊₂ + 1, main_grid, buoyancy, tracers) / g), w_lower₊₂, w_upper₊₂))

    @inbounds input[21, i, j, k] = scaling.wb(top_buoyancy_flux(i, j, main_grid, buoyancy, top_tracer_bcs, clock, merge(velocities, tracers)))
end

@inline function quiescent_condition(lo, hi)
    return hi < lo
end

@inline function within_zone_condition(z, lo, hi)
    return (z >= lo) & (z <= hi)
end

@kernel function _fill_adjust_nn_fluxes!(diffusivities, first_index, last_index, wrk_wT, wrk_wS, main_grid, wrk_grid, wrk_to_main, closure::NNFluxClosure, tracers, velocities, buoyancy, top_tracer_bcs, clock)
    i, j, k = @index(Global, NTuple)
    scaling = closure.scaling

    @inbounds k_first = first_index[i, j, 1]
    @inbounds k_last = last_index[i, j, 1]

    convecting = top_buoyancy_flux(i, j, main_grid, buoyancy, top_tracer_bcs, clock, merge(velocities, tracers)) > 0
    quiescent = quiescent_condition(k_first, k_last)

    z = main_grid.zᵃᵃᶠ[k]
    z_first = wrk_grid.zᵃᵃᶠ[k_first]
    z_last = wrk_grid.zᵃᵃᶠ[k_last]
    within_zone = within_zone_condition(z, z_first, z_last)

    N_levels = closure.grid_point_above + closure.grid_point_below
    @inbounds k_wrk_lower = clamp(Int(floor((z - z_first) / closure.resolution)), 1, N_levels)
    @inbounds k_wrk_upper = clamp(k_wrk_lower + 1, 1, N_levels)

    @inbounds w_lower = wrk_to_main.weights[clamp(k, 1, main_grid.Nz + 1), 1]
    @inbounds w_upper = wrk_to_main.weights[clamp(k, 1, main_grid.Nz + 1), 2]

    NN_active = convecting & !quiescent & within_zone

    @inbounds diffusivities.wT[i, j, k] = ifelse(NN_active, scaling.wT.σ * interpolate_value(wrk_wT[i, j, k_wrk_lower], wrk_wT[i, j, k_wrk_upper], w_lower, w_upper), 0)
    @inbounds diffusivities.wS[i, j, k] = ifelse(NN_active, scaling.wS.σ * interpolate_value(wrk_wS[i, j, k_wrk_lower], wrk_wS[i, j, k_wrk_upper], w_lower, w_upper), 0)
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