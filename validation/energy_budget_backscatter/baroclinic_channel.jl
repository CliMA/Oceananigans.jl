# Flat-bottomed baroclinic channel: the three-dimensional counterpart of forced_two_dimensional_turbulence.jl.
# A zonal-mean restoring toward a sloping two-layer stratification is released by baroclinic eddies at the
# deformation radius (≈ 15 km). A fine run (truth) is compared to coarse runs with and without EnergyBudgetBackscatter
# and Gent-McWilliams, whose potential energy release feeds the backscatter's sub-grid energy.
#
# Usage: ARCH=GPU julia --project baroclinic_channel.jl   (needs CUDA on GPU, plus CairoMakie and FFTW)
#
# Environment variables: TRUTH_KM (default 3.2), COARSE_KM (20), BIHARMONIC_DAYS (0.1), STOP_YEARS (4),
# GM_DIFFUSIVITY (500 m² s⁻¹), WENO (false: WENO=true swaps the energy-conserving momentum advection for WENO).

using Oceananigans
using Oceananigans.Units
using Oceananigans: UpdateStateCallsite
using Oceananigans.Advection: EnergyConserving
using Oceananigans.TurbulenceClosures: IsopycnalSkewSymmetricDiffusivity, AdvectiveFormulation
using Oceananigans.Operators: ℑxyᶠᶜᵃ, ℑxyᶜᶠᵃ, ℑyᵃᶠᵃ, Δzᶠᶜᶜ, Δzᶜᶠᶜ
using CairoMakie
using FFTW
using Printf
using Statistics

if get(ENV, "ARCH", "CPU") == "GPU"
    using CUDA
end

arch = get(ENV, "ARCH", "CPU") == "GPU" ? GPU() : CPU()
truth_Δ = parse(Float64, get(ENV, "TRUTH_KM", "3.2")) * kilometers
coarse_Δ = parse(Float64, get(ENV, "COARSE_KM", "20")) * kilometers
τ₄ = parse(Float64, get(ENV, "BIHARMONIC_DAYS", "0.1")) * day   # biharmonic damping time at the grid scale Δ / π
stop_time = parse(Float64, get(ENV, "STOP_YEARS", "4")) * 365days
κᴳᴹ = parse(Float64, get(ENV, "GM_DIFFUSIVITY", "500"))
weno = parse(Bool, get(ENV, "WENO", "false"))

Lx = Ly = 800kilometers
H = 2000
Δb = 9.81 * 7e-4             # reduced gravity of the layer interface
τ = 90days                   # zonal-mean restoring time scale
Cᴰ = 0.03                    # quadratic bottom drag coefficient
U₀ = 0.1                     # unresolved background bottom velocity
f₀ = 1.1e-4
β = 1.2e-11

interface_height(y) = -500 - 250 * tanh((y - Ly / 2) / 125kilometers)
two_layer_buoyancy(y, z) = Δb * (1 + tanh((z - interface_height(y)) / 25)) / 2

@inline zonal_mean_restoring(i, j, k, grid, clock, fields, p) =
    @inbounds - (p.b̄[1, j, k] - p.b★[1, j, k]) / p.τ

@inline function u_bottom_drag(i, j, k, grid, clock, fields, p)
    u = @inbounds fields.u[i, j, k]
    v = ℑxyᶠᶜᵃ(i, j, k, grid, fields.v)
    drag = - p.Cᴰ * sqrt(p.U₀^2 + u^2 + v^2) * u / Δzᶠᶜᶜ(i, j, k, grid)
    return ifelse(k == 1, drag, zero(grid))
end

@inline function v_bottom_drag(i, j, k, grid, clock, fields, p)
    u = ℑxyᶜᶠᵃ(i, j, k, grid, fields.u)
    v = @inbounds fields.v[i, j, k]
    drag = - p.Cᴰ * sqrt(p.U₀^2 + u^2 + v^2) * v / Δzᶜᶠᶜ(i, j, k, grid)
    return ifelse(k == 1, drag, zero(grid))
end

@inline upper_layer_v(i, j, k, grid, v, b, interface_buoyancy) = @inbounds v[i, j, k] * (ℑyᵃᶠᵃ(i, j, k, grid, b) > interface_buoyancy)

function run_channel(Δ; backscatter=false, gm=false)
    Nx = Ny = round(Int, Lx / Δ)
    z = vcat(range(-H, -1000, length=5)[1:end-1], range(-1000, 0, length=21))
    grid = RectilinearGrid(arch; size=(Nx, Ny, length(z) - 1), halo=(4, 4, 4),
                           x=(0, Lx), y=(0, Ly), z, topology=(Periodic, Bounded, Bounded))

    b★ = Field{Nothing, Center, Center}(grid)
    set!(b★, (y, z) -> two_layer_buoyancy(y, z))
    b̄ = Field{Nothing, Center, Center}(grid)

    forcing = (b = Forcing(zonal_mean_restoring, discrete_form=true, parameters=(; b̄, b★, τ)),
               u = Forcing(u_bottom_drag, discrete_form=true, parameters=(; Cᴰ, U₀)),
               v = Forcing(v_bottom_drag, discrete_form=true, parameters=(; Cᴰ, U₀)))

    biharmonic_viscosity = HorizontalScalarBiharmonicDiffusivity(ν = (Δ / π)^4 / τ₄)
    convective_adjustment = ConvectiveAdjustmentVerticalDiffusivity(convective_κz=0.1)
    closure = (biharmonic_viscosity, convective_adjustment)
    gm && (closure = (closure..., IsopycnalSkewSymmetricDiffusivity(κ_skew=κᴳᴹ, skew_flux_formulation=AdvectiveFormulation())))
    backscatter && (closure = (closure..., EnergyBudgetBackscatter()))

    model = HydrostaticFreeSurfaceModel(grid; closure, forcing,
                                        momentum_advection = weno ? WENOVectorInvariant(order=5) : VectorInvariant(vorticity_scheme=EnergyConserving()),
                                        tracer_advection = WENO(order=5),
                                        buoyancy = BuoyancyTracer(),
                                        coriolis = BetaPlane(; f₀, β),
                                        tracers = :b)

    set!(model, b = (x, y, z) -> two_layer_buoyancy(y, z) + 1e-3 * Δb * randn())

    simulation = Simulation(model; Δt = 20minutes * Δ / 20kilometers, stop_time)

    zonal_mean_buoyancy = Field(Average(model.tracers.b, dims=1), data=b̄.data)
    compute!(zonal_mean_buoyancy)
    simulation.callbacks[:zonal_mean] = Callback(model -> compute!(zonal_mean_buoyancy), callsite=UpdateStateCallsite())

    u, v, w = model.velocities
    b = model.tracers.b
    # The overturning of GM runs includes the eddy-induced velocity
    transport_velocity = gm ? v + model.closure_fields[3].v : v
    V₁ = Field(Integral(KernelFunctionOperation{Center, Face, Center}(upper_layer_v, grid, transport_velocity, b, Δb / 2), dims=3))

    times, energies, surface_velocities, overturnings = Float64[], Float64[], [], []

    function record(sim)
        uₛ = Array(interior(u, :, :, grid.Nz))
        vₛ = Array(interior(v, :, 1:Ny, grid.Nz))
        push!(times, time(sim))
        push!(energies, (mean((uₛ .- mean(uₛ, dims=1)) .^ 2) + mean((vₛ .- mean(vₛ, dims=1)) .^ 2)) / 2)

        if time(sim) > stop_time / 2
            push!(surface_velocities, (uₛ, vₛ))
            compute!(V₁)
            push!(overturnings, vec(mean(Array(interior(V₁)), dims=1)))
        end

        return nothing
    end

    progress(sim) = @info @sprintf("Δ = %g km%s%s: %s, surface EKE = %.2e m² s⁻², wall time: %s", Δ / kilometers,
                                   gm ? " GM" : "", backscatter ? " backscatter" : "", prettytime(sim), energies[end],
                                   prettytime(sim.run_wall_time))

    add_callback!(simulation, record, TimeInterval(5days))
    add_callback!(simulation, progress, TimeInterval(100days))
    run!(simulation)

    ζ = Field(∂x(v) - ∂y(u), indices=(:, :, grid.Nz))
    compute!(ζ)

    return (; times, energies, surface_velocities, Ψ = mean(overturnings), y = ynodes(grid, Face()), ζ)
end

# Isotropic spectrum E(K), with K in cycles per domain, normalized so that ∑ E(K) is the mean surface kinetic energy
function isotropic_spectrum(velocities)
    N = size(first(velocities)[1], 1)
    E = mean(abs2.(fft(u .- mean(u, dims=1))) .+ abs2.(fft(v .- mean(v, dims=1))) for (u, v) in velocities) ./ (2 * N^4)
    K = [round(Int, hypot(k, l)) for k in fftfreq(N, N), l in fftfreq(N, N)]
    return [sum(E[K .== n]) for n in 1:N÷2]
end

cases = [(@sprintf("%g km (truth)", truth_Δ / kilometers),          truth_Δ,  false, false),
         (@sprintf("%g km", coarse_Δ / kilometers),                  coarse_Δ, false, false),
         (@sprintf("%g km backscatter", coarse_Δ / kilometers),      coarse_Δ, true,  false),
         (@sprintf("%g km GM", coarse_Δ / kilometers),               coarse_Δ, false, true),
         (@sprintf("%g km GM backscatter", coarse_Δ / kilometers),   coarse_Δ, true,  true)]

results = [run_channel(Δ; backscatter, gm) for (_, Δ, backscatter, gm) in cases]

#####
##### Figure
#####

fig = Figure(size=(2000, 1100))
energy_axis = Axis(fig[1, 1:3], xlabel="t (days)", ylabel="surface EKE (m² s⁻²)")
spectrum_axis = Axis(fig[1, 4:5], xscale=log10, yscale=log10, xlabel="K (cycles per domain)", ylabel="E(K)",
                     title="Second-half mean, surface")
overturning_axis = Axis(fig[2, 1:5], xlabel="y (km)", ylabel="Ψ (m² s⁻¹)", title="Second-half mean overturning")

for ((label, _...), result) in zip(cases, results)
    lines!(energy_axis, result.times ./ day, result.energies; label)
    lines!(spectrum_axis, isotropic_spectrum(result.surface_velocities); label)
    lines!(overturning_axis, result.y ./ kilometers, result.Ψ; label)
    second_half = result.times .> stop_time / 2
    @info @sprintf("%s: second-half mean surface EKE = %.2e m² s⁻², peak |Ψ| = %.2f m² s⁻¹",
                   label, mean(result.energies[second_half]), maximum(abs, result.Ψ))
end

axislegend(energy_axis, position=:lt)

ζmax = maximum(abs, interior(first(results).ζ))
for (n, ((label, _...), result)) in enumerate(zip(cases, results))
    ζ_axis = Axis(fig[3, n], title="surface ζ, $label", aspect=1)
    heatmap!(ζ_axis, result.ζ, colormap=:balance, colorrange=(-ζmax, ζmax))
end

save(@sprintf("baroclinic_channel_%gday", τ₄ / day) * (weno ? "_weno" : "") * ".png", fig)
