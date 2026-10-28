# Flat-bottomed, high-latitude baroclinic channel on a latitude-longitude grid: the three-dimensional counterpart
# of forced_two_dimensional_turbulence.jl. A zonal-mean restoring toward a sloping two-layer stratification is released
# by baroclinic eddies at a deformation radius of ≈ 8 km, which an eddy-permitting 1/4° grid (Δx ≈ 14 km, Δy ≈ 28 km
# at 60°S) does not resolve. A fine run (truth) is compared to 1/4° runs with and without EnergyBudgetBackscatter.
#
# Usage: ARCH=GPU julia --project baroclinic_channel.jl   (needs CUDA on GPU, plus CairoMakie and FFTW)
#
# Environment variables: TRUTH_DEGREES (default 1/32), COARSE_DEGREES (1/4), BIHARMONIC_DAYS (0.1), STOP_YEARS (4),
# WENO (false: WENO=true swaps the energy-conserving momentum advection for WENO).

using Oceananigans
using Oceananigans.Units
using Oceananigans: UpdateStateCallsite
using Oceananigans.Advection: EnergyConserving
using Oceananigans.Operators: ℑxyᶠᶜᵃ, ℑxyᶜᶠᵃ, ℑyᵃᶠᵃ, Δxᶜᶜᶜ, Δyᶜᶜᶜ, Δzᶠᶜᶜ, Δzᶜᶠᶜ, ζ₃ᶠᶠᶜ
using CairoMakie
using FFTW
using Printf
using Statistics

if get(ENV, "ARCH", "CPU") == "GPU"
    using CUDA
end

arch = get(ENV, "ARCH", "CPU") == "GPU" ? GPU() : CPU()
parse_degrees(s) = prod(parse(Float64, n)^p for (n, p) in zip(split(s, "/"), (1, -1)))
truth_degrees = parse_degrees(get(ENV, "TRUTH_DEGREES", "1/32"))
coarse_degrees = parse_degrees(get(ENV, "COARSE_DEGREES", "1/4"))
τ₄ = parse(Float64, get(ENV, "BIHARMONIC_DAYS", "0.1")) * day   # biharmonic damping time at the grid scale Δ / π
stop_time = parse(Float64, get(ENV, "STOP_YEARS", "4")) * 365days
weno = parse(Bool, get(ENV, "WENO", "false"))

longitude = (0, 20)          # ≈ 1100 km at 60°S
latitude = (-65, -55)        # ≈ 1100 km
H = 2000
Δb = 9.81 * 3e-4             # reduced gravity of the layer interface: deformation radius ≈ 8 km
τ = 90days                   # zonal-mean restoring time scale
Cᴰ = 0.03                    # quadratic bottom drag coefficient
U₀ = 0.1                     # unresolved background bottom velocity

interface_height(φ) = -500 - 250 * tanh((φ + 60) / 1.25)
two_layer_buoyancy(φ, z) = Δb * (1 + tanh((z - interface_height(φ)) / 25)) / 2

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

# ν₄ = (Δ / π)⁴ / τ₄ with Δ = √(Δx Δy), the length scale of EnergyBudgetBackscatter
@inline biharmonic_viscosity(i, j, k, grid, clock, fields, p) = (Δxᶜᶜᶜ(i, j, k, grid) * Δyᶜᶜᶜ(i, j, k, grid))^2 / (π^4 * p.τ₄)

@inline upper_layer_v(i, j, k, grid, v, b, interface_buoyancy) = @inbounds v[i, j, k] * (ℑyᵃᶠᵃ(i, j, k, grid, b) > interface_buoyancy)

function run_channel(degrees; backscatter=false)
    Nλ = round(Int, (longitude[2] - longitude[1]) / degrees)
    Nφ = round(Int, (latitude[2] - latitude[1]) / degrees)
    z = vcat(range(-H, -1000, length=5)[1:end-1], range(-1000, 0, length=21))
    grid = LatitudeLongitudeGrid(arch; size=(Nλ, Nφ, length(z) - 1), halo=(5, 5, 4),
                                 longitude, latitude, z, topology=(Periodic, Bounded, Bounded))

    b★ = Field{Nothing, Center, Center}(grid)
    set!(b★, (φ, z) -> two_layer_buoyancy(φ, z))
    b̄ = Field{Nothing, Center, Center}(grid)

    forcing = (b = Forcing(zonal_mean_restoring, discrete_form=true, parameters=(; b̄, b★, τ)),
               u = Forcing(u_bottom_drag, discrete_form=true, parameters=(; Cᴰ, U₀)),
               v = Forcing(v_bottom_drag, discrete_form=true, parameters=(; Cᴰ, U₀)))

    viscosity = HorizontalScalarBiharmonicDiffusivity(ν=biharmonic_viscosity, discrete_form=true,
                                                      loc=(Center, Center, Center), parameters=(; τ₄))
    convective_adjustment = ConvectiveAdjustmentVerticalDiffusivity(convective_κz=0.1)
    closure = backscatter ? (viscosity, EnergyBudgetBackscatter(), convective_adjustment) :
                            (viscosity, convective_adjustment)

    model = HydrostaticFreeSurfaceModel(grid; closure, forcing,
                                        momentum_advection = weno ? WENOVectorInvariant(order=5) : VectorInvariant(vorticity_scheme=EnergyConserving()),
                                        tracer_advection = WENO(order=5),
                                        buoyancy = BuoyancyTracer(),
                                        coriolis = HydrostaticSphericalCoriolis(),
                                        tracers = :b)

    set!(model, b = (λ, φ, z) -> two_layer_buoyancy(φ, z) + 1e-3 * Δb * randn())

    # The biharmonic viscosity limits the time step of the coarse runs
    simulation = Simulation(model; Δt = min(15minutes * degrees / (1/4), 20minutes), stop_time)

    zonal_mean_buoyancy = Field(Average(model.tracers.b, dims=1), data=b̄.data)
    compute!(zonal_mean_buoyancy)
    simulation.callbacks[:zonal_mean] = Callback(model -> compute!(zonal_mean_buoyancy), callsite=UpdateStateCallsite())

    u, v, w = model.velocities
    b = model.tracers.b
    V₁ = Field(Integral(KernelFunctionOperation{Center, Face, Center}(upper_layer_v, grid, v, b, Δb / 2), dims=3))

    times, energies, surface_velocities, overturnings = Float64[], Float64[], [], []

    function record(sim)
        uₛ = Array(interior(u, :, :, grid.Nz))
        vₛ = Array(interior(v, :, 1:Nφ, grid.Nz))
        push!(times, time(sim))
        push!(energies, (mean((uₛ .- mean(uₛ, dims=1)) .^ 2) + mean((vₛ .- mean(vₛ, dims=1)) .^ 2)) / 2)

        if time(sim) > stop_time / 2
            push!(surface_velocities, (uₛ, vₛ))
            compute!(V₁)
            push!(overturnings, vec(mean(Array(interior(V₁)), dims=1)))
        end

        return nothing
    end

    progress(sim) = @info @sprintf("%g°%s: %s, surface EKE = %.2e m² s⁻², wall time: %s", degrees,
                                   backscatter ? " backscatter" : "", prettytime(sim), energies[end],
                                   prettytime(sim.run_wall_time))

    add_callback!(simulation, record, TimeInterval(5days))
    add_callback!(simulation, progress, TimeInterval(100days))
    run!(simulation)

    ζ = Field(KernelFunctionOperation{Face, Face, Center}(ζ₃ᶠᶠᶜ, grid, u, v), indices=(:, :, grid.Nz))
    compute!(ζ)

    return (; times, energies, surface_velocities, Ψ = mean(overturnings), φ = φnodes(grid, Face()), ζ)
end

# Isotropic spectrum E(K), with K in cycles per domain, normalized so that ∑ E(K) is the mean surface kinetic energy
function isotropic_spectrum(velocities)
    Nλ, Nφ = size(first(velocities)[1])
    E = mean(abs2.(fft(u .- mean(u, dims=1))) .+ abs2.(fft(v .- mean(v, dims=1))) for (u, v) in velocities) ./ (2 * (Nλ * Nφ)^2)
    K = [round(Int, hypot(k, l)) for k in fftfreq(Nλ, Nλ), l in fftfreq(Nφ, Nφ)]
    return [sum(E[K .== n]) for n in 1:min(Nλ, Nφ)÷2]
end

degree_label(degrees) = degrees < 1 ? "1/$(round(Int, 1 / degrees))°" : "$(degrees)°"

cases = [(degree_label(truth_degrees) * " (truth)",         truth_degrees,  false),
         (degree_label(coarse_degrees),                     coarse_degrees, false),
         (degree_label(coarse_degrees) * " backscatter",    coarse_degrees, true)]

results = [run_channel(degrees; backscatter) for (_, degrees, backscatter) in cases]

#####
##### Figure
#####

fig = Figure(size=(1500, 1100))
energy_axis = Axis(fig[1, 1:2], xlabel="t (days)", ylabel="surface EKE (m² s⁻²)")
spectrum_axis = Axis(fig[1, 3], xscale=log10, yscale=log10, xlabel="K (cycles per domain)", ylabel="E(K)",
                     title="Second-half mean, surface")
overturning_axis = Axis(fig[2, 1:3], xlabel="latitude (°)", ylabel="Ψ (m² s⁻¹)", title="Second-half mean overturning")

for ((label, _, _), result) in zip(cases, results)
    lines!(energy_axis, result.times ./ day, result.energies; label)
    lines!(spectrum_axis, isotropic_spectrum(result.surface_velocities); label)
    lines!(overturning_axis, result.φ, result.Ψ; label)
    second_half = result.times .> stop_time / 2
    @info @sprintf("%s: second-half mean surface EKE = %.2e m² s⁻², peak |Ψ| = %.2f m² s⁻¹",
                   label, mean(result.energies[second_half]), maximum(abs, result.Ψ))
end

axislegend(energy_axis, position=:lt)

for (n, ((label, _, _), result)) in enumerate(zip(cases, results))
    ζ_axis = Axis(fig[3, n], title="surface ζ, $label")
    ζmax = maximum(abs, interior(result.ζ))
    heatmap!(ζ_axis, result.ζ, colormap=:balance, colorrange=(-ζmax, ζmax))
end

save(@sprintf("baroclinic_channel_%gday", τ₄ / day) * (weno ? "_weno" : "") * ".png", fig)
