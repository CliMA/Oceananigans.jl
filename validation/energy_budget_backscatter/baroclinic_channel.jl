# Flat-bottomed, high-latitude baroclinic channel on a latitude-longitude grid: the three-dimensional counterpart
# of forced_two_dimensional_turbulence.jl. A zonal-mean restoring toward a sloping two-layer stratification is released
# by baroclinic eddies at a deformation radius of ≈ 8 km, which an eddy-permitting 1/4° grid does not resolve. The
# latitude spacing is the longitude spacing times cos 60°, so cells are square at 60°S (Δx = Δy ≈ 14 km at 1/4°).
# A fine run (truth) is compared to 1/4° runs with and without EnergyBudgetBackscatter.
#
# Usage: ARCH=GPU julia --project baroclinic_channel.jl, then julia --project plot_baroclinic_channel.jl with the
# same environment variables (the plot script needs CairoMakie and FFTW). Each run saves surface snapshots every 5 days
# and the overturning averaged over the second half of the run to baroclinic_channel_<settings>_<case>_*.jld2.
#
# Environment variables: TRUTH_DEGREES (default 1/32), COARSE_DEGREES (1/4), BIHARMONIC_DAYS (0.3), STOP_YEARS (4),
# WENO (false: WENO=true swaps the energy-conserving momentum advection for WENO).

using Oceananigans
using Oceananigans.Units
using Oceananigans: UpdateStateCallsite
using Oceananigans.Advection: EnergyConserving
using Oceananigans.Operators: ℑxyᶠᶜᵃ, ℑxyᶜᶠᵃ, ℑyᵃᶠᵃ, Δxᶜᶜᶜ, Δyᶜᶜᶜ, Δzᶠᶜᶜ, Δzᶜᶠᶜ, ζ₃ᶠᶠᶜ
using Printf

if get(ENV, "ARCH", "CPU") == "GPU"
    using CUDA
end

arch = get(ENV, "ARCH", "CPU") == "GPU" ? GPU() : CPU()
parse_degrees(s) = prod(parse(Float64, n)^p for (n, p) in zip(split(s, "/"), (1, -1)))
truth_degrees = parse_degrees(get(ENV, "TRUTH_DEGREES", "1/32"))
coarse_degrees = parse_degrees(get(ENV, "COARSE_DEGREES", "1/4"))
τ₄ = parse(Float64, get(ENV, "BIHARMONIC_DAYS", "0.3")) * day   # biharmonic damping time of the grid-scale wave
stop_time = parse(Float64, get(ENV, "STOP_YEARS", "4")) * 365days
weno = parse(Bool, get(ENV, "WENO", "false"))
prefix = @sprintf("baroclinic_channel_%gday", τ₄ / day) * (weno ? "_weno" : "")

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

# ν₄ = (Δ / π)⁴ / τ₄ with Δ the longer side of the cell, so τ₄ is the damping time in the worse-resolved direction
@inline biharmonic_viscosity(i, j, k, grid, clock, fields, p) = max(Δxᶜᶜᶜ(i, j, k, grid), Δyᶜᶜᶜ(i, j, k, grid))^4 / (π^4 * p.τ₄)

@inline upper_layer_v(i, j, k, grid, v, b, interface_buoyancy) = @inbounds v[i, j, k] * (ℑyᵃᶠᵃ(i, j, k, grid, b) > interface_buoyancy)

function run_channel(name, degrees; backscatter=false)
    Nλ = round(Int, (longitude[2] - longitude[1]) / degrees)
    Nφ = round(Int, (latitude[2] - latitude[1]) / (degrees * cosd(60)))
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
    Nz = grid.Nz
    V₁ = Integral(KernelFunctionOperation{Center, Face, Center}(upper_layer_v, grid, v, b, Δb / 2), dims=3)
    ζ = KernelFunctionOperation{Face, Face, Center}(ζ₃ᶠᶠᶜ, grid, u, v)

    surface = (u = view(u, :, :, Nz), v = view(v, :, :, Nz), ζ = Field(ζ, indices=(:, :, Nz)))
    backscatter && (surface = merge(surface, (; e = model.closure_fields[2].e)))

    simulation.output_writers[:surface] = JLD2Writer(model, surface; filename = name * "_surface",
                                                     schedule = TimeInterval(5days), overwrite_files = true)

    simulation.output_writers[:averages] = JLD2Writer(model, (; V₁); filename = name * "_averages",
                                                      schedule = AveragedTimeInterval(stop_time / 2, window = stop_time / 2),
                                                      overwrite_files = true)

    progress(sim) = @info @sprintf("%s: %s, max|u| = %.2e m s⁻¹, wall time: %s", name, prettytime(sim),
                                   maximum(abs, u), prettytime(sim.run_wall_time))

    add_callback!(simulation, progress, TimeInterval(100days))
    run!(simulation)

    return nothing
end

run_channel(prefix * "_truth", truth_degrees)
run_channel(prefix * "_coarse", coarse_degrees)
run_channel(prefix * "_coarse_backscatter", coarse_degrees; backscatter=true)
