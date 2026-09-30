# Gyrotactic bioconvection following the continuum model of Pedley & Kessler (1992),
# "Hydrodynamic phenomena in suspensions of swimming microorganisms",
# Annu. Rev. Fluid Mech. 24, 313-358.
#
# A dilute suspension of bottom-heavy swimming cells with concentration n is advected
# by the resolved flow plus a swimming velocity Vs p̂. The orientation p̂ obeys the
# quasi-steady gyrotactic balance between the gravitational righting torque and the
# viscous torque: pˣ = B ωʸ (clamped to the tumbling limit |B ωʸ| = 1) and
# pᶻ = √(1 - pˣ²). Cells are slightly denser than water and force the vertical
# momentum with -γ n. Up-swimming concentrates cells beneath the rigid lid and the
# resulting top-heavy sublayer overturns into descending gyrotactic plumes.
#
# Run with: julia --project validation/bioconvection/gyrotactic_bioconvection.jl

using Oceananigans
using Oceananigans.Units
using Oceananigans: UpdateStateCallsite
using Oceananigans.BoundaryConditions: fill_halo_regions!, ImpenetrableBoundaryCondition
using Oceananigans.Operators: ∂zᶠᶜᶠ, ∂xᶠᶜᶠ, ℑzᵃᵃᶜ, ℑxᶜᵃᵃ
using Printf
using Random
using Statistics

# Chlamydomonas-like parameters from Pedley & Kessler (1992)
H  = 5e-3   # layer depth (m)
Lx = 2H     # domain width (m)
ν  = 1e-6   # kinematic viscosity (m² s⁻¹)
D  = 5e-8   # effective cell diffusivity (m² s⁻¹)
Vs = 1e-4   # swimming speed (m s⁻¹)
B  = 3.0    # gyrotactic reorientation timescale (s)
n₀ = 1e12   # mean cell concentration (cells m⁻³)
g  = 9.81   # gravitational acceleration (m s⁻²)
cell_volume = 5e-16          # m³, radius ≈ 5 μm
density_excess_ratio = 0.05  # Δρ/ρ₀

# Buoyancy force per unit cell concentration (m⁴ s⁻²)
γ = g * cell_volume * density_excess_ratio

Sc = ν / D               # Schmidt number
Pe = Vs * H / D          # swimming Péclet number
G  = B * Vs / H          # gyrotaxis number
Ra = γ * n₀ * H^3 / (ν * D)  # bioconvection Rayleigh number

@info @sprintf("Gyrotactic bioconvection: Sc = %.1f, Pe = %.1f, G = %.3f, Ra = %.2e",
               Sc, Pe, G, Ra)

grid = RectilinearGrid(size = (128, 64), x = (0, Lx), z = (-H, 0),
                       topology = (Periodic, Flat, Bounded), halo = (4, 4))

# Quasi-steady gyrotactic orientation. ωʸ = ∂u/∂z - ∂w/∂x lives naturally at
# (Face, Center, Face); interpolate ω to the target location *before* applying the
# nonlinear closure so the clamp/sqrt never straddle the tumbling boundary |Bω| = 1.
@inline ωᶠᶜᶠ(i, j, k, grid, u, w) = ∂zᶠᶜᶠ(i, j, k, grid, u) - ∂xᶠᶜᶠ(i, j, k, grid, w)

@inline function uˢᶠᶜᶜ(i, j, k, grid, u, w, p)
    ω  = ℑzᵃᵃᶜ(i, j, k, grid, ωᶠᶜᶠ, u, w)
    pˣ = clamp(p.B * ω, -one(grid), one(grid))
    return p.Vs * pˣ
end

@inline function wˢᶜᶜᶠ(i, j, k, grid, u, w, p)
    ω  = ℑxᶜᵃᵃ(i, j, k, grid, ωᶠᶜᶠ, u, w)
    pˣ = clamp(p.B * ω, -one(grid), one(grid))
    return p.Vs * sqrt(one(grid) - pˣ^2)
end

# The impenetrable lids zero the swimming flux through the boundaries during halo
# fills; together with the default no-flux tracer conditions this conserves ∫n.
no_penetration = ImpenetrableBoundaryCondition()
swimming_bcs = FieldBoundaryConditions(grid, (Center(), Center(), Face()),
                                       top = no_penetration, bottom = no_penetration)
uˢ = XFaceField(grid)
wˢ = ZFaceField(grid, boundary_conditions = swimming_bcs)

swimming = AdvectiveForcing(u = uˢ, w = wˢ)
cell_buoyancy(x, z, t, n, γ) = -γ * n
buoyancy_forcing = Forcing(cell_buoyancy, field_dependencies = :n, parameters = γ)

model = NonhydrostaticModel(grid;
                            tracers = :n,
                            advection = WENO(order = 5),
                            closure = ScalarDiffusivity(ν = ν, κ = (; n = D)),
                            forcing = (n = swimming, w = buoyancy_forcing))

# Uniform suspension with 1% seeded noise. An alternative that skips the initial
# accumulation transient is the up-swimming equilibrium profile
# nᵢ(x, z) = n₀ * Pe * exp(Pe * (z / H + 1)) / (exp(Pe) - 1) * (1 + noise).
rng = Xoshiro(42)
nᵢ(x, z) = n₀ * (1 + 0.01 * (2rand(rng) - 1))
set!(model, n = nᵢ)

u, v, w = model.velocities
n = model.tracers.n

swimming_parameters = (; B, Vs)
𝒰ˢ = KernelFunctionOperation{Face, Center, Center}(uˢᶠᶜᶜ, grid, u, w, swimming_parameters)
𝒲ˢ = KernelFunctionOperation{Center, Center, Face}(wˢᶜᶜᶠ, grid, u, w, swimming_parameters)

function compute_swimming_velocities!(args...)
    uˢ .= 𝒰ˢ
    wˢ .= 𝒲ˢ
    fill_halo_regions!(uˢ)
    fill_halo_regions!(wˢ)
    return nothing
end

compute_swimming_velocities!()

Δz = minimum_zspacing(grid)
diffusive_time_step = 0.2 * Δz^2 / ν
swimming_time_step  = 0.2 * Δz / Vs

simulation = Simulation(model; Δt = min(diffusive_time_step, swimming_time_step),
                        stop_time = 5minutes)

# UpdateStateCallsite recomputes the drift at every RK3 substage, synchronous with the
# substage velocities. TimeStepCallsite (the default) is cheaper but lags by O(Δt).
simulation.callbacks[:swimming] = Callback(compute_swimming_velocities!, IterationInterval(1),
                                           callsite = UpdateStateCallsite())

# The wizard's advective CFL only sees the prognostic velocities, so the cap from the
# swimming speed must be imposed through max_Δt. diffusive_cfl = 0.2 keeps Δt below the
# RK3 explicit-diffusion stability limit 2.51 Δz² / (8ν) ≈ 0.31 Δz²/ν on this square grid.
wizard = TimeStepWizard(cfl = 0.5, diffusive_cfl = 0.2, max_change = 1.1,
                        max_Δt = swimming_time_step)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

ω  = Field(∂z(u) - ∂x(w))
pˣ = Field(KernelFunctionOperation{Face, Center, Center}(uˢᶠᶜᶜ, grid, u, w, (B = B, Vs = 1.0)))

total_cells = Field(Integral(n))
compute!(total_cells)
initial_total_cells = total_cells[1, 1, 1]

depth_averaged_concentration = Field(Average(n, dims = 3))
compute!(depth_averaged_concentration)
initial_pattern_variance = var(interior(depth_averaged_concentration, :, 1, 1))

growth_times = Float64[]
growth_energies = Float64[]

function progress(sim)
    compute!(total_cells)
    conservation_error = abs(total_cells[1, 1, 1] - initial_total_cells) / initial_total_cells
    vertical_kinetic_energy = mean(interior(w).^2)
    push!(growth_times, time(sim))
    push!(growth_energies, vertical_kinetic_energy)
    @info @sprintf("iter %d, t = %.1f s, Δt = %.2e s, max|w| = %.2e m/s, max n/n₀ = %.2f, Δ∫n/∫n = %.2e",
                   iteration(sim), time(sim), sim.Δt, maximum(abs, w),
                   maximum(n) / n₀, conservation_error)
    return nothing
end

simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))

filename = joinpath(@__DIR__, "gyrotactic_bioconvection.jld2")
simulation.output_writers[:fields] = JLD2Writer(model, (; n, w, ω, pˣ);
                                                filename,
                                                schedule = TimeInterval(2),
                                                overwrite_files = true)

run!(simulation)

compute!(total_cells)
conservation_error = abs(total_cells[1, 1, 1] - initial_total_cells) / initial_total_cells
@info @sprintf("Relative cell conservation error: %.2e", conservation_error)
@assert conservation_error < 1e-10

@info @sprintf("Final max n/n₀ = %.2f (surface accumulation + plume focusing)", maximum(n) / n₀)
@assert maximum(n) / n₀ > 2

compute!(depth_averaged_concentration)
final_pattern_variance = var(interior(depth_averaged_concentration, :, 1, 1))
@info @sprintf("Horizontal pattern variance: initial %.2e, final %.2e", initial_pattern_variance, final_pattern_variance)
@assert final_pattern_variance > 10 * initial_pattern_variance

# Growth rate of the instability from the exponential phase of w² ∝ exp(2σt)
peak_energy = maximum(growth_energies)
window = [i for i in eachindex(growth_energies) if 1e-3 * peak_energy < growth_energies[i] < 0.1 * peak_energy]
if length(window) > 2
    t = growth_times[window]
    ε = log.(growth_energies[window])
    slope = cov(t, ε) / var(t)
    @info @sprintf("Bioconvective growth rate σ ≈ %.3f s⁻¹ (e-folding time %.1f s)", slope / 2, 2 / slope)
else
    @warn "Too few samples in the exponential-growth window for a growth-rate fit"
end

using CairoMakie

nt = FieldTimeSeries(filename, "n")
wt = FieldTimeSeries(filename, "w")
times = nt.times
xn, yn, zn = nodes(nt)
xw, yw, zw = nodes(wt)

fig = Figure(size = (1100, 450))
axn = Axis(fig[1, 2], xlabel = "x (m)", ylabel = "z (m)")
axw = Axis(fig[1, 3], xlabel = "x (m)")

frame = Observable(1)
nk = @lift interior(nt[$frame], :, 1, :) ./ n₀
wk = @lift interior(wt[$frame], :, 1, :)

hn = heatmap!(axn, xn, zn, nk, colormap = :dense, colorrange = (0, 3))
hw = heatmap!(axw, xw, zw, wk, colormap = :balance, colorrange = (-2e-3, 2e-3))
Colorbar(fig[1, 1], hn, label = "n / n₀", flipaxis = false)
Colorbar(fig[1, 4], hw, label = "w (m s⁻¹)")

CairoMakie.record(fig, joinpath(@__DIR__, "gyrotactic_bioconvection.mp4"),
                  1:length(times), framerate = 12) do i
    frame[] = i
    axn.title = @sprintf("n/n₀, t = %.0f s", times[i])
    axw.title = "w"
end

@info "Wrote gyrotactic_bioconvection.mp4"
