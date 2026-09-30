# Swimmer-induced mixing in a stratified column, following Houghton, Koseff, Monismith
# & Dabiri (2018), "Vertically migrating swimmers generate aggregation-scale eddies in
# a stratified column", Nature 556, 497-500, and the continuum modeling of Ouillon,
# Houghton, Dabiri & Meiburg (2020), J. Fluid Mech. 902, A23.
#
# A dilute aggregation of slightly negatively buoyant swimmers (brine shrimp) migrates
# upward through a salt-stratified tank (two nearly homogeneous layers separated by an
# erf pycnocline, as in the experiments), driven in the laboratory by phototaxis
# toward a top light and modeled here as a prescribed upward swimming speed Vs that
# tapers to zero at the lid. The net far-field force each steadily swimming animal
# exerts on the fluid is its excess weight, downward: thrust and drag cancel into a
# force dipole, leaving the negative-buoyancy monopole. The aggregation therefore
# forces the momentum equation with -γ n, driving an aggregation-scale downward jet
# (~1-2 cm/s observed) whose eddies irreversibly mix the stratification with an
# effective diffusivity far above molecular.
#
# Run with: julia --project validation/bioconvection/migrating_swimmers_stratified.jl

using Oceananigans
using Oceananigans.Units
using Oceananigans.BoundaryConditions: fill_halo_regions!, ImpenetrableBoundaryCondition
using SpecialFunctions: erf
using JLD2: jldsave
using Printf
using Random
using Statistics

# Tank and stratification from Houghton et al. (2018): 0.5 m cross-section, 1.2 m tall,
# two nearly homogeneous salt layers separated by an erf pycnocline with interface
# buoyancy frequency N = 0.1 s⁻¹. Molecular salt diffusivity (1.5e-9, Sc ≈ 700) is
# unresolvable, so κb = 1e-7 and diffusivity enhancement is reported relative to κb.
Lx = 0.5    # tank width (m)
Lz = 1.2    # tank depth (m)
N² = 0.01   # interface buoyancy frequency squared (s⁻²)
δ  = 0.1    # pycnocline half-thickness (m)
zᵖ = -Lz/2  # pycnocline depth (m)
ν  = 1e-6   # kinematic viscosity (m² s⁻¹)
κb = 1e-7   # buoyancy (salt) diffusivity (m² s⁻¹)
κn = 5e-6   # swimmer dispersal diffusivity (m² s⁻¹)
Vs = 1e-2   # Artemia upward swimming speed (m s⁻¹)
h  = 0.05   # swimming shutoff scale beneath the lid (m)
g  = 9.81   # gravitational acceleration (m s⁻²)
animal_volume = 5e-9         # m³, ~1 cm Artemia
density_excess_ratio = 0.03  # Δρ/ρ₀
n₀ = 1.5e6                   # peak aggregation number density (animals m⁻³)
σx = 0.04   # initial aggregation half-width (m)
σz = 0.06   # initial aggregation half-height (m)
x₀ = Lx / 2
z₀ = -1.05

# Excess weight per animal per unit density: the forcing takes the same form as the
# gyrotactic bioconvection case, F_w = -γ n.
γ = g * animal_volume * density_excess_ratio

# The erf slope at the interface equals N²
Δb = √π * N² * δ

a = γ * n₀
jet_scale_speed    = sqrt(a * σx)
arrest_scale_speed = a / sqrt(N²)
penetration_scale  = a / N²

@info @sprintf("Migrating swimmers: γn₀ = %.2e m s⁻², jet scale √(γn₀σx) = %.2f cm/s, arrest scale γn₀/N = %.2f cm/s, penetration γn₀/N² = %.1f cm",
               a, 100jet_scale_speed, 100arrest_scale_speed, 100penetration_scale)

grid = RectilinearGrid(size = (256, 512), x = (0, Lx), z = (-Lz, 0),
                       topology = (Periodic, Flat, Bounded), halo = (4, 4))

# The impenetrable lids zero the swimming flux through the boundaries during halo
# fills; together with the default no-flux tracer conditions this conserves ∫n.
no_penetration = ImpenetrableBoundaryCondition()
swimming_bcs = FieldBoundaryConditions(grid, (Center(), Center(), Face()),
                                       top = no_penetration, bottom = no_penetration)
wˢ = ZFaceField(grid, boundary_conditions = swimming_bcs)

# The tanh taper models animals parking at the light; a constant Vs would pile the
# swimmers into an equilibrium lid layer of width κn/Vs ≈ 0.5 mm ≪ Δz.
set!(wˢ, (x, z) -> Vs * tanh(-z / h))
fill_halo_regions!(wˢ)

swimming = AdvectiveForcing(w = wˢ)

# With periodic x and rigid lids continuity forces ⟨w⟩(z) ≡ 0, so the horizontally
# uniform part of -γn is absorbed by the pressure; no mean-forcing subtraction needed.
swimmer_gravity(x, z, t, n, γ) = -γ * n
buoyancy_forcing = Forcing(swimmer_gravity, field_dependencies = :n, parameters = γ)

model = NonhydrostaticModel(grid;
                            tracers = (:b, :n),
                            buoyancy = BuoyancyTracer(),
                            advection = WENO(order = 5),
                            closure = ScalarDiffusivity(ν = ν, κ = (b = κb, n = κn)),
                            forcing = (n = swimming, w = buoyancy_forcing))

rng = Xoshiro(42)
bᵢ(x, z) = Δb/2 * erf((z - zᵖ) / δ)
nᵢ(x, z) = n₀ * exp(-(x - x₀)^2 / 2σx^2 - (z - z₀)^2 / 2σz^2) * (1 + 0.02 * (2rand(rng) - 1))
set!(model, b = bᵢ, n = nᵢ)

u, v, w = model.velocities
b = model.tracers.b
n = model.tracers.n

Δz = minimum_zspacing(grid)

# The wizard's advective CFL never sees the swimming velocity wˢ, so its cap must be
# imposed through max_Δt; diffusive_cfl = 0.2 respects the RK3 explicit-diffusion limit.
swimming_time_step = 0.2 * Δz / Vs

simulation = Simulation(model; Δt = swimming_time_step / 5, stop_time = 15minutes)

# The jet spins up in seconds, so the wizard updates every 5 iterations with
# max_change = 1.1 to track it from the reduced initial Δt.
wizard = TimeStepWizard(cfl = 0.5, diffusive_cfl = 0.2, max_change = 1.1,
                        max_Δt = swimming_time_step)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(5))

ω = Field(∂z(u) - ∂x(w))

# Pseudo-dissipation ν⟨|∇u|² + |∇w|²⟩, adequate for a mixing-efficiency estimate.
ϵ = Field(@at (Center, Center, Center) ν * (∂x(u)^2 + ∂z(u)^2 + ∂x(w)^2 + ∂z(w)^2))
E = Field(Average(ϵ))

B  = Field(Average(b, dims = 1))
W  = Field(Average(w, dims = 1))
WB = Field(Average(w * b, dims = 1))
N̄  = Field(Average(n, dims = 1))

total_swimmers = Field(Integral(n))
compute!(total_swimmers)
initial_total_swimmers = total_swimmers[1, 1, 1]

Nx, Nz = size(grid, 1), size(grid, 3)

# Uniform cells: ascending sorted index maps to height slots of thickness Lz/(Nx Nz).
Δz★ = Lz / (Nx * Nz)
z★  = collect(range(-Lz + Δz★/2, step = Δz★, length = Nx * Nz))
zᶜ  = reshape(znodes(grid, Center()), 1, Nz)

energetics_times = Float64[]
background_potential_energies = Float64[]
available_potential_energies = Float64[]
dissipation_rates = Float64[]
diffusive_bpe_fluxes = Float64[]

# Winters et al. (1995) bookkeeping: dBPE/dt = Φd + Φᵢ with diffusive flux
# Φᵢ = κb (b★max - b★min)/Lz, so the irreversible mixing rate Φd is diagnosed
# a posteriori by finite-differencing the BPE series.
function compute_energetics!(sim)
    b² = Array(interior(b, :, 1, :))
    sorted = sort!(vec(b²))
    bpe = -mean(sorted .* z★)
    pe  = -mean(b² .* zᶜ)
    Φᵢ  = κb * (sorted[end] - sorted[1]) / Lz
    compute!(ϵ)
    compute!(E)
    push!(energetics_times, time(sim))
    push!(background_potential_energies, bpe)
    push!(available_potential_energies, pe - bpe)
    push!(dissipation_rates, E[1, 1, 1])
    push!(diffusive_bpe_fluxes, Φᵢ)
    return nothing
end

simulation.callbacks[:energetics] = Callback(compute_energetics!, IterationInterval(20))

# Normalizing the turbulent flux by κb N² rather than the local gradient avoids
# division by near-zero gradients in mixed patches.
function flux_enhancement()
    compute!(B); compute!(W); compute!(WB)
    Bᶜ = interior(B, 1, 1, :)
    enhancement = 0.0
    for k in 2:Nz
        b̄ᶠ = (Bᶜ[k-1] + Bᶜ[k]) / 2
        w′b′ = WB[1, 1, k] - W[1, 1, k] * b̄ᶠ
        enhancement = max(enhancement, -w′b′ / (κb * N²))
    end
    return enhancement
end

jet_times = Float64[]
jet_speeds = Float64[]
flux_enhancements = Float64[]

function progress(sim)
    compute!(total_swimmers)
    conservation_error = abs(total_swimmers[1, 1, 1] - initial_total_swimmers) / initial_total_swimmers
    push!(jet_times, time(sim))
    push!(jet_speeds, -minimum(w))
    push!(flux_enhancements, flux_enhancement())
    @info @sprintf("iter %d, t = %.1f s, Δt = %.3f s, jet = %.2f cm/s, max n/n₀ = %.2f, flux enhancement = %.1f, Δ∫n/∫n = %.2e",
                   iteration(sim), time(sim), sim.Δt, 100jet_speeds[end],
                   maximum(n) / n₀, flux_enhancements[end], conservation_error)
    return nothing
end

simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))

prefix = joinpath(@__DIR__, "migrating_swimmers_stratified")

simulation.output_writers[:profiles] = JLD2Writer(model, (; B, W, WB, N̄);
                                                  filename = prefix * "_profiles.jld2",
                                                  schedule = TimeInterval(2),
                                                  overwrite_files = true)

# ~1 MB per snapshot per field: the fields file reaches ~250 MB at 15 s cadence.
simulation.output_writers[:fields] = JLD2Writer(model, (; n, w, b, ω);
                                                filename = prefix * "_fields.jld2",
                                                schedule = TimeInterval(15),
                                                overwrite_files = true)

run!(simulation)

jldsave(prefix * "_energetics.jld2";
        times = energetics_times,
        background_potential_energies,
        available_potential_energies,
        dissipation_rates,
        diffusive_bpe_fluxes,
        jet_times, jet_speeds, flux_enhancements)

compute!(total_swimmers)
conservation_error = abs(total_swimmers[1, 1, 1] - initial_total_swimmers) / initial_total_swimmers
@info @sprintf("Relative swimmer conservation error: %.2e", conservation_error)
@assert conservation_error < 1e-10

# Convection of the accumulated layer stirs swimmers through the unstratified upper
# layer, so assert arrival in the upper layer rather than a thin lid peak.
compute!(N̄)
n̄ = interior(N̄, 1, 1, :)
zc = znodes(grid, Center())
swimmer_center_of_mass = sum(n̄ .* zc) / sum(n̄)
upper_layer_fraction = sum(n̄[zc .> zᵖ]) / sum(n̄)
@info @sprintf("Final swimmer center of mass at z = %.2f m, fraction above the pycnocline %.2f",
               swimmer_center_of_mass, upper_layer_fraction)
@assert swimmer_center_of_mass > zᵖ + 0.1
@assert upper_layer_fraction > 0.8

# Compare against the observed 1-2 cm/s during the migration phase only: once the
# swimmers pile up under the lid (t ≳ 200 s) convection of that dense layer drives
# faster transients unrelated to the migration jet.
migration_phase = jet_times .< 200
peak_migration_jet = maximum(jet_speeds[migration_phase])
peak_jet_speed = maximum(jet_speeds)
@info @sprintf("Peak downward jet: %.2f cm/s during migration, %.2f cm/s overall (observed 1-2 cm/s)",
               100peak_migration_jet, 100peak_jet_speed)
@assert 0.005 < peak_migration_jet < 0.05
@assert peak_jet_speed < 0.15

diffusive_bpe_rise = sum((diffusive_bpe_fluxes[i] + diffusive_bpe_fluxes[i+1]) / 2 *
                         (energetics_times[i+1] - energetics_times[i])
                         for i in 1:length(energetics_times)-1)
Δbpe = background_potential_energies[end] - background_potential_energies[1]
@info @sprintf("ΔBPE = %.2e m² s⁻², diffusive-only rise ∫Φᵢdt = %.2e m² s⁻²", Δbpe, diffusive_bpe_rise)
@assert Δbpe > 2 * diffusive_bpe_rise

peak_flux_enhancement = maximum(flux_enhancements)
@info @sprintf("Peak turbulent flux enhancement -⟨w′b′⟩/(κb N²): %.0f", peak_flux_enhancement)
@assert peak_flux_enhancement > 10

# The convecting upper layer erodes the pycnocline by entrainment: it homogenizes and
# densifies while the interface thins and descends with its peak N² preserved (an
# entrainment interface, not diffusive smearing), the resolved counterpart of the
# erosion Houghton et al. measured. Assert the upper-layer buoyancy deficit.
compute!(B)
Bᶜ = interior(B, 1, 1, :)
zf = znodes(grid, Face())
N²final = [(Bᶜ[k] - Bᶜ[k-1]) / (zc[k] - zc[k-1]) for k in 2:Nz]
interface = findall(k -> abs(zf[k] - zᵖ) < 3δ, 2:Nz)
upper_layer = zc .> zᵖ + 1.5δ
entrainment_fraction = (Δb/2 - mean(Bᶜ[upper_layer])) / (Δb/2)
@info @sprintf("Final peak interface N²/N²₀ = %.2f, upper-layer entrainment fraction = %.2f",
               maximum(N²final[interface]) / N², entrainment_fraction)
@assert entrainment_fraction > 0.1

using CairoMakie

nt = FieldTimeSeries(prefix * "_fields.jld2", "n")
wt = FieldTimeSeries(prefix * "_fields.jld2", "w")
bt = FieldTimeSeries(prefix * "_fields.jld2", "b")
ωt = FieldTimeSeries(prefix * "_fields.jld2", "ω")
times = nt.times
xn, yn, zn = nodes(nt)
xw, yw, zw = nodes(wt)
xω, yω, zω = nodes(ωt)

fig = Figure(size = (1300, 500))
axn = Axis(fig[1, 2], xlabel = "x (m)", ylabel = "z (m)")
axw = Axis(fig[1, 3], xlabel = "x (m)")
axb = Axis(fig[1, 4], xlabel = "x (m)")
axω = Axis(fig[1, 5], xlabel = "x (m)")

frame = Observable(1)
b⁰  = interior(bt[1], :, 1, :)
nk  = @lift interior(nt[$frame], :, 1, :) ./ n₀
wk  = @lift interior(wt[$frame], :, 1, :)
b′k = @lift interior(bt[$frame], :, 1, :) .- b⁰
ωk  = @lift interior(ωt[$frame], :, 1, :)

hn = heatmap!(axn, xn, zn, nk, colormap = :dense, colorrange = (0, 2))
hw = heatmap!(axw, xw, zw, wk, colormap = :balance, colorrange = (-2e-2, 2e-2))
hb = heatmap!(axb, xn, zn, b′k, colormap = :curl, colorrange = (-1e-3, 1e-3))
hω = heatmap!(axω, xω, zω, ωk, colormap = :vik, colorrange = (-1, 1))
Colorbar(fig[1, 1], hn, label = "n / n₀", flipaxis = false)
Colorbar(fig[1, 6], hω, label = "ω (s⁻¹)")

CairoMakie.record(fig, prefix * ".mp4", 1:length(times), framerate = 12) do i
    frame[] = i
    axn.title = @sprintf("n/n₀, t = %.1f min", times[i] / 60)
    axw.title = "w"
    axb.title = "b - b₀"
    axω.title = "ω"
end

@info "Wrote migrating_swimmers_stratified.mp4"
