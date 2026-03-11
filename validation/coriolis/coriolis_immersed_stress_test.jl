# # Coriolis scheme stress test with complex immersed boundaries
#
# A stress test comparing all five Coriolis discretization schemes in a
# domain with deliberately hostile topography: an archipelago with narrow
# straits, sharp capes, isolated islands, and a jagged continental shelf.
# A strong barotropic jet impinges on this topography, exciting all the
# problematic interactions between Coriolis interpolation and immersed
# boundary masking.
#
# The test tries to find:
# - Spurious velocity amplification near masked stencil nodes
# - Energy injection by active-weighted schemes at coastlines
# - Scheme-dependent SSH biases downstream of topography
# - Blow-ups / NaNs from pathological active-node configurations

using Oceananigans
using Oceananigans.Units
using Oceananigans.Coriolis
using Oceananigans.Advection: EnstrophyConserving, EnergyConserving
using Oceananigans.ImmersedBoundaries: inactive_cell
using Oceananigans.OutputReaders: FieldTimeSeries

using Printf
using CairoMakie
using JLD2

#####
##### Domain and topography
#####

# 1/2-degree LatLon grid spanning 30-60N, roughly the Southern Ocean / Australia belt
Nx, Ny, Nz = 60, 60, 1
H = 500.0  # depth [m]

grid = LatitudeLongitudeGrid(CPU(),
                             size = (Nx, Ny, Nz),
                             latitude  = (30, 60),
                             longitude = (0, 30),
                             halo = (3, 3, 3),
                             z = MutableVerticalDiscretization((-H, 0)),
                             topology = (Bounded, Bounded, Bounded))

# Stress-inducing topography: multiple features combined
#   1. Continental shelf along the south (land for φ < 35)
#   2. Large island (mimicking Australia/NZ blocking a zonal jet)
#   3. Narrow strait through the island (2 cells wide)
#   4. Small isolated island north of the continent
#   5. A thin cape / peninsula protruding northward
#   6. Jagged shelf edge with single-cell inlets

function stress_bottom(λ, φ)
    # Start with deep ocean
    z = -H

    # 1. Southern continent (φ < 35)
    φ < 35 && return 0.0

    # 2. Large island: 10 < λ < 20, 40 < φ < 50
    #    with a narrow strait at λ = 14-15, φ = 44-46
    in_island = (10 < λ < 20) && (40 < φ < 50)
    in_strait = (13.5 < λ < 15.0) && (44 < φ < 46)
    in_island && !in_strait && return 0.0

    # 3. Small isolated island: 5 < λ < 7, 44 < φ < 46
    (5 < λ < 7) && (44 < φ < 46) && return 0.0

    # 4. Tiny single-cell island at (25, 48)
    (24.5 < λ < 25.5) && (47.5 < φ < 48.5) && return 0.0

    # 5. Thin cape from continent extending north at λ = 3
    (2.5 < λ < 3.5) && (35 < φ < 42) && return 0.0

    # 6. Jagged shelf: periodic inlets along the continental shelf edge
    #    at φ ≈ 35-37 with period ~2 degrees in longitude
    if 35 ≤ φ ≤ 37
        # Every 4 degrees in λ, there is a 1-degree-wide inlet
        λ_mod = mod(λ, 4.0)
        (λ_mod < 1.0) && return -H  # inlet (ocean)
        return 0.0  # shelf (land)
    end

    return z
end

ib_grid = ImmersedBoundaryGrid(grid, GridFittedBottom(stress_bottom))

#####
##### Physics
#####

# Light bottom drag for stability
κ_drag = 1e-4
@inline u_drag(i, j, grid, clock, fields, κ) = @inbounds -κ * fields.u[i, j, 1]
@inline v_drag(i, j, grid, clock, fields, κ) = @inbounds -κ * fields.v[i, j, 1]

u_bcs = FieldBoundaryConditions(bottom = FluxBoundaryCondition(u_drag, discrete_form=true, parameters=κ_drag))
v_bcs = FieldBoundaryConditions(bottom = FluxBoundaryCondition(v_drag, discrete_form=true, parameters=κ_drag))

# Small viscosity
closure = VerticalScalarDiffusivity(VerticallyImplicitTimeDiscretization(), ν=1e-2)

#####
##### Initial condition: strong barotropic jet + mesoscale eddy
#####

# Zonal jet centered at φ = 45 (hits the island head-on)
function u_init(λ, φ, z)
    φ_jet = 45.0
    σ_jet = 3.0
    U_jet = 0.5  # m/s - strong enough to stress the schemes
    return U_jet * exp(-(φ - φ_jet)^2 / (2 * σ_jet^2))
end

# Anticyclonic eddy west of the island to create cross-jet flow
function v_init(λ, φ, z)
    λ₀, φ₀ = 7.0, 45.0
    σ = 2.5
    V₀ = 0.2
    r² = (λ - λ₀)^2 + (φ - φ₀)^2
    return -V₀ * (λ - λ₀) / σ * exp(-r² / (2σ^2))
end

#####
##### Run all schemes
#####

import Oceananigans.Operators: ζ₃ᶠᶠᶜ
using Oceananigans.Operators: Γᶠᶠᶜ, Az⁻¹ᶠᶠᶜ
using Oceananigans.ImmersedBoundaries: peripheral_node

@inline ζ₃ᶠᶠᶜ(i, j, k, grid, u, v) = ifelse(peripheral_node(i, j, k, grid, Face(), Face(), Center()), zero(grid),
                                            Γᶠᶠᶜ(i, j, k, grid, u, v) * Az⁻¹ᶠᶠᶜ(i, j, k, grid))

schemes = [
    (EnstrophyConserving(),                "ES"),
    (EnergyConserving(),                   "EN"),
    (EENConserving(),                      "EEN"),
    (ActiveWeightedEnstrophyConserving(),  "AWES"),
    (ActiveWeightedEnergyConserving(),     "AWEN"),
]

Δt = 300.0  # 5 min timestep
stop_time = 30days
save_interval = 12hours

output_dir = "coriolis_stress_test_output"
mkpath(output_dir)

function run_stress_test(grid, scheme; label, Δt, stop_time, save_interval, output_dir)
    coriolis = HydrostaticSphericalCoriolis(scheme=scheme)
    free_surface = SplitExplicitFreeSurface(grid; substeps=30)

    model = HydrostaticFreeSurfaceModel(grid;
                                        coriolis,
                                        closure,
                                        free_surface,
                                        momentum_advection  = VectorInvariant(),
                                        tracer_advection    = nothing,
                                        tracers             = (),
                                        timestepper         = :SplitRungeKutta3,
                                        buoyancy            = nothing,
                                        boundary_conditions = (; u=u_bcs, v=v_bcs))

    set!(model, u=u_init, v=v_init)

    simulation = Simulation(model; Δt, stop_time)

    wall_clock = Ref(time_ns())

    function progress(sim)
        u_max = maximum(abs, sim.model.velocities.u)
        v_max = maximum(abs, sim.model.velocities.v)
        elapsed = (time_ns() - wall_clock[]) * 1e-9
        @info @sprintf("[%5s] t=%s iter=%d max|u|=%.3e max|v|=%.3e (%.1fs)",
                       label, prettytime(sim.model.clock.time),
                       sim.model.clock.iteration, u_max, v_max, elapsed)
        wall_clock[] = time_ns()

        # Early abort on blow-up
        if !isfinite(u_max) || !isfinite(v_max) || u_max > 100 || v_max > 100
            @warn "$label: BLOW-UP detected at t=$(prettytime(sim.model.clock.time))!"
            simulation.running = false
        end
    end

    simulation.callbacks[:progress] = Callback(progress, IterationInterval(500))

    # Save surface fields for video
    η = model.free_surface.displacement
    outputs = (; η, u=model.velocities.u, v=model.velocities.v)

    # KE comparison
    println("\nKinetic energy at initial time:")
    u, v, w = model.velocities
    KE_op = @at (Center, Center, Center) (u^2 + v^2) / 2
    KE_field = Field(KE_op)
    compute!(KE_field)
    KE = sum(KE_field)
    @printf("  %-6s: KE = %.6e\n", label, KE)
    println("="^80)

    simulation.output_writers[:fields] = JLD2Writer(model, outputs;
                                                    schedule = TimeInterval(save_interval),
                                                    filename = joinpath(output_dir, "stress_test_$(label)"),
                                                    overwrite_existing = true)

    @info "Running $label..."
    try
        run!(simulation)
    catch e
        @warn "$label: simulation failed with $e"
    end

    return model
end

results = Dict{String, Any}()

for (scheme, label) in schemes
    results[label] = run_stress_test(ib_grid, scheme;
                                     label, Δt, stop_time, save_interval, output_dir)
end

#####
##### Visualization: static final-state plot
#####

# Grid coordinates
λc = λnodes(ib_grid, Center())
φc = φnodes(ib_grid, Center())
λf = λnodes(ib_grid, Face())
φf = φnodes(ib_grid, Face())

# Land mask for plotting
land = zeros(Nx, Ny)
for j in 1:Ny, i in 1:Nx
    land[i, j] = inactive_cell(i, j, 1, ib_grid) ? NaN : 1.0
end

labels = ["ES", "EN", "EEN", "AWES", "AWEN"]

function get_field_data(model, name)
    if name == "η"
        return interior(model.free_surface.displacement, :, :, 1)
    elseif name == "u"
        return interior(model.velocities.u, :, :, Nz)
    elseif name == "v"
        return interior(model.velocities.v, :, :, Nz)
    end
end

fig = Figure(size = (400 * length(labels), 1200))

# Row config: (field name, λ-nodes, φ-nodes, colormap, symmetric?)
rows = [("η", λc, φc, :balance, true),
        ("u", λf, φc, :balance, true),
        ("v", λc, φf, :balance, true)]

last_hm = Dict{Int, Any}()

for (col, label) in enumerate(labels)
    t_final = prettytime(results[label].clock.time)
    for (row, (name, λn, φn, cmap, sym)) in enumerate(rows)
        ax = Axis(fig[row, col];
                  title = row == 1 ? "$label (t=$t_final)" : "",
                  ylabel = col == 1 ? name : "",
                  xlabel = row == 3 ? "λ [°]" : "",
                  xticklabelsvisible = row == 3,
                  yticklabelsvisible = col == 1,
                  aspect = DataAspect())

        data = get_field_data(results[label], name)
        dlim = max(maximum(filter(isfinite, abs.(data))), 1e-10)
        cr = sym ? (-dlim, dlim) : (0, dlim)
        hm = heatmap!(ax, λn, φn, data; colormap=cmap, colorrange=cr)
        last_hm[row] = hm
    end
end

for (row, (name, _, _, _, _)) in enumerate(rows)
    Colorbar(fig[row, length(labels)+1], last_hm[row]; label=name)
end

save("coriolis_stress_test.png", fig, px_per_unit=2)
@info "Saved coriolis_stress_test.png"

#####
##### Summary statistics
#####

println("\n" * "="^80)
println("STRESS TEST SUMMARY")
println("="^80)

for label in labels
    model = results[label]
    u_max = maximum(abs, interior(model.velocities.u))
    v_max = maximum(abs, interior(model.velocities.v))
    η_max = maximum(abs, interior(model.free_surface.displacement))
    t_final = model.clock.time

    status = (isfinite(u_max) && u_max < 100) ? "OK" : "BLOW-UP"

    @printf("  %-6s: t=%-12s  max|u|=%8.4f  max|v|=%8.4f  max|η|=%8.4f  [%s]\n",
            label, prettytime(t_final), u_max, v_max, η_max, status)
end

# KE comparison
println("\nKinetic energy at final time:")
for label in labels
    model = results[label]
    u, v, w = model.velocities
    KE_op = @at (Center, Center, Center) (u^2 + v^2) / 2
    KE_field = Field(KE_op)
    compute!(KE_field)
    KE = sum(KE_field)
    @printf("  %-6s: KE = %.6e\n", label, KE)
end
println("="^80)

#####
##### Video: evolution of η, u, v for all schemes
#####

@info "Generating comparison video..."

# Load time series for each scheme
ts_data = Dict(label => (η = FieldTimeSeries(joinpath(output_dir, "stress_test_$(label).jld2"), "η"),
                         u = FieldTimeSeries(joinpath(output_dir, "stress_test_$(label).jld2"), "u"),
                         v = FieldTimeSeries(joinpath(output_dir, "stress_test_$(label).jld2"), "v"))
               for label in labels)

times = ts_data[labels[1]].η.times
Nt = length(times)

# Field info: (name, k-index, λ-nodes, φ-nodes, colormap, clamp)
fields_info = [
    ("η", 1, λc, φc, :balance, 2.0),
    ("u", 1, λf, φc, :balance, 1.0),
    ("v", 1, λc, φf, :balance, 1.0),
]

fig_vid = Figure(size = (350 * length(labels) + 80, 900), fontsize=12)
iter = Observable(1)

last_hm = Dict{String, Any}()

for (row, (name, k, λn, φn, cmap, clmp)) in enumerate(fields_info)
    for (col, label) in enumerate(labels)
        d = ts_data[label]
        fts = getfield(d, Symbol(name))

        ax = Axis(fig_vid[row, col];
                  title = row == 1 ? label : "",
                  ylabel = col == 1 ? "$name" : "",
                  xlabel = row == 3 ? "λ [°]" : "",
                  xticklabelsvisible = row == 3,
                  yticklabelsvisible = col == 1,
                  aspect = DataAspect())

        data = @lift interior(fts[min($iter, length(fts.times))], :, :, k)
        hm = heatmap!(ax, λn, φn, data; colormap=cmap, colorrange=(-clmp, clmp))
        last_hm[name] = hm
    end

    Colorbar(fig_vid[row, length(labels)+1], last_hm[name]; label=name)
end

title_str = @lift "Coriolis scheme comparison — t = $(prettytime(times[min($iter, Nt)]))"
Label(fig_vid[0, 1:length(labels)], title_str; fontsize=16, font=:bold)

video_path = "coriolis_stress_test_evolution.mp4"
record(fig_vid, video_path, 1:Nt; framerate=10) do n
    @info "  Video frame $n / $Nt"
    iter[] = n
end
@info "Saved video: $video_path"
