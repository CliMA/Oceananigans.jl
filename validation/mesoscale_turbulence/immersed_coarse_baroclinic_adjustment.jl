using Printf
using Statistics
using Random
using Oceananigans
using Oceananigans.Units
using GLMakie
using Oceananigans.TurbulenceClosures: IsopycnalSkewSymmetricDiffusivity, SkewAdvectionISSD
using Oceananigans.TurbulenceClosures: FluxTapering, DiffusiveFormulation, AdvectiveFormulation

filename = "coarse_baroclinic_adjustment"
wall_clock = Ref(time_ns())

function progress(sim)
    @printf("[%05.2f%%] i: %d, t: %s, wall time: %s, max(u): (%6.3e, %6.3e, %6.3e) m/s, next Δt: %s\n",
            100 * (sim.model.clock.time / sim.stop_time),
            sim.model.clock.iteration,
            prettytime(sim.model.clock.time),
            prettytime(1e-9 * (time_ns() - wall_clock[])),
            maximum(abs, sim.model.velocities.u),
            maximum(abs, sim.model.velocities.v),
            maximum(abs, sim.model.velocities.w),
            prettytime(sim.Δt))

    wall_clock[] = time_ns()

    return nothing
end

# Domain
Ly = 1000kilometers  # north-south extent [m]
Lz = 1kilometers     # depth [m]
Ny = 20
Nz = 20

# Time stepping
Δt = 10minutes
save_fields_interval = 2day
stop_time = 300days

grid = RectilinearGrid(topology = (Flat, Bounded, Bounded),
                       size = (Ny, Nz),
                       y = (-Ly/2, Ly/2),
                       z = (-Lz, 0),
                       halo = (5, 5))

grid = ImmersedBoundaryGrid(grid, GridFittedBottom(y -> y > 0 ? -Lz : -Lz/2))

@info "Building a model..."

κ_skew = 1000
slope_limiter = FluxTapering(1000) # Allow very steep slopes

adv_closure = IsopycnalSkewSymmetricDiffusivity(; κ_skew, slope_limiter)
dif_closure = IsopycnalSkewSymmetricDiffusivity(; κ_skew, slope_limiter, skew_flux_formulation = DiffusiveFormulation())

function run_simulation(closure, grid)
    model = HydrostaticFreeSurfaceModel(; grid, closure,
                                        coriolis = FPlane(latitude = -45),
                                        buoyancy = BuoyancyTracer(),
                                        tracer_advection = WENO(order=7),
                                        tracers = (:b, :c))

    @info "Built $model."

    ramp(y, Δ) = min(max(0, y / Δ + 1/2), 1)

    # Parameters
    N² = 4e-6 # [s⁻²] buoyancy frequency / stratification
    M² = 8e-8 # [s⁻²] horizontal buoyancy gradient

    Δy = 100kilometers
    Δz = 100

    Δc = 2Δy
    Δb = Δy * M²
    ϵb = 1e-2 * Δb # noise amplitude

    bᵢ(y, z) = N² * z + Δb * ramp(y, Δy)
    cᵢ(y, z) = exp(-y^2 / 2Δc^2) * exp(-(z + Lz/4)^2 / 2Δz^2)

    set!(model, b=bᵢ, c=cᵢ)

    #####
    ##### Simulation building
    #####

    simulation = Simulation(model; Δt, stop_time)
    add_callback!(simulation, progress, IterationInterval(144))
    suffix = closure isa SkewAdvectionISSD ? "advective" : "diffusive"

    simulation.output_writers[:fields] = JLD2Writer(model, merge(model.velocities, model.tracers),
                                                    schedule = TimeInterval(save_fields_interval),
                                                    filename = filename * "_fields_" * suffix,
                                                    overwrite_existing = true)

    @info "Running the simulation..."

    run!(simulation)

    @info "Simulation completed in " * prettytime(simulation.run_wall_time)
end

run_simulation(adv_closure, grid)
run_simulation(dif_closure, grid)

#####
##### Visualize
#####

filepath_adv = filename * "_fields_advective.jld2"
filepath_dif = filename * "_fields_diffusive.jld2"

uta = FieldTimeSeries(filepath_adv, "u")
bta = FieldTimeSeries(filepath_adv, "b")
cta = FieldTimeSeries(filepath_adv, "c")

utd = FieldTimeSeries(filepath_dif, "u")
btd = FieldTimeSeries(filepath_dif, "b")
ctd = FieldTimeSeries(filepath_dif, "c")

x, y, z = nodes(bta)

#####
##### Plot buoyancy...
#####

times = bta.times
Nt = length(times)

max_u = max([maximum(abs, uta[n]) for n in 1:Nt]...) * 0.5
min_u = - max_u

n  = Observable(1)
ua = @lift uta[$n]
ba = @lift interior(bta[$n], 1, :, :)
ca = @lift cta[$n]

ud = @lift utd[$n]
bd = @lift interior(btd[$n], 1, :, :)
cd = @lift ctd[$n]

fig = Figure(size=(1800, 700))

axua = Axis(fig[2, 1], xlabel="y (km)", ylabel="z (km)", title="GM_Adv: Zonal velocity")
axca = Axis(fig[3, 1], xlabel="y (km)", ylabel="z (km)", title="GM_Adv: Tracer concentration")
axud = Axis(fig[2, 2], xlabel="y (km)", ylabel="z (km)", title="GM_Dif: Zonal velocity")
axcd = Axis(fig[3, 2], xlabel="y (km)", ylabel="z (km)", title="GM_Dif: Tracer concentration")

levels = [-0.0015 + 0.0005 * i for i in 0:19]

hm = heatmap!(axua, ua, colorrange=(min_u, max_u), colormap=:balance)
contour!(axua, y, z, ba; levels, color=:black, linewidth=2)

hm = heatmap!(axca, ca, colorrange=(0, 0.5), colormap=:speed)
contour!(axca, y, z, ba; levels, color=:black, linewidth=2)

hm = heatmap!(axud, ud, colorrange=(min_u, max_u), colormap=:balance)
contour!(axud, y, z, bd; levels, color=:black, linewidth=2)
cb = Colorbar(fig[2, 3], hm)

hm = heatmap!(axcd, cd, colorrange=(0, 0.5), colormap=:speed)
contour!(axcd, y, z, bd; levels, color=:black, linewidth=2)
cb = Colorbar(fig[3, 3], hm)

title_str = @lift "Baroclinic adjustment with GM at t = " * prettytime(times[$n])
ax_t = fig[1, 1:3] = Label(fig, title_str)

display(fig)

record(fig, filename * ".mp4", 1:Nt, framerate=8) do i
    @info "Plotting frame $i of $Nt"
    n[] = i
end
