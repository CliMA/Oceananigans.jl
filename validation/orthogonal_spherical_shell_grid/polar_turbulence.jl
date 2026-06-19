using Oceananigans
using Oceananigans.Units
using Printf
using Random

# Set OCEANANIGANS_VALIDATE_STOP_ITERATION for a short smoke run.
# Set OCEANANIGANS_VALIDATE_MOVIE=true to save JLD2 output and render a movie.

arch = CPU()

Nx, Ny, Nz = 64, 64, 1
latitude = (-60, 60)
longitude = (120, 240)
z = (-1000, 0)
topology = (Bounded, Bounded, Bounded)

grid = RotatedLatitudeLongitudeGrid(arch;
                                    size = (Nx, Ny, Nz),
                                    latitude,
                                    longitude,
                                    north_pole = (0, 0),
                                    halo = (7, 7, 3),
                                    z,
                                    topology)

model = HydrostaticFreeSurfaceModel(grid;
                                    coriolis = HydrostaticSphericalCoriolis(),
                                    momentum_advection = WENOVectorInvariant(),
                                    tracers = ())

Random.seed!(123)

U = 0.01meters / second
ϵᵢ(λ, φ, z) = U * (2rand() - 1)
set!(model, u = ϵᵢ, v = ϵᵢ)

Δt = 10minutes
stop_iteration = parse(Int, get(ENV, "OCEANANIGANS_VALIDATE_STOP_ITERATION", "0"))

simulation = if stop_iteration > 0
    Simulation(model; Δt, stop_iteration)
else
    Simulation(model; Δt, stop_time = 1day)
end

function progress_message(sim)
    @printf("Iteration: %04d, time: %s, Δt: %s, wall time: %s\n",
            iteration(sim), prettytime(sim), prettytime(sim.Δt), prettytime(sim.run_wall_time))

    return nothing
end

simulation.callbacks[:progress] = Callback(progress_message, IterationInterval(20))

make_movie = get(ENV, "OCEANANIGANS_VALIDATE_MOVIE", "false") == "true"

if make_movie
    u, v, w = model.velocities
    ζ = ∂x(v) - ∂y(u)
    s = @at (Center, Center, Center) sqrt(u^2 + v^2)

    simulation.output_writers[:jld2] = JLD2Writer(model, (; u, v, ζ, s),
                                                  schedule = TimeInterval(2hours),
                                                  filename = "polar_turbulence",
                                                  overwrite_existing = true)
end

@info "Run rotated-pole turbulence validation..."

run!(simulation)

u, v, w = model.velocities

@info "Finished rotated-pole turbulence validation" maximum(abs, interior(u)) maximum(abs, interior(v))

if make_movie
    @info "Load output and make a movie..."

    using GLMakie

    filepath = simulation.output_writers[:jld2].filepath

    ζ_timeseries = FieldTimeSeries(filepath, "ζ")
    u_timeseries = FieldTimeSeries(filepath, "u")
    v_timeseries = FieldTimeSeries(filepath, "v")
    s_timeseries = FieldTimeSeries(filepath, "s")

    times = u_timeseries.times
    n = Observable(1)

    title = lift(n -> @sprintf("t = %s", prettytime(times[n])), n)

    ζₙ = lift(n -> interior(ζ_timeseries[n], :, :, 1), n)
    uₙ = lift(n -> interior(u_timeseries[n], :, :, 1), n)
    vₙ = lift(n -> interior(v_timeseries[n], :, :, 1), n)
    sₙ = lift(n -> interior(s_timeseries[n], :, :, 1), n)

    s_lim = maximum(abs, interior(s_timeseries))
    ζ_lim = maximum(abs, interior(ζ_timeseries[end])) / 2

    fig = Figure(size = (1200, 800))

    ax_u = Axis(fig[1, 1], aspect = 1)
    ax_v = Axis(fig[1, 3], aspect = 1)
    ax_ζ = Axis(fig[2, 1], aspect = 1)
    ax_s = Axis(fig[2, 3], aspect = 1)

    hm_u = heatmap!(ax_u, uₙ; colormap = :balance, colorrange = (-s_lim, s_lim))
    Colorbar(fig[1, 2], hm_u; label = "u (m s⁻¹)")

    hm_v = heatmap!(ax_v, vₙ; colormap = :balance, colorrange = (-s_lim, s_lim))
    Colorbar(fig[1, 4], hm_v; label = "v (m s⁻¹)")

    hm_ζ = heatmap!(ax_ζ, ζₙ; colormap = :balance, colorrange = (-ζ_lim, ζ_lim))
    Colorbar(fig[2, 2], hm_ζ; label = "ζ")

    hm_s = heatmap!(ax_s, sₙ; colormap = :speed, colorrange = (0, s_lim))
    Colorbar(fig[2, 4], hm_s; label = "√u²+v² (m s⁻¹)")

    fig[0, :] = Label(fig, title, tellwidth = false)

    frames = 1:length(times)

    record(fig, "polar_turbulence.mp4", frames, framerate = 12) do i
        n[] = i
    end
end
