# A turbulent open channel with a recycled inflow and a hill, compared with the periodic channel it is
# initialized from.
#
# The flow is a half channel: a pressure-gradient force u★² / H drives it over a rough bottom with quadratic drag.
# A periodic precursor is run until it is turbulent. The open channel starts from four copies of its final state
# laid along x, with the recycling plane one copy in from the west boundary and a radiating outflow on the east.
# A Gaussian hill with the same drag sits on its bottom at x = 12 H, between the recycling plane and the outflow.
# The reference is the periodic channel, which has no hill, continued from the same state over the same time. Three
# open channels are run:
#
#   * `fluctuations`: the precursor's mean profile as the boundary value plus recycled fluctuations about the
#     `y`-mean, with the transport pinned by `target_transport`. The pressure-gradient force then only changes
#     the pressure, since the transport is set at the inflow.
#   * `absolute`: the recycling plane imposed as is, with no boundary value and no target, so that the force
#     sets the bulk flow as in the periodic channel.
#   * `laminar`: the same mean profile imposed without recycling, for reference.
#
# The script writes a figure of statistics and an animation of the first part of the four runs, in which a dashed
# line marks the recycling plane and the hill is gray.

using Oceananigans
using Oceananigans.BoundaryConditions: ObliqueRadiation
using Oceananigans.ImmersedBoundaries: mask_immersed_field!
using Oceananigans.Solvers: ConjugateGradientPoissonSolver
using CairoMakie
using JLD2
using Printf
using Random
using Statistics: mean

const κ = 0.4

H = 1                    # depth
u★ = 1                   # friction velocity set by the pressure-gradient force
z₀ = 1e-4 * H            # roughness length
N = 64                   # cells along x and y in one periodic box, and twice the number over the depth
box_length = 2π * H
boxes = 4                # periodic boxes along the open channel
hill_height = 0.4 * H
hill_width = 0.8 * H     # standard deviation of the Gaussian
hill_position = 12 * H
precursor_stop_time = 40 * H / u★
stop_time = 20 * H / u★
averaging_start_time = 8 * H / u★
sampling_interval = 0.25 * H / u★
animation_times = (0:0.02:8) .* (H / u★)

Nz = N ÷ 2
animation_level = Nz ÷ 8     # vertical index of the horizontal plane in the animation
Δy, Δz = box_length / N, H / Nz
cᴰ = (κ / log(Δz / 2z₀))^2

@inline pressure_gradient_force(x, y, z, t, p) = p.u★^2 / p.H

bottom_drag = BulkDrag(coefficient = cᴰ)
periodic_boundary_conditions = (u = FieldBoundaryConditions(bottom = bottom_drag), v = FieldBoundaryConditions(bottom = bottom_drag))

# An immersed boundary needs one more halo point than the advection scheme does
channel_grid(topology, Nx) = RectilinearGrid(; size = (Nx, N, Nz), halo = (4, 4, 4), x = (0, Nx * box_length / N), y = (0, box_length), z = (0, H), topology)

hill(x, y) = hill_height * exp(-((x - hill_position)^2 + (y - box_length / 2)^2) / 2hill_width^2)

function channel_simulation(grid, boundary_conditions; stop_time, pressure_solver = nothing)
    model = NonhydrostaticModel(grid; timestepper = :RungeKutta3, advection = WENO(order = 5), closure = SmagorinskyLilly(), boundary_conditions, pressure_solver,
                                forcing = (; u = Forcing(pressure_gradient_force, parameters = (; u★, H))))

    simulation = Simulation(model; Δt = 1e-3 * H / u★, stop_time)
    conjure_time_step_wizard!(simulation, IterationInterval(10); cfl = 0.7)

    return simulation
end

# Samples of `y`-averaged statistics on the (x, z) plane, of the bottom stress, of the pressure and of the transport
function sample_statistics!(simulation)
    model = simulation.model
    u, v, w = model.velocities
    Nx = size(model.grid, 1)
    centered = map(q -> Field(@at (Center, Center, Center) q), (; u, v, w))
    spanwise_mean(a) = dropdims(mean(a, dims = 2), dims = 2)
    samples = (; times = Float64[], transport = Float64[], u = [], uu = [], vv = [], ww = [], uw = [], stress = [], pressure = [], variance = [], covariance = [])

    # Index of the lowest fluid cell of each column, which is above 1 on the hill
    fluid = CenterField(model.grid)
    set!(fluid, 1)
    mask_immersed_field!(fluid)
    lowest = Nz + 1 .- round.(Int, sum(Array(interior(fluid)), dims = 3))

    function sample(simulation)
        foreach(compute!, centered)
        uᶜ, vᶜ, wᶜ = map(q -> Array(interior(q)), centered)
        u₁, v₁ = map(q -> [q[i, j, lowest[i, j]] for i in 1:Nx, j in 1:N], (uᶜ, vᶜ))
        u′ = uᶜ[:, :, Nz ÷ 2] .- mean(uᶜ[:, :, Nz ÷ 2], dims = 2)

        push!(samples.times, time(simulation))
        push!(samples.transport, sum(interior(u, 1, :, :)) * Δy * Δz)
        push!(samples.u, spanwise_mean(uᶜ))
        push!(samples.uu, spanwise_mean(uᶜ.^2))
        push!(samples.vv, spanwise_mean(vᶜ.^2))
        push!(samples.ww, spanwise_mean(wᶜ.^2))
        push!(samples.uw, spanwise_mean(uᶜ .* wᶜ))
        push!(samples.stress, spanwise_mean(cᴰ .* sqrt.(u₁.^2 .+ v₁.^2) .* u₁))
        push!(samples.pressure, vec(mean(Array(interior(model.pressures.pNHS)), dims = (2, 3))))

        # Mid-depth covariance of u between points one box apart along x
        push!(samples.variance, spanwise_mean(u′.^2))
        push!(samples.covariance, spanwise_mean(u′[1:Nx-N, :] .* u′[N+1:Nx, :]))

        return nothing
    end

    add_callback!(simulation, sample, TimeInterval(sampling_interval))

    return samples
end

function progress(simulation)
    u, v, w = simulation.model.velocities
    @printf("iteration %d, time %.2f, Δt %.2e, max|w| %.2f, bulk velocity %.3f\n", iteration(simulation), time(simulation),
            simulation.Δt, maximum(abs, w), sum(interior(u, 1, :, :)) * Δy * Δz / (box_length * H))
    return nothing
end

# The log law plus large eddies with random phases, strong enough to trip the turbulence. The eddies are the curl of
# a vector potential (Aˣ, Aʸ, 0) = Σ sin(m π z / H) (cos(k x + l y + θˣ), cos(k x + l y + θʸ)) / H, so they have no divergence.
function run_precursor(filename)
    simulation = channel_simulation(channel_grid((Periodic, Periodic, Bounded), N), periodic_boundary_conditions; stop_time = precursor_stop_time)

    Random.seed!(1)
    modes = [(k, l, m, 2π * rand(), 2π * rand()) for k in 0:3 for l in -3:3 for m in 1:2 if (k, l) != (0, 0)]
    u′(x, y, z) = - sum(m * π * cos(m * π * z / H) * cos((k * x + l * y) / H + θʸ) for (k, l, m, θˣ, θʸ) in modes)
    v′(x, y, z) =   sum(m * π * cos(m * π * z / H) * cos((k * x + l * y) / H + θˣ) for (k, l, m, θˣ, θʸ) in modes)
    w′(x, y, z) =   sum(sin(m * π * z / H) * (l * sin((k * x + l * y) / H + θˣ) - k * sin((k * x + l * y) / H + θʸ)) for (k, l, m, θˣ, θʸ) in modes)
    a = 3u★ / sqrt(mean(u′(x, y, H / 2)^2 for x in range(0, box_length, 32), y in range(0, box_length, 32)))

    set!(simulation.model, u = (x, y, z) -> u★ / κ * log(z / z₀) + a * u′(x, y, z), v = (x, y, z) -> a * v′(x, y, z), w = (x, y, z) -> a * w′(x, y, z))

    samples = sample_statistics!(simulation)
    add_callback!(simulation, progress, IterationInterval(1000))
    run!(simulation)

    velocities = map(q -> Array(interior(q)), simulation.model.velocities)
    jldsave(filename; samples, velocities)

    return nothing
end

# Copies of the periodic state laid along x. In an open channel a field at x-faces has one more face than the copies hold.
function tiled(a, Nx)
    copies = repeat(a, Nx ÷ N, 1, 1)
    return Nx == size(copies, 1) ? copies : cat(copies, a[1:1, :, :], dims = 1)
end

# A channel that starts from copies of the precursor's final state
function run_channel(filename, grid, boundary_conditions, precursor_velocities; pressure_solver = nothing)
    simulation = channel_simulation(grid, boundary_conditions; stop_time, pressure_solver)
    model = simulation.model
    Nx = size(grid, 1)
    set!(model; map((q, a) -> tiled(a, size(q, 1)), model.velocities, precursor_velocities)...)

    δ = Field(∂x(model.velocities.u) + ∂y(model.velocities.v) + ∂z(model.velocities.w))
    maximum_divergence = Ref(0.0)
    track_divergence(simulation) = maximum_divergence[] = max(maximum_divergence[], maximum(abs, compute!(δ)))

    # Frames of u on a horizontal plane near the bottom and on a vertical plane, for the animation
    u = model.velocities.u
    frames = (; times = Float64[], horizontal = Matrix{Float32}[], vertical = Matrix{Float32}[])

    function save_frame(simulation)
        push!(frames.times, time(simulation))
        push!(frames.horizontal, Array(interior(u, 1:Nx, :, animation_level)))
        push!(frames.vertical, Array(interior(u, 1:Nx, N ÷ 2, :)))
        return nothing
    end

    samples = sample_statistics!(simulation)
    add_callback!(simulation, progress, IterationInterval(1000))
    add_callback!(simulation, track_divergence, IterationInterval(100))
    add_callback!(simulation, save_frame, SpecifiedTimes(collect(animation_times)))
    run!(simulation)

    jldsave(filename; samples, frames, maximum_divergence = maximum_divergence[])

    return nothing
end

#####
##### Runs
#####

precursor_file = "recycled_channel_precursor.jld2"
isfile(precursor_file) || run_precursor(precursor_file)
precursor = load(precursor_file)

# Mean of the samples taken after `start_time`
time_mean(samples, name, start_time) = mean(samples[name][samples.times .≥ start_time])

precursor_start_time = precursor_stop_time - (stop_time - averaging_start_time)
mean_profile = vec(mean(time_mean(precursor["samples"], :u, precursor_start_time), dims = 1))
exterior_velocity = repeat(reshape(mean_profile, 1, Nz), N, 1)
target_transport = sum(mean_profile) * Δz * box_length

recycling(averaging; kwargs...) = TurbulenceRecycling(; recycling_distance = box_length, averaging, kwargs...)
outflow = NormalFlowBoundaryCondition(exterior_velocity; scheme = ObliqueRadiation())

west_conditions = (fluctuations = (u = NormalFlowBoundaryCondition(exterior_velocity; scheme = recycling(2; target_transport)),
                                   v = ValueBoundaryCondition(0; scheme = recycling(2)),
                                   w = ValueBoundaryCondition(0; scheme = recycling(2))),
                   absolute = (u = NormalFlowBoundaryCondition(0; scheme = recycling(nothing)),
                               v = ValueBoundaryCondition(0; scheme = recycling(nothing)),
                               w = ValueBoundaryCondition(0; scheme = recycling(nothing))),
                   laminar = (u = NormalFlowBoundaryCondition(exterior_velocity), v = ValueBoundaryCondition(0), w = ValueBoundaryCondition(0)))

periodic_file = "recycled_channel_periodic.jld2"
isfile(periodic_file) || run_channel(periodic_file, channel_grid((Periodic, Periodic, Bounded), N), periodic_boundary_conditions, precursor["velocities"])
periodic = load(periodic_file)

open_grid = ImmersedBoundaryGrid(channel_grid((Open, Periodic, Bounded), boxes * N), GridFittedBottom(hill))
hill_drag = ImmersedBoundaryCondition(bottom = bottom_drag)

for name in keys(west_conditions)
    west = west_conditions[name]
    boundary_conditions = (u = FieldBoundaryConditions(bottom = bottom_drag, immersed = hill_drag, west = west.u, east = outflow),
                           v = FieldBoundaryConditions(bottom = bottom_drag, immersed = hill_drag, west = west.v),
                           w = FieldBoundaryConditions(west = west.w))

    filename = "recycled_channel_$name.jld2"
    isfile(filename) || run_channel(filename, open_grid, boundary_conditions, precursor["velocities"]; pressure_solver = ConjugateGradientPoissonSolver(open_grid; maxiter = 20))
end

runs = map(name -> load("recycled_channel_$name.jld2"), (; fluctuations = :fluctuations, absolute = :absolute, laminar = :laminar))

#####
##### Statistics
#####

# Means over `y` and time on the (x, z) plane; fluctuations are about that mean
function statistics(samples, start_time)
    U = time_mean(samples, :u, start_time)
    uu = time_mean(samples, :uu, start_time) .- U.^2
    vv = time_mean(samples, :vv, start_time)
    ww = time_mean(samples, :ww, start_time)
    uw = time_mean(samples, :uw, start_time)
    stress = vec(time_mean(samples, :stress, start_time))
    pressure = vec(time_mean(samples, :pressure, start_time))
    kinetic_energy = vec(sum(uu .+ vv .+ ww, dims = 2)) .* Δz ./ 2
    variance = vec(time_mean(samples, :variance, start_time))
    covariance = vec(time_mean(samples, :covariance, start_time))
    correlation = covariance ./ sqrt.(variance[1:end-N] .* variance[N+1:end])
    return (; U, uu, vv, ww, uw, stress, pressure, kinetic_energy, correlation)
end

reference = statistics(periodic["samples"], averaging_start_time)
results = map(r -> statistics(r["samples"], averaging_start_time), runs)

x = (1:boxes*N) .* (box_length / N) .- box_length / 2N
y = (1:N) .* Δy .- Δy / 2
z = (1:Nz) .* Δz .- Δz / 2
colors = (fluctuations = :royalblue, absolute = :firebrick, laminar = :gray50)
descriptions = (fluctuations = "Open channel, inflow: mean profile + recycled fluctuations\n(averaging = 2, target_transport)",
                absolute = "Open channel, inflow: recycled plane as is\n(averaging = nothing, no target_transport)",
                laminar = "Open channel, inflow: mean profile only\n(no recycling)")
mark_recycling_plane!(axis) = vlines!(axis, box_length; color = :limegreen, linestyle = :dash, linewidth = 2.5, label = "Recycling plane")
upstream = N+1:3N÷2    # cells between the recycling plane and the hill, 2π < x / H < 3π
upstream_mean(a) = vec(mean(a[upstream, :], dims = 1))

fig = Figure(size = (1500, 1300))
stress_axis = Axis(fig[1, 1]; title = "Bottom stress", xlabel = "x / H", ylabel = "τ / u★²")
energy_axis = Axis(fig[1, 2]; title = "Depth-integrated resolved kinetic energy of fluctuations", xlabel = "x / H", ylabel = "∫ k dz / (u★² H)")
correlation_axis = Axis(fig[1, 3]; title = "Correlation of u(x) with u(x + 2π H) at mid-depth", xlabel = "x / H", ylabel = "correlation")
velocity_axis = Axis(fig[2, 1]; title = "Mean velocity, 2π < x / H < 3π", xlabel = "z / H", ylabel = "U / u★", xscale = log10)
variance_axis = Axis(fig[2, 2]; title = "Variances, 2π < x / H < 3π (solid: u, dashed: w)", xlabel = "variance / u★²", ylabel = "z / H")
flux_axis = Axis(fig[2, 3]; title = "Resolved momentum flux, 2π < x / H < 3π", xlabel = "-⟨u′w′⟩ / u★²", ylabel = "z / H")
transport_axis = Axis(fig[3, 1]; title = "Bulk velocity through the plane x = 0", xlabel = "t u★ / H", ylabel = "transport / (area u★)")
pressure_axis = Axis(fig[3, 2]; title = "Mean nonhydrostatic pressure, relative to the east end", xlabel = "x / H", ylabel = "p / u★²")

hlines!(stress_axis, mean(reference.stress); color = :black, linewidth = 3, label = "Periodic channel (reference)")
hlines!(energy_axis, mean(reference.kinetic_energy); color = :black, linewidth = 3)
lines!(velocity_axis, z, vec(mean(reference.U, dims = 1)); color = :black, linewidth = 3)
lines!(variance_axis, vec(mean(reference.uu, dims = 1)), z; color = :black, linewidth = 3)
lines!(variance_axis, vec(mean(reference.ww, dims = 1)), z; color = :black, linewidth = 3, linestyle = :dash)
lines!(flux_axis, -vec(mean(reference.uw, dims = 1)), z; color = :black, linewidth = 3)

for name in keys(runs)
    result, samples, color = results[name], runs[name]["samples"], colors[name]
    lines!(stress_axis, x, result.stress; color, label = descriptions[name])
    lines!(energy_axis, x, result.kinetic_energy; color)
    name == :laminar || lines!(correlation_axis, x[1:end-N], result.correlation; color)
    lines!(velocity_axis, z, upstream_mean(result.U); color)
    lines!(variance_axis, upstream_mean(result.uu), z; color)
    lines!(variance_axis, upstream_mean(result.ww), z; color, linestyle = :dash)
    lines!(flux_axis, -upstream_mean(result.uw), z; color)
    lines!(transport_axis, samples.times, samples.transport ./ (box_length * H); color)
    lines!(pressure_axis, x, result.pressure .- result.pressure[end]; color)

    late = samples.times .≥ averaging_start_time
    pressure_gradient = (result.pressure[end] - result.pressure[N+1]) / (x[end] - x[N+1])
    @printf("%-13s transport / target between %.6f and %.6f, maximum divergence %.1e, mean ∂x p downstream of the plane %+.2f u★² / H, stress upstream of the hill %.3f u★²\n",
            name, extrema(samples.transport[late] ./ target_transport)..., runs[name]["maximum_divergence"], pressure_gradient * H / u★^2, mean(result.stress[upstream]))
end

lines!(transport_axis, periodic["samples"].times, periodic["samples"].transport ./ (box_length * H); color = :black, linewidth = 3)
@printf("periodic      stress %.3f u★², bulk velocity at the end %.3f u★, of the inflow profile %.3f u★\n", mean(reference.stress),
        periodic["samples"].transport[end] / (box_length * H), target_transport / (box_length * H))

for axis in (stress_axis, energy_axis, correlation_axis, pressure_axis)
    mark_recycling_plane!(axis)
    vspan!(axis, hill_position - 2hill_width, hill_position + 2hill_width; color = (:gray50, 0.3), label = "Hill")
end

Legend(fig[3, 3], stress_axis; tellwidth = false)
save("recycled_channel_inflow.png", fig)

#####
##### Animation of u minus the mean profile
#####

frame = Observable(1)
frame_times = periodic["frames"].times
xᶠ = x .- box_length / 2N
animation = Figure(size = (1700, 1450))
Label(animation[0, 1:2], @lift "Streamwise velocity minus the mean profile of the periodic channel, (u - U(z)) / u★, " * @sprintf("at t = %.2f H / u★\n", frame_times[$frame]) *
                               "The flow goes from left to right. Dashed lines mark the recycling plane, which is sampled and fed back at the inflow (x = 0). The hill is gray.";
      fontsize = 20)

for (row, name) in enumerate((:periodic, keys(runs)...))
    frames = name == :periodic ? periodic["frames"] : runs[name]["frames"]
    copies = name == :periodic ? boxes : 1
    description = name == :periodic ? "Periodic channel (reference), shown $boxes times side by side" : descriptions[name]

    # u is exactly zero inside the hill, where it is replaced by NaN to be drawn in gray
    horizontal = @lift repeat(replace(frames.horizontal[$frame], 0 => NaN), copies, 1) .- mean_profile[animation_level]
    vertical = @lift repeat(replace(frames.vertical[$frame], 0 => NaN), copies, 1) .- reshape(mean_profile, 1, Nz)

    Label(animation[2row - 1, 1:2], description; font = :bold, fontsize = 17)
    horizontal_axis = Axis(animation[2row, 1]; title = "Horizontal plane at z = $(round(z[animation_level], digits = 2)) H", titlefont = :regular,
                           xlabel = "x / H", ylabel = "y / H", aspect = DataAspect())
    heatmap!(horizontal_axis, xᶠ, y, horizontal; colormap = :balance, colorrange = (-6, 6), nan_color = :gray50)
    vertical_axis = Axis(animation[2row, 2]; title = "Vertical plane at y = π H", titlefont = :regular, xlabel = "x / H", ylabel = "z / H")
    heatmap!(vertical_axis, xᶠ, z, vertical; colormap = :balance, colorrange = (-6, 6), nan_color = :gray50)
    name in (:fluctuations, :absolute) && foreach(mark_recycling_plane!, (horizontal_axis, vertical_axis))
end

Colorbar(animation[1:2length(runs)+2, 3]; colormap = :balance, limits = (-6, 6), label = "(u - U) / u★")

record(animation, "recycled_channel_inflow.mp4", eachindex(frame_times); framerate = 25) do n
    frame[] = n
end
