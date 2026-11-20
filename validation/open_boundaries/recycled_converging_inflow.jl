# Turbulent boundary layers enter a square domain through its west and east boundaries, meet, and leave through the
# south and north boundaries. Nothing forces the flow: `target_transport` on the two inflows sets how much comes in,
# and the radiating outflows share what has to leave.
#
# The turbulence comes from the periodic channel of `recycled_channel_inflow.jl`, which has to be run first in the
# same directory: the initial state is four copies of that channel with the mean flow turned into u = -a x,
# v = a y. Recycling is used here outside its assumptions: the mean flow slows down from the boundary on, and at
# the recycling plane, half-way to the centre, it is half as fast as at the boundary. Four runs are compared:
#
#   * `absolute`: the recycling plane imposed as is and shifted to the target transport.
#   * `fluctuations`: the channel's mean profile as the boundary value plus recycled fluctuations about the `y`-mean.
#   * `scaled`: the same, with the fluctuations scaled by the ratio of the mean velocity at the plane to that at
#     the boundary.
#   * `laminar`: the mean profile imposed without recycling, for reference.
#
# The targets hold in all four. In the first two the recycled fluctuations grow from pass to pass and the mean inflow
# tilts along y. Scaling stops the growth and leaves the turbulence at the inflow intermittent.
#
# The script writes a figure of statistics and an animation of the four runs. Dashed lines mark the recycling planes.

using Oceananigans
using CairoMakie
using JLD2
using Printf
using Statistics: mean, var

const κ = 0.4

H = 1                    # depth
u★ = 1                   # friction velocity of the periodic channel
z₀ = 1e-4 * H            # roughness length
N = 64                   # cells along x and y in the periodic channel, and twice the number over the depth
box_length = 2π * H
recycling_distance = box_length / 2
velocity_ratio = 1 - recycling_distance / box_length    # mean velocity at the recycling plane over that at the boundary
stop_time = 4 * H / u★
averaging_start_time = 1.5 * H / u★
sampling_interval = 0.05 * H / u★
animation_times = (0:0.02:4) .* (H / u★)

Nz = N ÷ 2
Δ, Δz = box_length / N, H / Nz
cᴰ = (κ / log(Δz / 2z₀))^2

@inline drag_u(x, y, t, u, v, p) = - p.cᴰ * sqrt(u^2 + v^2) * u
@inline drag_v(x, y, t, u, v, p) = - p.cᴰ * sqrt(u^2 + v^2) * v

grid = RectilinearGrid(size = (2N, 2N, Nz), x = (-box_length, box_length), y = (-box_length, box_length), z = (0, H), topology = (Open, Open, Bounded))

precursor = load("recycled_channel_precursor.jld2")
late = precursor["samples"].times .≥ precursor["samples"].times[end] - (stop_time - averaging_start_time)
mean_profile = vec(mean(mean(precursor["samples"].u[late]), dims = 1))
bulk_velocity = mean(mean_profile)
target_transport = bulk_velocity * 2 * box_length * H
exterior_velocity = repeat(reshape(mean_profile, 1, Nz), 2N, 1)

# Boundary conditions of a recycled inflow; `direction` is +1 on the west boundary and -1 on the east boundary
function recycled_inflow(direction, exterior_velocity, averaging; kwargs...)
    recycling(; target...) = TurbulenceRecycling(; recycling_distance, averaging, kwargs..., target...)
    return (u = NormalFlowBoundaryCondition(direction * exterior_velocity; scheme = recycling(target_transport = direction * target_transport)),
            v = ValueBoundaryCondition(0; scheme = recycling()),
            w = ValueBoundaryCondition(0; scheme = recycling()))
end

inflows = (absolute = direction -> recycled_inflow(direction, 0, nothing),
           fluctuations = direction -> recycled_inflow(direction, exterior_velocity, 2),
           scaled = direction -> recycled_inflow(direction, exterior_velocity, 2; damping = velocity_ratio),
           laminar = direction -> (u = NormalFlowBoundaryCondition(direction * exterior_velocity), v = ValueBoundaryCondition(0), w = ValueBoundaryCondition(0)))

# Four copies of the periodic state. A field at faces along an open direction has one more face than the copies hold.
function tiled(a, field_size)
    copies = repeat(a, 2, 2, 1)
    copies = field_size[1] == size(copies, 1) ? copies : cat(copies, copies[1:1, :, :], dims = 1)
    return field_size[2] == size(copies, 2) ? copies : cat(copies, copies[:, 1:1, :], dims = 2)
end

function run_converging_inflow(filename, inflow)
    west, east = inflow(1), inflow(-1)
    outflow(direction) = NormalFlowBoundaryCondition(direction * bulk_velocity; scheme = PerturbationAdvection())

    bottom_drag(drag) = FluxBoundaryCondition(drag, field_dependencies = (:u, :v), parameters = (; cᴰ))

    boundary_conditions = (u = FieldBoundaryConditions(west = west.u, east = east.u, bottom = bottom_drag(drag_u)),
                           v = FieldBoundaryConditions(west = west.v, east = east.v, south = outflow(-1), north = outflow(1), bottom = bottom_drag(drag_v)),
                           w = FieldBoundaryConditions(west = west.w, east = east.w))

    model = NonhydrostaticModel(grid; timestepper = :RungeKutta3, advection = WENO(order = 5), closure = SmagorinskyLilly(), boundary_conditions)

    u, v, w = model.velocities
    xᶠ, yᶠ = xnodes(grid, Face()), reshape(ynodes(grid, Face()), 1, :)
    uᵢ = - xᶠ ./ box_length .* tiled(precursor["velocities"].u, size(u))
    vᵢ = tiled(precursor["velocities"].v, size(v)) .+ yᶠ ./ box_length .* reshape(mean_profile, 1, 1, Nz)
    set!(model, u = uᵢ, v = vᵢ, w = tiled(precursor["velocities"].w, size(w)))

    simulation = Simulation(model; Δt = 1e-3 * H / u★, stop_time)
    conjure_time_step_wizard!(simulation, IterationInterval(10); cfl = 0.7)

    centered = map(q -> Field(@at (Center, Center, Center) q), (; u, v, w))
    sums = map(_ -> zeros(size(grid)), (; u = 0, v = 0, w = 0, uu = 0, vv = 0, ww = 0))
    samples = Ref(0)
    transports = (; times = Float64[], west = Float64[], east = Float64[], south = Float64[], north = Float64[])
    δ = Field(∂x(u) + ∂y(v) + ∂z(w))
    maximum_divergence = Ref(0.0)

    function sample(simulation)
        push!(transports.times, time(simulation))
        push!(transports.west, sum(interior(u, 1, :, :)) * Δ * Δz)
        push!(transports.east, sum(interior(u, 2N + 1, :, :)) * Δ * Δz)
        push!(transports.south, sum(interior(v, :, 1, :)) * Δ * Δz)
        push!(transports.north, sum(interior(v, :, 2N + 1, :)) * Δ * Δz)
        maximum_divergence[] = max(maximum_divergence[], maximum(abs, compute!(δ)))

        time(simulation) < averaging_start_time && return nothing

        for name in (:u, :v, :w)
            q = interior(compute!(centered[name]))
            sums[name] .+= q
            sums[Symbol(name, name)] .+= q.^2
        end

        samples[] += 1

        return nothing
    end

    # Frames of u and w at mid-depth, for the animation
    frames = (; times = Float64[], u = Matrix{Float32}[], w = Matrix{Float32}[])

    function save_frame(simulation)
        push!(frames.times, time(simulation))
        push!(frames.u, Array(interior(u, :, :, Nz ÷ 2)))
        push!(frames.w, Array(interior(w, :, :, Nz ÷ 2)))
        return nothing
    end

    progress(simulation) = @printf("iteration %d, time %.2f, Δt %.2e, max|w| %.2f\n", iteration(simulation), time(simulation), simulation.Δt, maximum(abs, w))

    add_callback!(simulation, sample, TimeInterval(sampling_interval))
    add_callback!(simulation, progress, IterationInterval(1000))
    add_callback!(simulation, save_frame, SpecifiedTimes(collect(animation_times)))
    run!(simulation)

    jldsave(filename; sums, samples = samples[], transports, frames, maximum_divergence = maximum_divergence[], u = Array(interior(u, :, :, Nz ÷ 2)))

    return nothing
end

for name in keys(inflows)
    filename = "recycled_converging_inflow_$name.jld2"
    isfile(filename) || run_converging_inflow(filename, inflows[name])
end

runs = NamedTuple(name => load("recycled_converging_inflow_$name.jld2") for name in keys(inflows))

#####
##### Statistics: time means and variances at every point
#####

function statistics(run)
    means = map(name -> run["sums"][name] ./ run["samples"], (; u = :u, v = :v, w = :w))
    variances = map(name -> run["sums"][Symbol(name, name)] ./ run["samples"] .- means[name].^2, (; u = :u, v = :v, w = :w))
    kinetic_energy = dropdims(sum(variances.u .+ variances.v .+ variances.w, dims = 3), dims = 3) .* Δz ./ 2
    U = dropdims(mean(means.u, dims = 3), dims = 3)
    V = dropdims(mean(means.v, dims = 3), dims = 3)
    return (; kinetic_energy, U, V)
end

results = map(statistics, runs)

x = xnodes(grid, Center())
y = ynodes(grid, Center())
colors = (absolute = :firebrick, fluctuations = :royalblue, scaled = :seagreen, laminar = :gray50)
descriptions = (absolute = "Inflow: recycled plane as is\n(averaging = nothing)",
                fluctuations = "Inflow: mean profile + recycled fluctuations\n(averaging = 2)",
                scaled = "Inflow: mean profile + $velocity_ratio × recycled fluctuations\n(averaging = 2, damping = $velocity_ratio)",
                laminar = "Inflow: mean profile only\n(no recycling)")
limits = (-box_length, box_length, -box_length, box_length)
stride = 8
mark_recycling_planes!(axis) = vlines!(axis, [-1, 1] .* (box_length - recycling_distance); color = :limegreen, linestyle = :dash, linewidth = 2.5)

fig = Figure(size = (2000, 1750))
Label(fig[1, 1:4], "Velocity along x at mid-depth at the end of the run, u / u★. Dashed lines mark the recycling planes."; font = :bold, fontsize = 18)
Label(fig[3, 1:4], "Depth-integrated kinetic energy of the fluctuations about the time mean, ∫ k dz / (u★² H), with the depth-averaged mean flow as arrows";
      font = :bold, fontsize = 18)

for (column, name) in enumerate(keys(runs))
    result = results[name]
    axis = Axis(fig[2, column]; title = descriptions[name], xlabel = "x / H", ylabel = "y / H", aspect = DataAspect(), limits)
    heatmap!(axis, xnodes(grid, Face()), y, runs[name]["u"]; colormap = :balance, colorrange = (-1.3, 1.3) .* bulk_velocity)
    name == :laminar || mark_recycling_planes!(axis)

    axis = Axis(fig[4, column]; title = descriptions[name], xlabel = "x / H", ylabel = "y / H", aspect = DataAspect(), limits)
    heatmap!(axis, x, y, max.(result.kinetic_energy, 1e-1); colormap = :magma, colorscale = log10, colorrange = (1e-1, 1e3))
    arrows2d!(axis, x[stride÷2:stride:end], y[stride÷2:stride:end], result.U[stride÷2:stride:end, stride÷2:stride:end], result.V[stride÷2:stride:end, stride÷2:stride:end];
              lengthscale = 0.6 * stride * Δ / bulk_velocity, color = :white)
    name == :laminar || mark_recycling_planes!(axis)
end

Colorbar(fig[2, 5]; colormap = :balance, limits = (-1.3, 1.3) .* bulk_velocity, label = "u / u★")
Colorbar(fig[4, 5]; colormap = :magma, scale = log10, limits = (1e-1, 1e3), label = "∫ k dz / (u★² H)")

inflow_axis = Axis(fig[5, 1]; title = "Mean inflow at the boundaries (solid: west, dashed: east)", xlabel = "y / H", ylabel = "depth-averaged inward u / bulk velocity")
energy_axis = Axis(fig[5, 2]; title = "Kinetic energy of fluctuations, averaged over |y| < π H", xlabel = "x / H", ylabel = "∫ k dz / (u★² H)", yscale = log10)
variance_axis = Axis(fig[5, 3]; title = "Variance of u along y at mid-depth, next to the west boundary", xlabel = "t u★ / H", ylabel = "variance / u★²", yscale = log10)
transport_axis = Axis(fig[5, 4]; title = "Outflow (solid: north, dashed: south)", xlabel = "t u★ / H", ylabel = "transport out / target")

for name in keys(runs)
    result, transports, frames, color = results[name], runs[name]["transports"], runs[name]["frames"], colors[name]
    central_energy = vec(mean(result.kinetic_energy[:, N÷2+1:3N÷2], dims = 2))
    lines!(energy_axis, x, max.(central_energy, 1e-3); color, label = descriptions[name])
    lines!(variance_axis, frames.times, [mean(var(u[i, :]) for i in 1:8) for u in frames.u]; color)
    lines!(transport_axis, transports.times, transports.north ./ target_transport; color)
    lines!(transport_axis, transports.times, -transports.south ./ target_transport; color, linestyle = :dash)
    lines!(inflow_axis, y, result.U[1, :] ./ bulk_velocity; color)
    lines!(inflow_axis, y, -result.U[end, :] ./ bulk_velocity; color, linestyle = :dash)

    @printf("%-13s west transport / target between %.6f and %.6f, east between %.6f and %.6f, maximum divergence %.1e\n", name,
            extrema(transports.west ./ target_transport)..., extrema(transports.east ./ target_transport)..., runs[name]["maximum_divergence"])
end

Legend(fig[6, 1:4], energy_axis; orientation = :horizontal)
save("recycled_converging_inflow.png", fig)

#####
##### Animation of u and w at mid-depth
#####

frame = Observable(1)
frame_times = first(runs)["frames"].times
animation = Figure(size = (2000, 1250))
Label(animation[0, 1:length(runs)], @lift "The flow enters through the west and east boundaries (left and right edges) and leaves through the south and north boundaries. " *
                                          @sprintf("t = %.2f H / u★\n", frame_times[$frame]) *
                                          "Dashed lines mark the recycling planes. Each is sampled and fed back at the inflow boundary on its side."; fontsize = 20)
Label(animation[1, 1:length(runs)], "Velocity along x at mid-depth, u / u★"; font = :bold, fontsize = 18)
Label(animation[3, 1:length(runs)], "Vertical velocity at mid-depth, w / u★"; font = :bold, fontsize = 18)

for (column, name) in enumerate(keys(runs))
    frames = runs[name]["frames"]
    axis = Axis(animation[2, column]; title = descriptions[name], xlabel = "x / H", ylabel = "y / H", aspect = DataAspect(), limits)
    heatmap!(axis, xnodes(grid, Face()), y, @lift(frames.u[$frame]); colormap = :balance, colorrange = (-1.3, 1.3) .* bulk_velocity)
    name == :laminar || mark_recycling_planes!(axis)
    axis = Axis(animation[4, column]; title = descriptions[name], xlabel = "x / H", ylabel = "y / H", aspect = DataAspect(), limits)
    heatmap!(axis, x, y, @lift(frames.w[$frame]); colormap = :balance, colorrange = (-5, 5) .* u★)
    name == :laminar || mark_recycling_planes!(axis)
end

Colorbar(animation[2, length(runs) + 1]; colormap = :balance, limits = (-1.3, 1.3) .* bulk_velocity, label = "u / u★")
Colorbar(animation[4, length(runs) + 1]; colormap = :balance, limits = (-5, 5) .* u★, label = "w / u★")

record(animation, "recycled_converging_inflow.mp4", eachindex(frame_times); framerate = 25) do n
    frame[] = n
end
