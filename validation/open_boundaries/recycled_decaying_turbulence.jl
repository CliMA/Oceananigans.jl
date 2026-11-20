# Decaying two-dimensional turbulence carried by a uniform flow through an open channel with a recycled inflow.
# The channel is three periodic boxes long and starts from three copies of the box's initial state, so the
# stretch upstream of the recycling plane should evolve like the periodic box. Three recycled runs are compared
# with it:
#
#   * `averaging = nothing` with second-order centered advection, which keeps its order next to the open side.
#     With `averaging = 2` on `u` alone the result is the same, because the `y`-mean of `u` is the mean flow.
#   * `averaging = nothing` with WENO, which drops to low order in the cells next to the open side.
#   * `averaging = 2` on `u` and `v`, which removes the part of `v` that does not depend on `y`. The periodic box
#     keeps that part, so this run departs from it.
#
# The script writes a figure and an animation of the vorticity. A dashed line marks the recycling plane.

using Oceananigans
using CairoMakie
using Printf
using Random
using Statistics: mean

N = 64                 # cells across the periodic box
L = 2π                 # box size
U = 1                  # mean flow
tiles = 3              # boxes in the open channel
stop_time = 4L / U
save_interval = L / 32U

periodic_grid = RectilinearGrid(size = (N, N), x = (0, L), y = (0, L), topology = (Periodic, Periodic, Flat))
open_grid = RectilinearGrid(size = (tiles * N, N), x = (0, tiles * L), y = (0, L), topology = (Open, Periodic, Flat))

numerics = (centered = (advection = Centered(order = 2), closure = ScalarDiffusivity(ν = 5e-3)),
            weno = (advection = WENO(order = 5), closure = nothing))

experiments = (; absolute = (numerics = :centered, averaging = nothing, label = "averaging = nothing, Centered(order = 2)", color = :royalblue),
                 weno = (numerics = :weno, averaging = nothing, label = "averaging = nothing, WENO(order = 5)", color = :seagreen),
                 fluctuations = (numerics = :centered, averaging = 2, label = "averaging = 2, Centered(order = 2)", color = :firebrick))

# Random streamfunction ψ = Σ a cos(k x + l y + θ) with integer wavenumbers, so u = ∂y ψ and v = -∂x ψ have the period of the box
Random.seed!(1)
modes = [(k, l, 2π * rand()) for k in -6:6 for l in -6:6 if 3 ≤ hypot(k, l) ≤ 6]
u′(x, y) = - sum(l * sin(k * x + l * y + θ) for (k, l, θ) in modes)
v′(x, y) =   sum(k * sin(k * x + l * y + θ) for (k, l, θ) in modes)
a = 0.3U / sqrt(mean(u′(x, y)^2 for x in xnodes(periodic_grid, Face()), y in ynodes(periodic_grid, Center())))

function recycled_boundary_conditions(averaging)
    exterior_velocity = isnothing(averaging) ? 0 : U
    recycling(; kwargs...) = TurbulenceRecycling(; recycling_distance = L, averaging, kwargs...)
    u = FieldBoundaryConditions(west = NormalFlowBoundaryCondition(exterior_velocity; scheme = recycling(target_transport = U * L)),
                                east = NormalFlowBoundaryCondition(U; scheme = PerturbationAdvection()))
    v = FieldBoundaryConditions(west = ValueBoundaryCondition(0; scheme = recycling()))
    return (; u, v)
end

# Returns the saved times, vorticity snapshots and west transports
function run_decaying_turbulence(grid, numerics; boundary_conditions = NamedTuple())
    model = NonhydrostaticModel(grid; numerics.advection, numerics.closure, boundary_conditions)
    set!(model, u = (x, y) -> U + a * u′(x, y), v = (x, y) -> a * v′(x, y))

    u, v, w = model.velocities
    ζ = Field(∂x(v) - ∂y(u))
    times, vorticity, transport = Float64[], Array{Float64, 2}[], Float64[]

    function save_state(simulation)
        compute!(ζ)
        push!(times, time(simulation))
        push!(vorticity, Array(interior(ζ, :, :, 1)))
        push!(transport, sum(interior(u, 1, :, 1)) * L / N)
        return nothing
    end

    simulation = Simulation(model; Δt = 0.2 * L / N / 2U, stop_time, verbose = false)
    add_callback!(simulation, save_state, TimeInterval(save_interval))
    run!(simulation)

    return (; times, vorticity, transport)
end

periodic = map(n -> run_decaying_turbulence(periodic_grid, n), numerics)
recycled = map(e -> run_decaying_turbulence(open_grid, numerics[e.numerics]; boundary_conditions = recycled_boundary_conditions(e.averaging)), experiments)

# Vorticity is at faces in x, with one more point in the open channel than its three boxes hold
box(ζ, n) = ζ[(n - 1) * N .+ (1:N), :]
enstrophy(ζ) = mean(ζ.^2) / 2
relative_difference(vorticity, reference_vorticity) = sqrt(mean((box(vorticity, 1) .- reference_vorticity).^2) / mean(reference_vorticity.^2))
box_crossings(times) = times * U / L

fig = Figure(size = (1400, 1400))
x = xnodes(open_grid, Face())[1:tiles*N]
y = ynodes(open_grid, Face())
colorrange(ζ) = maximum(abs, ζ) .* (-1, 1)

periodic_vorticity = periodic.centered.vorticity[end]
axis = Axis(fig[1, 1:2]; title = "Periodic box, Centered(order = 2), repeated three times, at t U / L = $(stop_time * U / L)", ylabel = "y", aspect = DataAspect())
heatmap!(axis, x, y, repeat(periodic_vorticity, tiles, 1); colormap = :balance, colorrange = colorrange(periodic_vorticity))

for (row, name) in enumerate(keys(experiments))
    reference = periodic[experiments[name].numerics].vorticity[end]
    snapshot_axis = Axis(fig[row + 1, 1:2]; title = "Recycled inflow, $(experiments[name].label)", xlabel = "x", ylabel = "y", aspect = DataAspect())
    heatmap!(snapshot_axis, x, y, recycled[name].vorticity[end][1:tiles*N, :]; colormap = :balance, colorrange = colorrange(reference))
    vlines!(snapshot_axis, L; color = :limegreen, linestyle = :dash, linewidth = 2.5)
end

enstrophy_axis = Axis(fig[5, 1]; title = "Enstrophy by box (solid: first, dashed: second, dotted: third)", xlabel = "t U / L", ylabel = "⟨ζ²⟩ / 2", yscale = log10)
difference_axis = Axis(fig[5, 2]; title = "Vorticity difference from the periodic box, in the first box", xlabel = "t U / L", ylabel = "rms(ζ - ζ periodic) / rms(ζ periodic)", yscale = log10)

for (name, linewidth) in ((:centered, 5), (:weno, 2))
    lines!(enstrophy_axis, box_crossings(periodic[name].times), enstrophy.(periodic[name].vorticity); color = :black, linewidth, label = "periodic, $name")
end

for name in keys(experiments)
    experiment, result = experiments[name], recycled[name]
    reference = periodic[experiment.numerics]

    for (n, linestyle) in zip(1:tiles, (:solid, :dash, :dot))
        label = n == 1 ? experiment.label : nothing
        lines!(enstrophy_axis, box_crossings(result.times), [enstrophy(box(ζ, n)) for ζ in result.vorticity]; experiment.color, linestyle, label)
    end

    difference = relative_difference.(result.vorticity, reference.vorticity)
    lines!(difference_axis, box_crossings(result.times)[2:end], difference[2:end]; experiment.color, experiment.label)

    after(crossings) = difference[findfirst(≥(crossings * L / U), result.times)]
    @printf("%-42s west transport / (U L) between %.12f and %.12f, vorticity difference after 1/16, 1 and 4 box crossings: %.1e, %.1e, %.1e\n",
            experiment.label, minimum(result.transport) / (U * L), maximum(result.transport) / (U * L), after(1/16), after(1), difference[end])
end

axislegend(enstrophy_axis; position = :lb)
axislegend(difference_axis; position = :rb)

save("recycled_decaying_turbulence.png", fig)

#####
##### Animation of the vorticity, with the color range following the decay of the periodic run
#####

frame = Observable(1)
animation = Figure(size = (1300, 1900))
Label(animation[0, 1], @lift "Vorticity at t U / L = " * string(round(box_crossings(periodic.centered.times[$frame]), digits = 2)) * " (dashed: recycling plane)";
      fontsize = 22, tellwidth = false)

periodic_axis = Axis(animation[1, 1]; title = "Periodic box, Centered(order = 2), repeated three times", ylabel = "y", aspect = DataAspect())
heatmap!(periodic_axis, x, y, @lift(repeat(periodic.centered.vorticity[$frame], tiles, 1)); colormap = :balance,
         colorrange = @lift(colorrange(periodic.centered.vorticity[$frame])))

for (row, name) in enumerate(keys(experiments))
    reference = periodic[experiments[name].numerics].vorticity
    recycled_axis = Axis(animation[row + 1, 1]; title = "Recycled inflow, $(experiments[name].label)", xlabel = "x", ylabel = "y", aspect = DataAspect())
    heatmap!(recycled_axis, x, y, @lift(recycled[name].vorticity[$frame][1:tiles*N, :]); colormap = :balance, colorrange = @lift(colorrange(reference[$frame])))
    vlines!(recycled_axis, L; color = :limegreen, linestyle = :dash, linewidth = 2.5)
end

record(animation, "recycled_decaying_turbulence.mp4", eachindex(periodic.centered.times); framerate = 16) do n
    frame[] = n
end
