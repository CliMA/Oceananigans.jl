# Decaying two-dimensional turbulence carried by a uniform flow through an open channel with a recycled inflow,
# past a small no-slip cylinder. The channel is three periodic boxes long and starts from three copies of the box's
# initial state. The cylinder is at x = 12, downstream of the recycling plane, and the stretch upstream of the plane
# is compared with the periodic box, which has no cylinder. Two recycled inflows are compared with it:
#
#   * `averaging = nothing`, which imposes the recycling plane as is. With `averaging = 2` on `u` alone the result
#     is the same, because the `y`-mean of `u` is the mean flow.
#   * `averaging = 2` on `u` and `v`, which removes the part of `v` that does not depend on `y`. The periodic box
#     keeps that part, so these runs depart from it.
#
# Each is run with second-order centered advection, which keeps its order next to the open side, and with WENO,
# which drops to low order in the cells next to the open side. The WENO runs have no viscosity, so there the
# cylinder only blocks the flow.
#
# The script writes a figure and an animation of the vorticity. A dashed line marks the recycling plane and the
# cylinder is gray.

using Oceananigans
using Oceananigans.Solvers: ConjugateGradientPoissonSolver
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
cylinder_center = (12, L / 2)
cylinder_radius = 0.5

inside_cylinder(x, y) = hypot(x - cylinder_center[1], y - cylinder_center[2]) ≤ cylinder_radius

periodic_grid = RectilinearGrid(size = (N, N), x = (0, L), y = (0, L), topology = (Periodic, Periodic, Flat))
underlying_grid = RectilinearGrid(size = (tiles * N, N), halo = (4, 4), x = (0, tiles * L), y = (0, L), topology = (Open, Periodic, Flat))
open_grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBoundary(inside_cylinder))

numerics = (centered = (advection = Centered(order = 2), closure = ScalarDiffusivity(ν = 5e-3), label = "Centered(order = 2)"),
            weno = (advection = WENO(order = 5), closure = nothing, label = "WENO(order = 5)"))

inflows = (absolute = (averaging = nothing, label = "averaging = nothing", description = "Recycled inflow, plane as is (averaging = nothing)"),
           fluctuations = (averaging = 2, label = "averaging = 2", description = "Recycled inflow, mean flow + fluctuations about the y-mean (averaging = 2)"))

colors = (centered = (absolute = :royalblue, fluctuations = :firebrick), weno = (absolute = :seagreen, fluctuations = :darkorange))

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
                                east = NormalFlowBoundaryCondition(U; scheme = PerturbationAdvection()), immersed = ValueBoundaryCondition(0))
    v = FieldBoundaryConditions(west = ValueBoundaryCondition(0; scheme = recycling()), immersed = ValueBoundaryCondition(0))
    return (; u, v)
end

# Returns the saved times, vorticity snapshots and west transports
function run_decaying_turbulence(grid, numerics; boundary_conditions = NamedTuple(), pressure_solver = nothing)
    model = NonhydrostaticModel(grid; numerics.advection, numerics.closure, boundary_conditions, pressure_solver)
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

    # Saving every so many steps keeps steps from being shortened to hit save times, which can make them tiny
    add_callback!(simulation, save_state, IterationInterval(round(Int, save_interval / simulation.Δt)))
    run!(simulation)

    return (; times, vorticity, transport)
end

periodic = map(n -> run_decaying_turbulence(periodic_grid, n), numerics)
recycled = map(n -> map(inflow -> run_decaying_turbulence(open_grid, n; boundary_conditions = recycled_boundary_conditions(inflow.averaging),
                                                          pressure_solver = ConjugateGradientPoissonSolver(open_grid)), inflows), numerics)

# Vorticity is at faces in x, with one more point in the open channel than its three boxes hold
box(ζ, n) = ζ[(n - 1) * N .+ (1:N), :]
enstrophy(ζ) = mean(ζ.^2) / 2
relative_difference(vorticity, reference_vorticity) = sqrt(mean((box(vorticity, 1) .- reference_vorticity).^2) / mean(reference_vorticity.^2))
box_crossings(times) = times * U / L

x = xnodes(open_grid, Face())[1:tiles*N]
y = ynodes(open_grid, Face())
times = periodic.centered.times

# Vorticity at the saved state `frame`, an observable. Each column holds one advection scheme, with the periodic box
# on top and the recycled runs below. The color range of a column follows the decay of its periodic run.
function vorticity_panels!(figure, frame)
    for (column, advection) in enumerate(keys(numerics))
        reference = @lift periodic[advection].vorticity[$frame]
        colorrange = @lift maximum(abs, $reference) .* (-1, 1)

        Label(figure[1, column], "Advection: $(numerics[advection].label)"; font = :bold, fontsize = 20, tellwidth = false)
        axis = Axis(figure[2, column]; title = "Periodic box, shown $tiles times side by side", ylabel = "y", aspect = DataAspect())
        heatmap!(axis, x, y, @lift(repeat($reference, tiles, 1)); colormap = :balance, colorrange)

        for (row, inflow) in enumerate(keys(inflows))
            vorticity = @lift recycled[advection][inflow].vorticity[$frame][1:tiles*N, :]
            axis = Axis(figure[row + 2, column]; title = inflows[inflow].description, xlabel = "x", ylabel = "y", aspect = DataAspect())
            heatmap!(axis, x, y, vorticity; colormap = :balance, colorrange)
            vlines!(axis, L; color = :limegreen, linestyle = :dash, linewidth = 2.5)
            poly!(axis, Circle(Point2f(cylinder_center...), cylinder_radius); color = :gray50)
        end
    end

    return nothing
end

fig = Figure(size = (1900, 1700))
Label(fig[0, 1:2], "Vorticity at t U / L = $(stop_time * U / L). Dashed lines mark the recycling plane. The cylinder is gray."; fontsize = 22)
vorticity_panels!(fig, Observable(lastindex(times)))

enstrophy_axis = Axis(fig[5, 1]; title = "Enstrophy by box (solid: first, dashed: second, which holds the cylinder, dotted: third)", xlabel = "t U / L", ylabel = "⟨ζ²⟩ / 2", yscale = log10)
difference_axis = Axis(fig[5, 2]; title = "Vorticity difference from the periodic box, in the first box", xlabel = "t U / L", ylabel = "rms(ζ - ζ periodic) / rms(ζ periodic)", yscale = log10)

for (advection, linewidth) in ((:centered, 5), (:weno, 2))
    lines!(enstrophy_axis, box_crossings(periodic[advection].times), enstrophy.(periodic[advection].vorticity); color = :black, linewidth, label = "Periodic box, $(numerics[advection].label)")
end

for advection in keys(numerics), inflow in keys(inflows)
    result, reference, color = recycled[advection][inflow], periodic[advection], colors[advection][inflow]
    label = "$(inflows[inflow].label), $(numerics[advection].label)"

    for (n, linestyle) in zip(1:tiles, (:solid, :dash, :dot))
        lines!(enstrophy_axis, box_crossings(result.times), [enstrophy(box(ζ, n)) for ζ in result.vorticity]; color, linestyle, label = n == 1 ? label : nothing)
    end

    difference = relative_difference.(result.vorticity, reference.vorticity)
    lines!(difference_axis, box_crossings(result.times)[2:end], difference[2:end]; color, label)

    after(crossings) = difference[argmin(abs.(result.times .- crossings * L / U))]
    @printf("%-42s west transport / (U L) between %.12f and %.12f, vorticity difference after 1/16, 1 and 4 box crossings: %.1e, %.1e, %.1e\n",
            label, minimum(result.transport) / (U * L), maximum(result.transport) / (U * L), after(1/16), after(1), difference[end])
end

axislegend(enstrophy_axis; position = :lb)
axislegend(difference_axis; position = :rb)

save("recycled_decaying_turbulence.png", fig)

#####
##### Animation of the vorticity
#####

frame = Observable(1)
animation = Figure(size = (1900, 1300))
Label(animation[0, 1:2], @lift "Vorticity at t U / L = " * @sprintf("%.2f", box_crossings(times[$frame])) * ". The flow goes from left to right. " *
                               "Dashed lines mark the recycling plane, which is sampled and fed back at the inflow (x = 0). The cylinder is gray."; fontsize = 22)
vorticity_panels!(animation, frame)

record(animation, "recycled_decaying_turbulence.mp4", eachindex(times); framerate = 16) do n
    frame[] = n
end
