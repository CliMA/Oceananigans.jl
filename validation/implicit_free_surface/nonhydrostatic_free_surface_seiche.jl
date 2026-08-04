# A standing surface wave (seiche) in a closed basin, simulated with `NonhydrostaticModel`
# and an implicit free surface on three grids:
#
#   1. a regular grid, where the pressure Poisson equation with its free-surface Robin
#      boundary condition is solved directly with `FourierTridiagonalPoissonSolver`,
#   2. a stretched-x grid, where it is solved with `ConjugateGradientPoissonSolver`
#      (`FreeSurfaceLaplacian` operator, deflated Fourier-tridiagonal preconditioner),
#   3. an immersed boundary grid with a flat bottom, also solved with CG.
#
# All three are compared against the analytical standing-wave solution
#
#     η(x, t) = a cos(kx) cos(ωt),    ω² = g k tanh(kH),
#
# which the nonhydrostatic model captures including dispersion.

using Oceananigans
using Oceananigans.Models.HydrostaticFreeSurfaceModels: ImplicitFreeSurface
using Printf

L = 10      # basin length
H = 5       # basin depth
g = 9.81
a = 1e-3    # wave amplitude (linear regime)

k = π / L
ω = sqrt(g * k * tanh(k * H))
wave_period = 2π / ω

function run_seiche(grid; Δt=wave_period/200, stop_time=2wave_period)
    free_surface = ImplicitFreeSurface(gravitational_acceleration=g)
    model = NonhydrostaticModel(grid; free_surface)
    set!(model.free_surface.displacement, (x, z) -> a * cos(k * x))

    simulation = Simulation(model; Δt, stop_time)

    t = Float64[]
    η = Float64[]
    η₁ = model.free_surface.displacement
    record_displacement(sim) = (push!(t, time(sim)); push!(η, first(interior(η₁))))
    add_callback!(simulation, record_displacement, IterationInterval(1))

    run!(simulation)
    return t, η, model
end

grid_regular = RectilinearGrid(size=(64, 32), x=(0, L), z=(-H, 0), topology=(Bounded, Flat, Bounded))

# Hyperbolically stretched x, refined towards the left wall
stretching(ξ) = L * (sinh(2ξ) / sinh(2))
grid_stretched = RectilinearGrid(size=(64, 32), x=stretching.(range(0, 1, length=65)),
                                 z=(-H, 0), topology=(Bounded, Flat, Bounded))

# Immersed flat bottom at z = -H inside a deeper underlying grid
underlying = RectilinearGrid(size=(64, 48), x=(0, L), z=(-1.5H, 0), topology=(Bounded, Flat, Bounded))
grid_immersed = ImmersedBoundaryGrid(underlying, GridFittedBottom(x -> -H))

results = Dict()
for (name, grid) in ("regular" => grid_regular, "stretched" => grid_stretched, "immersed" => grid_immersed)
    t, η, model = run_seiche(grid)
    x₁ = first(xnodes(grid, Center()))
    results[name] = (; t, η, x₁)
    @info @sprintf("%9s grid: pressure solver %s", name, nameof(typeof(model.pressure_solver)))
end

# Compare with the analytical solution at the recording point
for name in ("regular", "stretched", "immersed")
    t, η, x₁ = results[name]
    η_analytical = @. a * cos(k * x₁) * cos(ω * t)
    error = maximum(abs, η .- η_analytical) / a
    @info @sprintf("%9s grid: max|η - η_analytical| / a = %.4f over %.1f wave periods", name, error, last(t) / wave_period)
end

#=
using GLMakie

fig = Figure()
ax = Axis(fig[1, 1], xlabel="t / T", ylabel="η(x₁) / a")
for name in ("regular", "stretched", "immersed")
    t, η = results[name]
    lines!(ax, t ./ wave_period, η ./ a, label=name)
end
t = results["regular"].t
x₁ = xnodes(grid_regular, Center())[1]
lines!(ax, t ./ wave_period, cos(k * x₁) .* cos.(ω .* t), linestyle=:dash, color=:black, label="analytical")
axislegend(ax)
fig
=#
