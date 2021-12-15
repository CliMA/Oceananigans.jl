pushfirst!(LOAD_PATH, joinpath(@__DIR__, "..", ".."))

using Printf
using Plots
using Oceananigans
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBoundary

const κ = 1
const ν = 1
const nz = 3
const H = nz * 5
const topo = -H + 5

stop_time = 10.

underlying_grid = RectilinearGrid(size=nz, z = (-H,0), topology = (Flat, Flat, Bounded))

solid(x, y, z) = (z <= topo)

immersed_grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBoundary(solid))

model = NonhydrostaticModel(
                               advection = CenteredSecondOrder(),
                             timestepper = :RungeKutta3,
                                    grid = immersed_grid,
                                 tracers = (:b),
                                 closure = IsotropicDiffusivity(ν=ν, κ=κ),
                                buoyancy = BuoyancyTracer())

set!(model, b=1, u=0, v=0, w=0)

simulation = Simulation(model, Δt=1.0, stop_time=stop_time)

z = znodes(model.tracers.b)
b = view(interior(model.tracers.b), 1, 1, :)

# plot immersed wall and initial buoyancy
bplot = plot(b, ones(size(z),).*topo, label = "", color = :black, lw = 2, xlabel = "b", ylabel = "z", legend = :top)
plot!(bplot, b, z, lw = 3, label = "Initial", color = :blue)

run!(simulation)

Δb = abs(maximum(b .-1))
@info "Largest change in buoyancy is $(Δb)."

# plotting final buoyancy

plot!(bplot, b, z, lw = 3, label = "Final", color = :red, linestyle = :dash, title = @sprintf("Buoyancy, |Δb|ₘₐₓ = %.1f", Δb))

display(bplot)




