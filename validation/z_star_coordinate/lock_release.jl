using Oceananigans
using Oceananigans.Grids
using Oceananigans.Units
using Oceananigans.Utils: prettytime
using Oceananigans.Advection: WENOVectorInvariant
using Oceananigans.AbstractOperations: GridMetricOperation
using Oceananigans.DistributedComputations
using Printf
using GLMakie

arch = CPU() # Distributed(CPU())#; synchronized_communication=true) 
z_faces = MutableVerticalDiscretization((-20, 0))

grid = RectilinearGrid(arch; 
                       size = (128, 20),
                          x = (0, 64kilometers),
                          z = z_faces,
                       halo = (6, 6),
                   topology = (Bounded, Flat, Bounded))

bottom(x) = x < 52kilometers && x > 45kilometers ? -10 : -20
grid  = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom))

model = HydrostaticFreeSurfaceModel(; grid,
                         momentum_advection = Centered(),
                           tracer_advection = Centered(),
                                   buoyancy = BuoyancyTracer(),
                                    tracers = (:b, :c),
                                    closure = HorizontalScalarDiffusivity(κ=100, ν=100),
                                timestepper = :SplitRungeKutta3,
                               free_surface = ExplicitFreeSurface()) # SplitExplicitFreeSurface(grid; substeps=30)) # ImplicitFreeSurface()) # #

g = model.free_surface.gravitational_acceleration
bᵢ(x, z) = x > 20kilometers ? 6 // 100 : 1 // 100
set!(model, b = bᵢ, c = (x, z) -> - 1)

# Same timestep as in the ilicak paper
Δt = 0.1 # Oceananigans.defaults.FloatType(1 // 10)

@info "the time step is $Δt"

simulation = Simulation(model; Δt, stop_time=1.7hours)

Δz = zspacings(grid, Center(), Center(), Center())
V  = KernelFunctionOperation{Center, Center, Center}(Oceananigans.Operators.Vᶜᶜᶜ, grid)

field_outputs = merge(model.velocities, model.tracers, (; Δz))

simulation.output_writers[:other_variables] = JLD2Writer(model, field_outputs,
                                                         overwrite_existing = true,
                                                         schedule = IterationInterval(100),
                                                         filename = "zstar_model")

# The two different estimates of the free-surface
et1 = []
et2 = []

# Initial conditions
bav = [sum(model.tracers.b * V) / sum(V)]
cav = [sum(model.tracers.c * V) / sum(V)]
vav = [sum(V)]
mxc = [maximum(model.tracers.c)]
mnc = [minimum(model.tracers.c)]

function progress(sim)
    w  = interior(sim.model.velocities.w, :, :, sim.model.grid.Nz+1)
    u  = sim.model.velocities.u
    Nx = size(grid, 1)

    push!(bav, sum(model.tracers.b * V) / sum(V))
    push!(cav, sum(model.tracers.c * V) / sum(V))
    push!(vav, sum(V))
    push!(mxc, maximum(model.tracers.c))
    push!(mnc, minimum(model.tracers.c))

    Δη = maximum(abs, interior(model.free_surface.η, :, 1, 1) .- model.grid.z.ηⁿ[1:Nx, 1, 1])

    msgn = @sprintf("") #Rank: %d, ", arch.local_rank)
    msg0 = @sprintf("Time: %s, ", prettytime(sim.model.clock.time))
    msg1 = @sprintf("extrema w: %.2e %.2e ",  maximum(w),  minimum(w))
    msg2 = @sprintf("drift b: %6.3e ", bav[end] - bav[1])
    msg4 = @sprintf("drift c: %6.3e ", cav[end] - cav[1])

    push!(et1, deepcopy(interior(model.free_surface.η, :, 1, 1)))
    push!(et2, deepcopy(model.grid.z.ηⁿ[1:Nx, 1, 1]))

    @handshake @info msgn * msg0 * msg1 * msg2 * msg4

    return nothing
end

simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))

run!(simulation)
 
b = model.tracers.b
x, y, z = nodes(b)

function running_mean(v, points)
    n  = length(v)
    rm = zeros(length(v) - 2points+1)
    for i in points+1:n-points
        rm[i-points] = mean(v[i - points:i+points])
    end
    return rm[1:end-1]
end

fig = Figure()
ax  = Axis(fig[1, 1], title = "Integral property conservation")
lines!(ax, (vav .- vav[1]) ./ vav[1], label = "Volume anomaly")
lines!(ax, (bav .- bav[1]) ./ bav[1], label = "Buoyancy anomaly")
lines!(ax, (cav .- cav[1]) ./ cav[1], label = "Tracer anomaly")
axislegend(ax, position=:lt)
ax  = Axis(fig[1, 2], title = "Final buoyancy field") 
contourf!(ax, x, z, interior(model.tracers.b, :, 1, :), colormap=:balance, levels=20)
vlines!(ax, 62.3e3, linestyle = :dash, linewidth = 3, color = :black)
