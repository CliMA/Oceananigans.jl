using Oceananigans
using Oceananigans.Grids
using Oceananigans.Units
using Oceananigans.Utils: prettytime
using Oceananigans.Advection: WENOVectorInvariant
using Oceananigans.AbstractOperations: GridMetricOperation
using Oceananigans.DistributedComputations
using Printf

arch    = CPU() 
z_faces = MutableVerticalDiscretization((-20, 0))

grid = RectilinearGrid(arch; 
                       size = (512, 100),
                          x = (0, 1kilometers),
                          z = z_faces,
                       halo = (6, 6),
                   topology = (Periodic, Flat, Bounded))


gaussian(x, L) = exp(-x^2 / 2L^2)
bottom(x) = -20 + 2 * gaussian(x - x₀, L)
grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom))

L = grid.Lx / 40 # gaussian width
x₀ = grid.Lx / 2 # gaussian center

@inline ηᴳ(x, z) = 2 * gaussian(x - x₀, L)

@inline function precipitation(i, j, k, grid, clock, fields)
    η = fields.η
    x = xnode(i, grid, Center())
    z = znode(k, grid, Center())

    return 1 / 100 * (ηᴳ(x, z) - η[i, j, k])
end

Fe = Forcing(precipitation, discrete_form=true)

model = HydrostaticFreeSurfaceModel(; grid,
                         momentum_advection = WENO(order=5),
                           tracer_advection = WENO(order=5),
                                   buoyancy = BuoyancyTracer(),
                                    tracers = (:b, :c),
                                    # forcing = (; η = Fe),
                                timestepper = :SplitRungeKutta3,
                               free_surface = SplitExplicitFreeSurface(grid; substeps=40)) 

g = model.free_surface.gravitational_acceleration
bᵢ(x, z) = z > -10 ? 0.06 : 0.01
set!(model, b = bᵢ, c = (x, z) -> 1, η = ηᴳ)
Oceananigans.BoundaryConditions.fill_halo_regions!(model.free_surface.η)

parent(model.grid.z.ηⁿ) .= parent(model.free_surface.η)
Oceananigans.Models.HydrostaticFreeSurfaceModels.ab2_step_grid!(grid, model, model.vertical_coordinate, 0, 0)
Oceananigans.Models.HydrostaticFreeSurfaceModels.ab2_step_grid!(grid, model, model.vertical_coordinate, 0, 0)
Oceananigans.Models.HydrostaticFreeSurfaceModels.ab2_step_grid!(grid, model, model.vertical_coordinate, 0, 0)
Δt = 1

@info "the time step is $Δt"

simulation = Simulation(model; Δt, stop_time=50hours)

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
    msg3 = @sprintf("max Δη: %6.3e ", Δη)
    msg4 = @sprintf("extrema c: %.2e %.2e ", mxc[end]-1, mnc[end]-1)

    push!(et1, deepcopy(interior(model.free_surface.η, :, 1, 1)))
    push!(et2, deepcopy(model.grid.z.ηⁿ[1:Nx, 1, 1]))

    @handshake @info msgn * msg0 * msg1 * msg2 * msg3 * msg4

    return nothing
end

simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))

run!(simulation)
 
b = model.tracers.b
x, y, z = nodes(b)

fig = Figure()
ax  = Axis(fig[1, 1], title = "Integral property conservation")
lines!(ax, (vav .- vav[1]) ./ vav[1], label = "Volume anomaly")
lines!(ax, (bav .- bav[1]) ./ bav[1], label = "Buoyancy anomaly")
lines!(ax, (cav .- cav[1]) ./ cav[1], label = "Tracer anomaly")
axislegend(ax, position=:lt)
ax  = Axis(fig[1, 2], title = "Final buoyancy field") 
contourf!(ax, x, z, interior(model.tracers.b, :, 1, :), colormap=:balance, levels=20)
vlines!(ax, 62.3e3, linestyle = :dash, linewidth = 3, color = :black)

b = FieldTimeSeries("zstar_model.jld2", "b")
bn = @lift(interior(b[$iter], :, 1, :))

fig = Figure()
ax  = Axis(fig[1, 1]) 
heatmap!(ax, x, z, bn)

c = FieldTimeSeries("zstar_model.jld2", "c")
cn = @lift(interior(c[$iter], :, 1, :))

fig = Figure()
ax  = Axis(fig[1, 1]) 
heatmap!(ax, x, z, cn)
