using Oceananigans
using Oceananigans.Advection: VelocityStencil
using Oceananigans.MultiRegion: reconstruct_global_field, multi_region_object_from_array
# using GLMakie

arch = CPU()
Nh   = 512
Nz   = 1
grid = RectilinearGrid(arch, size=(Nh, Nh, Nz), halo=(4, 4, 4), x=(0, 2π), y=(0, 2π), z=(0, 1), topology=(Periodic, Periodic, Bounded))
mrg  = MultiRegionGrid(grid, partition=XPartition(2))

Δh = 2π / grid.Nx
Δt = 0.1 * Δh

ϵ(x, y, z)  =  2rand() - 1
u_init = Array(interior(set!(Field((Face, Center, Center), grid), ϵ)))
v_init = Array(interior(set!(Field((Face, Center, Center), grid), ϵ)))

u_init_mrg = multi_region_object_from_array(u_init, mrg)
v_init_mrg = multi_region_object_from_array(v_init, mrg)

momentum_advection = WENO5()
# momentum_advection = WENO5(vector_invariant=VelocityStencil())

free_surface = ImplicitFreeSurface(gravitational_acceleration=1, solver_method = :HeptadiagonalIterativeSolver)
# free_surface = ExplicitFreeSurface(gravitational_acceleration=1) 

progress(sim) = @info "Iteration: $(iteration(sim)), time: $(time(sim))"

#####
##### Running and comparing the two models
#####

#### Multi region model ----------------------------------------------------------

model_1 = HydrostaticFreeSurfaceModel(; grid = mrg, momentum_advection, free_surface,
                                    tracers = (),
                                    buoyancy = nothing,
                                    closure = ScalarDiffusivity(ν=1e-4))

set!(model_1, u=u_init_mrg, v=v_init_mrg)

simulation = Simulation(model_1; Δt, stop_iteration=10)
run!(simulation)

simulation.stop_iteration += 1000

simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))

start_time = time_ns()
run!(simulation)
elapsed_time = 1e-9 * (time_ns() - start_time)
@info "Simulation ran for " * prettytime(elapsed_time)

u, v, w = model_1.velocities

u_1 = reconstruct_global_field(u)
v_1 = reconstruct_global_field(v)

ζ_1 = compute!(Field(∂x(v_1) - ∂y(u_1)))

#### Single region model ----------------------------------------------------------

model_2 = HydrostaticFreeSurfaceModel(; grid, momentum_advection,
                                    tracers = (),
                                    buoyancy = nothing,
                                    free_surface = ImplicitFreeSurface(gravitational_acceleration=1),
                                    closure = ScalarDiffusivity(ν=1e-4))

set!(model_2, u=u_init, v=v_init)

simulation = Simulation(model_2; Δt, stop_iteration=10)
run!(simulation)

simulation.stop_iteration += 1000

progress(sim) = @info "Iteration: $(iteration(sim)), time: $(time(sim))"
simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))

start_time = time_ns()
run!(simulation)
elapsed_time = 1e-9 * (time_ns() - start_time)
@info "Simulation ran for " * prettytime(elapsed_time)

u_2, v_2, w = model_2.velocities

ζ_2 = compute!(Field(∂x(v_2) - ∂y(u_2)))

# fig = Figure()
# ax = Axis(fig[1, 1])
# heatmap!(ax, interior(ζ, :, :, 1))
# display(fig)