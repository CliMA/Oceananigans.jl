using Oceananigans, Oceananigans.Units, StructArrays

grid = RectilinearGrid(size = (32, 32, 16), x = (-100, 100), y = (-100, 100), z = (-25, 0))

struct CTrackingParticle{T}
    x :: T
    y :: T
    z :: T
    c :: T
end

n_particles = 41

drogue_depths = -20:20/(n_particles-1):0

c = CenterField(grid)

particle_properties = StructArray{CTrackingParticle}((zeros(n_particles), 
                                                      zeros(n_particles), 
                                                      zeros(n_particles), 
                                                      zeros(n_particles)))

particles = LagrangianParticles(particle_properties; dynamics = DroguedDynamics(drogue_depths), tracked_fields = (; c))

ρₒ = 1024
u₁₀ = 10    # m s⁻¹, average wind velocity 10 meters above the ocean
cᴰ = 2.5e-3 # dimensionless drag coefficient
ρₐ = 1.225  # kg m⁻³, average density of air at sea-levelOce

τx = - ρₐ / ρₒ * cᴰ * u₁₀ * abs(u₁₀) # m² s⁻²

u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(τx))

coriolis = FPlane(latitude = 45)
closure = ScalarDiffusivity(κ = 1e-4, ν = 1e-4)
advection = UpwindBiased()

model = NonhydrostaticModel(; grid, particles, boundary_conditions = (; u = u_bcs), coriolis, closure, advection, tracers = (; c))

# Random noise damped at top and bottom
Ξ(z) = randn() * z / model.grid.Lz * (1 + z / model.grid.Lz) # noise

# Velocity initial condition: random noise scaled by the friction velocity.
uᵢ(x, y, z) = sqrt(abs(τx)) * 1e-3 * Ξ(z)

set!(model, u = uᵢ)

simulation = Simulation(model, Δt = 10, stop_time = 4hours)

conjure_time_step_wizard!(simulation, IterationInterval(100), cfl = 0.5, diffusive_cfl = 0.5)

simulation.output_writers[:velocities] = JLD2Writer(model, model.velocities;
                                                    overwrite_existing = true, 
                                                    filename = "drogued_velocities.jld2", 
                                                    schedule = TimeInterval(10minutes))

prog(sim) = @info prettytime(sim) * " in " * prettytime(sim.run_wall_time) * " with Δt = " * prettytime(sim.Δt)

add_callback!(simulation, prog, IterationInterval(50))

run!(simulation)

simulation.output_writers[:particles] = JLD2Writer(model, (; particles);
                                                   overwrite_existing = true, 
                                                   filename = "drogued_particles.jld2", 
                                                   schedule = TimeInterval(0.1minutes))
simulation.output_writers[:tracer] = JLD2Writer(model, model.tracers;
                                                overwrite_existing = true, 
                                                filename = "drogued_tracer.jld2", 
                                                schedule = TimeInterval(0.1minutes))

particles.properties.x .= 0 
particles.properties.y .= 0 

simulation.stop_time += 0.2hours

set!(model, c = (x, y, z) -> exp(-(x^2+y^2)/(2π * 10^2)) * max(0, 1+z/10))

run!(simulation)

c = FieldTimeSeries("drogued_tracer.jld2", "c");

using JLD2, CairoMakie

file = jldopen("drogued_particles.jld2")

iterations = keys(file["timeseries/particles"])[2:end]

n = Observable(1)

fig = Figure(size=(1000, 1200));

ax = Axis(fig[1:4, 1], ylabel = "y (m)", aspect = DataAspect(), title = (@lift "Surface tracer concentration " *  prettytime(c.times[$n])))
ax2 = Axis(fig[5, 1], xlabel = "x (m)", ylabel = "z (m)",  aspect = DataAspect(), title = "Maximum tracer concentration")

c_plt = @lift interior(c[$n], :, :, grid.Nz)
c_slice_plt = @lift maximum(c[$n], dims = 2)[1:grid.Nx, 1, 1:grid.Nz]

x_plt = @lift file["timeseries/particles/$(iterations[$n])"].x
y_plt = @lift file["timeseries/particles/$(iterations[$n])"].y
particle_c_plt = @lift file["timeseries/particles/$(iterations[$n])"].c

heatmap!(ax, xnodes(c), ynodes(c), c_plt, colorrange = (0, 1), alpha = 0.5)
scatter!(ax, x_plt, y_plt, color = particle_c_plt, colorrange = (0, 1))
heatmap!(ax2, xnodes(c), znodes(c), c_slice_plt, colorrange = (0, 1), alpha = 0.5)
scatter!(ax2, x_plt, drogue_depths, color = particle_c_plt, colorrange = (0, 1))

