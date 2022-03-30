using Oceananigans
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBoundary, PartialCellBottom, mask_immersed_field!
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Printf


underlying_grid = RectilinearGrid(arch,
                                  size=(128, 64), halo=(3, 3), 
                                  y(-1, 1), z=(-1, 0),
                                  topology=(Flat, Periodic, Bounded))

# A bump
h₀ = 0.5 # bump height
L = 0.25 # bump width
@inline h(y) = h₀ * exp(- y^2 / L^2)
@inline seamount(x, y) = - 1 + h(y)

#grid = ImmersedBoundaryGrid(underlying_grid, PartialCellBottom(seamount, 0.1))
grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(seamount))

# Terrain following coordinate
ζ(y, z) = z / (h(y) - 1)

# Calculate streamfunction
Ψᵢ(x, y, z) = (1 - ζ(y, z))^2
Ψ = Field{Center, Face, Face}(grid)
set!(Ψ, Ψᵢ)
fill_halo_regions!(Ψ, arch)
mask_immersed_field!(Ψ)

# Set velocity field from streamfunction
v = YFaceField(grid)
w = ZFaceField(grid)
v .= + ∂z(Ψ)
w .= - ∂y(Ψ)

fill_halo_regions!(v, arch)
fill_halo_regions!(w, arch)
# mask_immersed_field!(v)
# mask_immersed_field!(w)

D = compute!(Field(∂y(v) + ∂z(w)))
@info @sprintf("Maximum of pointwise divergence is %.2e.", maximum(D))

## Set up Model
velocities = PrescribedVelocityFields(; v, w)
model = HydrostaticFreeSurfaceModel(; grid, velocities,
                                    tracer_advection = WENO5(),
                                    tracers = :θ,
                                    buoyancy = nothing)

θᵢ(x, y, z) = 1 + z
set!(model, θ = θᵢ)

# Simulation                             
stop_time = 1.0
Δt = 1e-3
simulation = Simulation(model; Δt, stop_time)

progress(s) = @info @sprintf("[%.2f%%], iteration: %d, time: %.3f, max|θ|: %.2e",
                             100 * s.model.clock.time / s.stop_time, s.model.clock.iteration,
                             s.model.clock.time, maximum(abs, model.tracers.θ))

progress(sim) = @info "Iteration: $(iteration(sim)), time: $(time(sim))"
simulation.callbacks[:progress] = Callback(progress, IterationInterval(10))

simulation.output_writers[:fields] = JLD2OutputWriter(model, model.tracers,
                                                      schedule = TimeInterval(0.02),
                                                      prefix = "tracer_advection_over_bump",
                                                      force = true)

run!(simulation)

