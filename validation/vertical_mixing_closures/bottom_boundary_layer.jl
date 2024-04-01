using GLMakie
using Oceananigans
using Oceananigans.Units
using Printf

using Oceananigans.TurbulenceClosures.CATKEVerticalDiffusivities:
    CATKEVerticalDiffusivity, MixingLength

# Parameters
Δz = 4          # Vertical resolution
Lz = 256        # Extent of vertical domain
Nz = Int(Lz/Δz) # Vertical resolution
f₀ = 0.0        # Coriolis parameter (s⁻¹)
N² = 1e-6       # Buoyancy gradient (s⁻²)
ℓ₀ = 1e-4       # Roughness length
ϰ  = 0.4        # "Von Karman constant"
u₀ = 1.0        # Initial bottom velocity
stop_time = 1days

mixing_length = MixingLength(Cᵇ=0.1)
catke = CATKEVerticalDiffusivity()

# Set up simulation

grid = RectilinearGrid(size=Nz, z=(0, Lz), topology=(Flat, Flat, Bounded))
coriolis = FPlane(f=f₀)

# Fluxes from similarity theory...
Cᴰ = (ϰ / log(Δz/2ℓ₀))^2
@inline τˣ(x, y, t, u, v, Cᴰ) = - Cᴰ * u * sqrt(u^2 + v^2)
@inline τʸ(x, y, t, u, v, Cᴰ) = - Cᴰ * v * sqrt(u^2 + v^2)

u_bottom_bc = FluxBoundaryCondition(τˣ, field_dependencies=(:u, :v), parameters=Cᴰ)
v_bottom_bc = FluxBoundaryCondition(τʸ, field_dependencies=(:u, :v), parameters=Cᴰ)

u_bcs = FieldBoundaryConditions(bottom = u_bottom_bc)
v_bcs = FieldBoundaryConditions(bottom = v_bottom_bc)

model = HydrostaticFreeSurfaceModel(; grid, closure, coriolis,
                                    tracers = (:b, :e),
                                    buoyancy = BuoyancyTracer(),
                                    boundary_conditions = (; u=u_bcs, v=v_bcs))
                                    
bᵢ(z) = N² * z
set!(model, b=bᵢ, u=u₀, e=1e-6)

simulation = Simulation(model; Δt=2minutes, stop_time)

diffusivities = (κᵘ = model.diffusivity_fields.κᵘ,
                 κᶜ = model.diffusivity_fields.κᶜ)

outputs = merge(model.velocities, model.tracers, diffusivities)

simulation.output_writers[:fields] = JLD2OutputWriter(model, outputs,
                                                      schedule = TimeInterval(20minutes),
                                                      filename = "bottom_boundary_layer.jld2",
                                                      overwrite_existing = true)

progress(sim) = @info string("Iter: ", iteration(sim), " t: ", prettytime(sim),
                             ", max(u): ", maximum(model.velocities.u))
simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))

@info "Running a simulation of $model..."

run!(simulation)

