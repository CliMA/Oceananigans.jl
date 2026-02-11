# MWE: PrescribedFreeSurface with PrescribedVelocityFields and MutableVerticalDiscretization (ZStar)
#
# Tests that:
# 1. Model construction succeeds with PrescribedFreeSurface + PrescribedVelocityFields + ZStar grid
# 2. σ scaling factors update correctly during time stepping
# 3. After a full period of η(t) = A*sin(2πt), σ returns to 1.0

using Oceananigans
using Oceananigans.Grids: MutableVerticalDiscretization

# Small grid with MutableVerticalDiscretization (triggers ZStarCoordinate)
H = 10
z_faces = MutableVerticalDiscretization((-H, 0))

grid = RectilinearGrid(CPU();
                       size = (4, 4, 4),
                       x = (0, 100),
                       y = (0, 100),
                       z = z_faces,
                       halo = (3, 3, 3),
                       topology = (Periodic, Periodic, Bounded))

# Prescribed velocity fields (zero divergence)
u_prescribed(x, y, z, t) = 0.0
v_prescribed(x, y, z, t) = 0.0

velocities = PrescribedVelocityFields(; u=u_prescribed, v=v_prescribed)

# Prescribed free surface: η oscillates over one period
A = 0.1  # amplitude (small relative to H=10)
η_prescribed(x, y, z, t) = A * sin(2π * t)

free_surface = PrescribedFreeSurface(displacement=η_prescribed)

# Build model — this previously errored
@info "Building model..."
model = HydrostaticFreeSurfaceModel(grid;
                                     velocities,
                                     free_surface,
                                     tracers = nothing,
                                     buoyancy = nothing)

@info "Model construction succeeded!"
@info "Free surface type: $(typeof(model.free_surface))"
@info "Displacement type: $(typeof(model.free_surface.displacement))"

# Check that η appears in fields(model)
model_fields = fields(model)
@assert haskey(model_fields, :η) "η should appear in fields(model)"
@info "η found in fields(model) ✓"

# Time step for one full period with dt = 0.01
dt = 0.01
T_period = 1.0  # one full period of sin(2πt)
simulation = Simulation(model; Δt=dt, stop_time=T_period)

# σᶜᶜⁿ is an OffsetArray with halo; extract interior values using OffsetArray indexing
Nx, Ny, _ = size(grid)
σ_interior(σ) = [σ[i, j, 1] for i in 1:Nx, j in 1:Ny]

# Check initial σ ≈ 1.0 (η(0) = 0)
σ_initial = σ_interior(grid.z.σᶜᶜⁿ)
@info "Initial σᶜᶜⁿ range: [$(minimum(σ_initial)), $(maximum(σ_initial))]"
@assert all(σ_initial .≈ 1.0) "Initial σ should be 1.0 (η=0 at t=0)"

# Run for one full period
run!(simulation)

# Run for one full period
run!(simulation)

# σ should reflect η at the current clock time (step_free_surface! advances
# the PFS clock to tⁿ⁺¹ before the grid update, matching prognostic behavior).
σ_final = σ_interior(grid.z.σᶜᶜⁿ)
expected_σ = (H + A * sin(2π * model.clock.time)) / H

@info "Final σᶜᶜⁿ range at t=$(model.clock.time): [$(minimum(σ_final)), $(maximum(σ_final))]"
@info "Expected σ = $expected_σ"
@info "Actual σ[1,1,1] = $(σ_final[1,1,1])"

tol = 1e-10
@assert all(abs.(σ_final .- expected_σ) .< tol) "σ should match (H + η(t))/H"
@info "σ correctness test passed at t=$(model.clock.time) ✓"

# Also verify that σ oscillates correctly at a non-trivial time
simulation.stop_time = 1.25
run!(simulation)

σ_quarter = σ_interior(grid.z.σᶜᶜⁿ)
expected_σ_quarter = (H + A * sin(2π * model.clock.time)) / H

@info "σ at t=$(model.clock.time): $(σ_quarter[1,1,1])"
@info "Expected: $expected_σ_quarter"
@assert all(abs.(σ_quarter .- expected_σ_quarter) .< tol) "σ at t=1.25 should match"
@info "Mid-period σ test passed ✓"

@info "All tests passed!"
