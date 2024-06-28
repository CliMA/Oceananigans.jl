# This validation script shows open boundaries working in a simple case where the
# flow remains largely unidirectional and so at one end we have no matching scheme
# but just prescribe the inflow. At the other end we then make no assumptions about
# the flow and use a very simple open boundary condition to permit information to 
# exit the domain. If, for example, the flow at the prescribed boundary was reversed
# then the model would likely fail.

using Oceananigans, Adapt, CairoMakie
using Oceananigans.BoundaryConditions: FlatExtrapolationOpenBoundaryCondition

import Adapt: adapt_structure

@kwdef struct Cylinder{FT}
    D :: FT = 1.
   x₀ :: FT = 0.
   y₀ :: FT = 0.
end

@inline (cylinder::Cylinder)(x, y) = ifelse((x - cylinder.x₀)^2 + (y - cylinder.y₀)^2 < (cylinder.D/2)^2, 1, 0)

Adapt.adapt_structure(to, cylinder::Cylinder) = cylinder

architecture = GPU()

# model parameters
Re = 200
U = 1
D = 1.
resolution = D / 40

# add extra downstream distance to see if the solution near the cylinder changes
extra_downstream = 0

cylinder = Cylinder(; D)

x = (-5, 5 + extra_downstream) .* D
y = (-5, 5) .* D

Ny = Int(10 / resolution)
Nx = Ny + Int(extra_downstream / resolution)

ν = U * D / Re

closure = ScalarDiffusivity(;ν, κ = ν)

grid = RectilinearGrid(architecture; topology = (Bounded, Periodic, Flat), size = (Nx, Ny), x, y)

@inline u(y, t, U) = U * (1 + 0.01 * randn())

u_boundaries = FieldBoundaryConditions(east = FlatExtrapolationOpenBoundaryCondition(),
                                       west = OpenBoundaryCondition(u, parameters = U))

v_boundaries = FieldBoundaryConditions(east = GradientBoundaryCondition(0),
                                       west = GradientBoundaryCondition(0))

Δt = .3 * resolution / U

u_forcing = Relaxation(; rate = 1 / (2 * Δt), mask = cylinder)
v_forcing = Relaxation(; rate = 1 / (2 * Δt), mask = cylinder) 

model = NonhydrostaticModel(; grid, 
                              closure, 
                              forcing = (u = u_forcing, v = v_forcing),
                              boundary_conditions = (u = u_boundaries, v = v_boundaries))

@info "Constructed model"

# initial noise to induce turbulance faster
set!(model, u = U, v = (x, y) -> randn() * U * 0.01)

@info "Set initial conditions"

simulation = Simulation(model; Δt = Δt, stop_time = 300)

wizard = TimeStepWizard(cfl = 0.3)

simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(100))

progress(sim) = @info "$(time(sim)) with Δt = $(prettytime(sim.Δt)) in $(prettytime(sim.run_wall_time))"

simulation.callbacks[:progress] = Callback(progress, IterationInterval(1000))

simulation.output_writers[:velocity] = JLD2OutputWriter(model, model.velocities,
                                                        overwrite_existing = true, 
                                                        filename = "cylinder_$(extra_downstream)_Re_$Re.jld2", 
                                                        schedule = TimeInterval(1),
                                                        with_halos = true)

run!(simulation)

# load the results 

u_ts = FieldTimeSeries("cylinder_$(extra_downstream)_Re_$Re.jld2", "u")
v_ts = FieldTimeSeries("cylinder_$(extra_downstream)_Re_$Re.jld2", "v")

u′, v′, w′ = Oceananigans.Fields.VelocityFields(u_ts.grid)

ζ = Field((@at (Center, Center, Center) ∂x(v′)) - (@at (Center, Center, Center) ∂y(u′)))

# there is probably a more memory efficient way todo this

ζ_ts = zeros(size(grid, 1), size(grid, 2), length(u_ts.times)) # u_ts.grid so its always on cpu

for n in 1:length(u_ts.times)
    set!(u′, u_ts[n])
    set!(v′, v_ts[n])
    compute!(ζ)
    ζ_ts[:, :, n] = interior(ζ, :, :, 1)
end

@info "Loaded results"

# plot the results

fig = Figure(size = (600, 600))

ax = Axis(fig[1, 1], aspect = DataAspect())

xc, yc, zc = nodes(ζ)

n = Observable(1)

ζ_plt = @lift ζ_ts[:, :, $n]

contour!(ax, xc, yc, ζ_plt, levels = [-2, 2], colorrange = (-2, 2), colormap = :roma)

record(fig, "ζ_Re_$Re.mp4", 1:length(u_ts.times), framerate = 5) do i;
    n[] = i
    i % 10 == 0 && @info "$(n.val) of $(length(u_ts.times))"
end