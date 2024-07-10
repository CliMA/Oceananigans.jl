# This validation script shows open boundaries working in a simple case where the
# oscillates sinusoidally so changes sign across two open boundaries. This is similar
# to a more realistic case where we know some arbitary external conditions. 
# This necessitates using a combination allowing information to exit the domain, in 
# this case by setting the wall normal velocity gradient to zero, as well as forcing
# to the external value in this example by relaxing to it.

# This case also has a stretched grid to validate the zero wall normal velocity 
# gradient matching scheme on a stretched grid.

using Oceananigans, Adapt, CairoMakie
using Oceananigans.BoundaryConditions: FlatExtrapolationOpenBoundaryCondition

import Adapt: adapt_structure

@kwdef struct Cylinder{FT}
    D :: FT = 1.0
   x₀ :: FT = 0.0
   y₀ :: FT = 0.0
end

@inline (cylinder::Cylinder)(x, y) = ifelse((x - cylinder.x₀)^2 + (y - cylinder.y₀)^2 < (cylinder.D/2)^2, 1, 0)

Adapt.adapt_structure(to, cylinder::Cylinder) = cylinder

architecture = CPU()

# model parameters
Re = 200
U = 1
D = 1.
resolution = D / 10

# add extra downstream distance to see if the solution near the cylinder changes
extra_downstream = 0

cylinder = Cylinder(; D)

x = (-5, 5 + extra_downstream) .* D

Ny = Int(10 / resolution)

Nx = Int((10 + extra_downstream) / resolution)

function Δy(j)
    if Ny/2 - 2/resolution < j < Ny/2 + 2/resolution
        return resolution
    elseif j <= Ny/2 - 2/resolution 
        return resolution * (1 + (Ny/2 - 2/resolution - j) / (Ny/2 - 2/resolution))
    elseif j >= Ny/2 + 2/resolution
        return resolution * (1 + (j - Ny/2 - 2/resolution) / (Ny/2 - 2/resolution))
    else
        Throw(ArgumentError("$j not in range"))
    end
end

y(j) = sum(Δy.([1:j;])) - sum(Δy.([1:Ny;]))/2

ν = U * D / Re

closure = ScalarDiffusivity(;ν, κ = ν)

grid = RectilinearGrid(architecture; topology = (Bounded, Bounded, Flat), size = (Nx, Ny), x = y, y = x)

T = 20 / U

@inline u(t, p)      = p.U * sin(t * 2π / p.T)
@inline u(y, t, p)   = u(t, p)

relaxation_timescale = 0.15

u_boundaries = FieldBoundaryConditions(east = FlatExtrapolationOpenBoundaryCondition(u; relaxation_timescale, parameters = (; U, T)),
                                       west = FlatExtrapolationOpenBoundaryCondition(u; relaxation_timescale, parameters = (; U, T)),
                                       south = GradientBoundaryCondition(0),
                                       north = GradientBoundaryCondition(0))

v_boundaries = FieldBoundaryConditions(east = GradientBoundaryCondition(0),
                                       west = GradientBoundaryCondition(0),
                                       south = FlatExtrapolationOpenBoundaryCondition(0; relaxation_timescale),
                                       north = FlatExtrapolationOpenBoundaryCondition(0; relaxation_timescale))

Δt = .3 * resolution / U

@show Δt

u_forcing = Relaxation(; rate = 1 / (2 * Δt), mask = cylinder)
v_forcing = Relaxation(; rate = 1 / (2 * Δt), mask = cylinder) 

model = NonhydrostaticModel(; grid, 
                              closure, 
                              forcing = (u = u_forcing, v = v_forcing),
                              boundary_conditions = (u = u_boundaries, v = v_boundaries))

@info "Constructed model"

# initial noise to induce turbulance faster
set!(model, u = (x, y) -> randn() * U * 0.01, v = (x, y) -> randn() * U * 0.01)

@info "Set initial conditions"

simulation = Simulation(model; Δt = Δt, stop_time = 300)

wizard = TimeStepWizard(cfl = 0.3, max_Δt = Δt)

simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(100))

progress(sim) = @info "$(time(sim)) with Δt = $(prettytime(sim.Δt)) in $(prettytime(sim.run_wall_time))"

simulation.callbacks[:progress] = Callback(progress, IterationInterval(1000))

simulation.output_writers[:velocity] = JLD2OutputWriter(model, model.velocities,
                                                        overwrite_existing = true, 
                                                        filename = "oscillating_cylinder_$(extra_downstream)_Re_$Re.jld2", 
                                                        schedule = TimeInterval(1),
                                                        with_halos = true)

run!(simulation)

# load the results 

u_ts = FieldTimeSeries("oscillating_cylinder_$(extra_downstream)_Re_$Re.jld2", "u")
v_ts = FieldTimeSeries("oscillating_cylinder_$(extra_downstream)_Re_$Re.jld2", "v")

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

xc, yc, zc = nodes(ζ)

ax = Axis(fig[1, 1], aspect = DataAspect(), limits = (minimum(xc), maximum(xc), minimum(yc), maximum(yc)))

n = Observable(1)

ζ_plt = @lift ζ_ts[:, :, $n]

contour!(ax, xc, yc, ζ_plt, levels = [-1, 1], colorrange = (-1, 1), colormap = :roma)

record(fig, "oscillating_ζ_Re_$(Re)_no_exterior.mp4", 1:length(u_ts.times), framerate = 5) do i;
    n[] = i
    i % 10 == 0 && @info "$(n.val) of $(length(u_ts.times))"
end