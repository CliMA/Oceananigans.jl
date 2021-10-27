using Printf
using Statistics
using GLMakie
using Random
using JLD2

using Oceananigans
using Oceananigans.Units
using Oceananigans: fields
using Oceananigans.TurbulenceClosures: VerticallyImplicitTimeDiscretization

#=
# Domain
Lx = 500kilometers  # east-west extent [m]
Ly = 1000kilometers # north-south extent [m]
Lz = 1kilometers    # depth [m]

Nx = 1
Ny = 128
Nz = 32

architecture = CPU()

movie_interval = 0.2day
stop_time = 40days
Δt₀ = 10minutes

grid = RegularRectilinearGrid(topology = (Periodic, Bounded, Bounded), 
                              size = (Nx, Ny, Nz), 
                              x = (0, Lx),
                              y = (-Ly/2, Ly/2),
                              z = (-Lz, 0),
                              halo = (3, 3, 3))

coriolis = BetaPlane(latitude=-45)

Δx, Δy, Δz = Lx/Nx, Ly/Ny, Lz/Nz

𝒜 = Δz/Δx # Grid cell aspect ratio.

κh = 0.1    # [m² s⁻¹] horizontal diffusivity
νh = 0.1    # [m² s⁻¹] horizontal viscosity
κz = 𝒜 * κh # [m² s⁻¹] vertical diffusivity
νz = 𝒜 * νh # [m² s⁻¹] vertical viscosity

diffusive_closure = AnisotropicDiffusivity(νh = νh,
                                           νz = νz,
                                           κh = κh,
                                           κz = κz,
					                       time_discretization = VerticallyImplicitTimeDiscretization())

convective_adjustment = ConvectiveAdjustmentVerticalDiffusivity(convective_κz = 1.0,
                                                                convective_νz = 0.0)

gerdes_koberle_willebrand_tapering = Oceananigans.TurbulenceClosures.FluxTapering(1e-2)

gent_mcwilliams_diffusivity = IsopycnalSkewSymmetricDiffusivity(κ_skew = 1000,
                                                                κ_symmetric = (b=0, c=1000),
                                                                slope_limiter = gerdes_koberle_willebrand_tapering)
#####
##### Model building
#####

@info "Building a model..."

closures = (diffusive_closure, convective_adjustment, gent_mcwilliams_diffusivity)

model = HydrostaticFreeSurfaceModel(architecture = architecture,
                                    grid = grid,
                                    coriolis = coriolis,
                                    buoyancy = BuoyancyTracer(),
                                    closure = closures,
                                    tracers = (:b, :c),
                                    momentum_advection = WENO5(),
                                    tracer_advection = WENO5(),
                                    free_surface = ImplicitFreeSurface())

@info "Built $model."

#####
##### Initial conditions
#####

"""
Linear ramp from 0 to 1 between -Δy/2 and +Δy/2.

For example:

y < y₀           => ramp = 0
y₀ < y < y₀ + Δy => ramp = y / Δy
y > y₀ + Δy      => ramp = 1
"""
ramp(y, Δy) = min(max(0, y/Δy + 1/2), 1)

# Parameters
N² = 4e-6 # [s⁻²] buoyancy frequency / stratification
M² = 8e-8 # [s⁻²] horizontal buoyancy gradient

Δy = 50kilometers
Δc = 2Δy
Δb = Δy * M²
ϵb = 1e-2 * Δb # noise amplitude

bᵢ(x, y, z) = N² * z + Δb * ramp(y, Δy)
cᵢ(x, y, z) = exp(-y^2 / 2Δc^2)

set!(model, b=bᵢ, c=cᵢ)

#####
##### Simulation building
#####

wall_clock = [time_ns()]

function print_progress(sim)
    @printf("[%05.2f%%] i: %d, t: %s, wall time: %s, max(u): (%6.8e, %6.8e, %6.8e) m/s, next Δt: %s\n",
            100 * (sim.model.clock.time / sim.stop_time),
            sim.model.clock.iteration,
            prettytime(sim.model.clock.time),
            prettytime(1e-9 * (time_ns() - wall_clock[1])),
            maximum(abs, sim.model.velocities.u),
            maximum(abs, sim.model.velocities.v),
            maximum(abs, sim.model.velocities.w),
            prettytime(sim.Δt.Δt))

    wall_clock[1] = time_ns()
    
    return nothing
end

wizard = TimeStepWizard(cfl=0.2, Δt=Δt₀, max_Δt=Δt₀)

simulation = Simulation(model, Δt=wizard, stop_time=stop_time, progress=print_progress, iteration_interval=100)

#####
##### Output
#####

Redi_diffusivity = IsopycnalSkewSymmetricDiffusivity(κ_skew = (b=0, c=0),
                                                     κ_symmetric = (b=1, c=0),
                                                     slope_limiter = gerdes_koberle_willebrand_tapering)

dependencies = (Redi_diffusivity,
                model.tracers.b,
                Val(1),
                model.clock,
                model.diffusivity_fields,
                model.tracers,
                model.buoyancy,
                model.velocities)

using Oceananigans.TurbulenceClosures: ∇_dot_qᶜ

∇_q_op = KernelFunctionOperation{Center, Center, Center}(∇_dot_qᶜ,
                                                         grid,
                                                         architecture = model.architecture,
                                                         computed_dependencies = dependencies)
# R(b) eg the Redi operator applied to buoyancy
Rb = ComputedField(∇_q_op)

outputs = merge(fields(model), (; Rb))

simulation.output_writers[:fields] = JLD2OutputWriter(model, outputs,
                                                      schedule = TimeInterval(movie_interval),
                                                      field_slicer = nothing,
                                                      prefix = "zonally_averaged_baroclinic_adj_fields",
                                                      force = true)

@info "Running the simulation..."

run!(simulation, pickup=false)

@info "Simulation completed in " * prettytime(simulation.run_time)
=#

#####
##### Visualize
#####

fig = Figure(resolution = (1400, 700))

filepath = "zonally_averaged_baroclinic_adj_fields.jld2"

ut = FieldTimeSeries(filepath, "u")
bt = FieldTimeSeries(filepath, "b")
ct = FieldTimeSeries(filepath, "c")
rt = FieldTimeSeries(filepath, "Rb")

# Build coordinates, rescaling the vertical coordinate
x, y, z = nodes((Center, Center, Center), grid)

zscale = 100
z = z .* zscale

#####
##### Plot buoyancy...
#####

times = bt.times
Nt = length(times)

un(n) = interior(ut[n])[1, :, :]
bn(n) = interior(bt[n])[1, :, :]
cn(n) = interior(ct[n])[1, :, :]
rn(n) = interior(rt[n])[1, :, :]

@show min_c = 0
@show max_c = 1
@show max_u = maximum(abs, un(Nt))
min_u = - max_u

@show max_r = maximum(abs, rn(Nt))
@show min_r = - max_r

n = Node(1)
u = @lift un($n)
b = @lift bn($n)
c = @lift cn($n)
r = @lift rn($n)

ax = Axis(fig[1, 1], title="Zonal velocity")
hm = heatmap!(ax, y, z, u, colorrange=(min_u, max_u), colormap=:balance)
contour!(ax, y, z, b, levels = 25, color=:black, linewidth=2)
cb = Colorbar(fig[1, 2], hm)

ax = Axis(fig[2, 1], title="Tracer concentration")
hm = heatmap!(ax, y, z, c, colorrange=(min_c, max_c), colormap=:thermal)
contour!(ax, y, z, b, levels = 25, color=:black, linewidth=2)
cb = Colorbar(fig[2, 2], hm)

ax = Axis(fig[3, 1], title="R(b)")
hm = heatmap!(ax, y, z, r, colorrange=(min_r, max_r), colormap=:balance)
contour!(ax, y, z, b, levels = 25, color=:black, linewidth=2)
cb = Colorbar(fig[3, 2], hm)

title_str = @lift "Parameterized baroclinic adjustment at t = " * prettytime(times[$n])
ax_t = fig[0, :] = Label(fig, title_str)

display(fig)

record(fig, "zonally_averaged_baroclinic_adj.mp4", 1:Nt, framerate=8) do i
    @info "Plotting frame $i of $Nt"
    n[] = i
end