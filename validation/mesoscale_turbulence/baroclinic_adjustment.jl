using Printf
using Statistics
using Oceananigans
using Oceananigans.Units
using Oceananigans.TurbulenceClosures: VerticallyImplicitTimeDiscretization

# Domain
Lx = 250kilometers # east-west extent [m]
Ly = 500kilometers # north-south extent [m]
Lz = 1kilometers    # depth [m]

Nx = 64
Ny = 128
Nz = 32

grid = RegularRectilinearGrid(topology = (Periodic, Bounded, Bounded), 
                              size = (Nx, Ny, Nz), 
                              x = (0, Lx),
                              y = (0, Ly),
                              z = (-Lz, 0),
                              halo = (3, 3, 3))

coriolis = BetaPlane(latitude=-45)

Δx, Δy, Δz = Lx/Nx, Ly/Ny, Lz/Nz

𝒜 = Δz/Δx # Grid cell aspect ratio.

κh = 0.25   # [m²/s] horizontal diffusivity
νh = 0.25   # [m²/s] horizontal viscocity
κv = 𝒜 * κh # [m²/s] vertical diffusivity
νv = 𝒜 * νh # [m²/s] vertical viscocity

diffusive_closure = AnisotropicDiffusivity(νx=νh, νy=νh, νz=νv, κx=κh, κy=κh, κz=κv,
					                       time_discretization = VerticallyImplicitTimeDiscretization())

convective_adjustment = ConvectiveAdjustmentVerticalDiffusivity(convective_κz = 1.0,
                                                                convective_νz = 0.0)

#####
##### Model building
#####

@info "Building a model..."

closures = (diffusive_closure, convective_adjustment)

model = HydrostaticFreeSurfaceModel(architecture = GPU(),
                                    grid = grid,
                                    coriolis = coriolis,
                                    buoyancy = BuoyancyTracer(),
                                    closure = closures,
                                    tracers = :b,
                                    momentum_advection = WENO5(),
                                    tracer_advection = WENO5(),
                                    free_surface = ImplicitFreeSurface())

@info "Built $model."

#####
##### Initial conditions
#####

"""
Linear ramp from 0 to 1 between y₀ and y₀ + Δy.

For example:

y < y₀           => ramp = 0
y₀ < y < y₀ + Δy => ramp = y / Δy
y > y₀ + Δy      => ramp = 1
"""
ramp(y, y₀, Δy) = min(max(0, (y - y₀) / Δy), 1)

# Parameters
N² = 4e-6   # [s⁻²] buoyancy frequency / stratification
M² = 1.2e-8 # [s⁻²] horizontal buoyancy gradient

y₀ = 200kilometers
Δy = Ly - y₀
Δb = Δy * M²
ϵb = 1e-2 * Δb

bᵢ(x, y, z) = N² * z + Δb * ramp(y, y₀, Δy) * ϵb * randn()

set!(model, b=bᵢ)

#####
##### Simulation building
#####

wall_clock = [time_ns()]

get_Δt(wizard::TimeStepWizard) = prettytime(wizard.Δt)
get_Δt(Δt) = prettytime(Δt)

function print_progress(sim)
    @printf("[%05.2f%%] i: %d, t: %s, wall time: %s, max(u): (%6.8e, %6.8e, %6.8e) m/s, next Δt: %s\n",
            100 * (sim.model.clock.time / sim.stop_time),
            sim.model.clock.iteration,
            prettytime(sim.model.clock.time),
            prettytime(1e-9 * (time_ns() - wall_clock[1])),
            maximum(abs, sim.model.velocities.u),
            maximum(abs, sim.model.velocities.v),
            maximum(abs, sim.model.velocities.w),
            get_Δt(sim.Δt))

    wall_clock[1] = time_ns()
    
    return nothing
end

wizard = TimeStepWizard(cfl=0.2, Δt=5minutes, max_Δt=10minutes)

simulation = Simulation(model, Δt=wizard, stop_time=10days, progress=print_progress, iteration_interval=100)

@info "Running the simulation..."

try
    run!(simulation, pickup=false)
catch err
    @info "run! threw an error! The error message is"
    showerror(stdout, err)
end

@info "Simulation completed in " * prettytime(sim.run_time)

# Plotting
using GLMakie

xsurf = range(0, Lx,  length = Nx)
ysurf = range(0, Ly,  length = Ny)
zsurf = range(-Lz, 0, length = Nz)
ϕsurf = Array(interior(simulation.model.tracers.b))
clims = extrema(ϕsurf)
zscale = 100
fig = Figure(resolution = (1920, 1080))
ax = fig[1, 1] = LScene(fig, title= "Baroclinic Adjustment")

# edge 1
ϕedge1 = ϕsurf[:, 1, :]
GLMakie.surface!(ax, xsurf, zsurf .* zscale, ϕedge1,
                 transformation = (:xz, 0), colorrange = clims, colormap = :balance, show_axis=false)

# edge 2
ϕedge2 = ϕsurf[:, end, :]
GLMakie.surface!(ax, xsurf, zsurf .* zscale, ϕedge2,
                 transformation = (:xz, Ly),  colorrange = clims, colormap = :balance)

# edge 3
ϕedge3 = ϕsurf[1, :, :]
GLMakie.surface!(ax, ysurf, zsurf .* zscale, ϕedge3,
                 transformation = (:yz, 0),  colorrange = clims, colormap = :balance)

# edge 4
ϕedge4 = ϕsurf[end, :, :]
GLMakie.surface!(ax, ysurf, zsurf .* zscale, ϕedge4,
                 transformation = (:yz, Lx),  colorrange = clims, colormap = :balance)

# edge 5
ϕedge5 = ϕsurf[:, :, 1]
GLMakie.surface!(ax, xsurf, ysurf, ϕedge5,
                 transformation =(:xy, -Lz * zscale), colorrange = clims, colormap = :balance)


# edge 6
ϕedge6 = ϕsurf[:, :, end]
GLMakie.surface!(ax, xsurf, ysurf, ϕedge6,
                 transformation = (:xy, 0 * zscale), colorrange = clims, colormap = :balance)

display(fig)

