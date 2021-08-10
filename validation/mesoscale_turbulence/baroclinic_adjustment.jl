using Printf
using Statistics
using Plots

using Oceananigans
using Oceananigans.Units
using Oceananigans.OutputReaders: FieldTimeSeries
using Oceananigans.Grids: xnode, ynode, znode

# nobs
const stretched_grid = false
const hydrostatic = false

# domain
const Lx = 250kilometers # east-west extent [m]
const Ly = 500kilometers # north-south extent [m]
const Lz = 1kilometers    # depth [m]

Nx = 64
Ny = 128
Nz = 8

s = 1.2 # stretching factor
z_faces(k) = - Lz * (1 - tanh(s * (k - 1) / Nz) / tanh(s))

arch = CPU()

if stretched_grid
    println("using a stretched grid")
    grid = VerticallyStretchedRectilinearGrid(architecture = arch,
                                            topology = (Periodic, Bounded, Bounded),
                                            size = (Nx, Ny, Nz),
                                            halo = (3, 3, 3),
                                            x = (0, Lx),
                                            y = (0, Ly),
                                            z_faces = z_faces)
else
    println("using a regular grid")
    grid = RegularRectilinearGrid(topology=(Periodic, Bounded, Bounded), 
                            size=(Nx, Ny, Nz), 
                            x=(0, Lx), y=(0, Ly), z=(-Lz, 0))

end

@info "Built a grid: $grid."

#####
##### Coriolis
#####

coriolis = BetaPlane(latitude=-45)

#####
##### Dissipation
#####

#horizontal_diffusivity = AnisotropicBiharmonicDiffusivity(νh=κ₄h, κh=κ₄h)

@show κ₂h = 1 #1e0 / day * grid.Δx^2 # [m² s⁻¹] horizontal viscosity and diffusivity
@show ν₂h = 30 #1e0 / day * grid.Δx^2 # [m² s⁻¹] horizontal viscosity and diffusivity
@show κ₄h = 1e-1 / day * grid.Δx^4 # [m⁴ s⁻¹] horizontal hyperviscosity and hyperdiffusivity

using Oceananigans.TurbulenceClosures: HorizontallyCurvilinearAnisotropicBiharmonicDiffusivity

horizontal_diffusivity = HorizontallyCurvilinearAnisotropicDiffusivity(νh=ν₂h, κh=κ₂h)
#horizontal_diffusivity = HorizontallyCurvilinearAnisotropicBiharmonicDiffusivity(νh=κ₄h, κh=κ₄h)

convective_adjustment = ConvectiveAdjustmentVerticalDiffusivity(convective_κz = 1.0,
                                                                convective_νz = 0.0,
                                                                background_κz = 5e-6,
                                                                background_νz = 3e-4)
#####
##### Model building
#####

@info "Building a model..."

if hydrostatic
    println("constructing hydrostatic model")
    model = HydrostaticFreeSurfaceModel(architecture = arch,
                                        grid = grid,
                                        coriolis = coriolis,
                                        buoyancy = BuoyancyTracer(),
                                        closure = (horizontal_diffusivity, convective_adjustment),
                                        tracers = :b,
                                        momentum_advection = WENO5(),
                                        tracer_advection = WENO5(),
                                        free_surface = ImplicitFreeSurface(),
                                        )
else
    println("constructing nonhydrostatic model")
    model = NonhydrostaticModel(
           architecture = arch,
                   grid = grid,
               coriolis = coriolis,
               buoyancy = BuoyancyTracer(),
                closure = closure,
                tracers = (:b,),
              advection = WENO5(),
)
end

@info "Built $model."

#####
##### Initial conditions
#####

const Ty = 4e-5  # Meridional temperature gradient [K/m].
const Tz = 2e-3  # Vertical temperature gradient [K/m].

# Initial temperature field [°C].
T₀(x, y, z) = 10 + Ty*min(max(0, y-225e3), 50e3) + Tz*z + 0.0001*rand()
B₀(x, y, z) = 2e-3 * T₀(x, y, z)

set!(model, b=B₀)

#####
##### Simulation building
#####

wizard = TimeStepWizard(cfl=0.1, Δt=1minute, max_change=1.1, max_Δt=20minutes)

wall_clock = [time_ns()]

function print_progress(sim)
    @printf("[%05.2f%%] i: %d, t: %s, wall time: %s, max(u): (%6.3e, %6.3e, %6.3e) m/s, next Δt: %s\n",
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

simulation = Simulation(model, Δt=wizard, stop_time=60days, progress=print_progress, iteration_interval=10)


@info "Running the simulation..."

try
    run!(simulation, pickup=false)
catch err
    @info "run! threw an error! The error message is"
    showerror(stdout, err)
end