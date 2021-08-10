using Printf
using Statistics
using Oceananigans
using Oceananigans.Units
using Oceananigans.OutputReaders: FieldTimeSeries
using Oceananigans.Grids: xnode, ynode, znode

# nobs
const stretched_grid = false
const hydrostatic = true

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

#####
##### Coriolis
#####

coriolis = BetaPlane(latitude=-45)

#####
##### Closures
#####

Δx, Δy, Δz = Lx/Nx, Ly/Ny, Lz/Nz

𝒜 = Δz/Δx # Grid cell aspect ratio.

κh = 0.25   # [m²/s] horizontal diffusivity
νh = 0.25   # [m²/s] horizontal viscocity
κv = 𝒜 * κh # [m²/s] vertical diffusivity
νv = 𝒜 * νh # [m²/s] vertical viscocity

diffusive_closure = AnisotropicDiffusivity(νx = νh, νy = νh, νz =νv, 
                                 κx = κh, κy = κh, κz=κv)

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
                                        closure = (diffusive_closure, convective_adjustment),
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
                closure = diffusive_closure,
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

wizard = TimeStepWizard(cfl=0.15, Δt=30.0, max_change=1.5, max_Δt=300.0)

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

simulation = Simulation(model, Δt=wizard, stop_time=30days, progress=print_progress, iteration_interval=10)


@info "Running the simulation..."

try
    run!(simulation, pickup=false)
catch err
    @info "run! threw an error! The error message is"
    showerror(stdout, err)
end


## plotting
using GLMakie

# plotting 
xsurf = range(0, Lx,  length = Nx)
ysurf = range(0, Ly,  length = Ny)
zsurf = range(-Lz, 0, length = Nz)
ϕsurf = interior(simulation.model.tracers.b)
clims = extrema(ϕsurf)
zscale = 100
fig = Figure(resolution = (1920, 1080))
ax = fig[1,1] = LScene(fig, title= "Baroclinic Adjustment")

# edge 1
ϕedge1 = ϕsurf[:,1,:]
GLMakie.surface!(ax, xsurf, zsurf .* zscale, ϕedge1, transformation = (:xz, 0),  colorrange = clims, colormap = :balance, show_axis=false)

# edge 2
ϕedge2 = ϕsurf[:,end,:]
GLMakie.surface!(ax, xsurf, zsurf .* zscale, ϕedge2, transformation = (:xz, Ly),  colorrange = clims, colormap = :balance)


# edge 3
ϕedge3 = ϕsurf[1,:,:]
GLMakie.surface!(ax, ysurf, zsurf .* zscale, ϕedge3, transformation = (:yz, 0),  colorrange = clims, colormap = :balance)

# edge 4
ϕedge4 = ϕsurf[end,:,:]
GLMakie.surface!(ax, ysurf, zsurf .* zscale, ϕedge4, transformation = (:yz, Lx),  colorrange = clims, colormap = :balance)

# edge 5
ϕedge5 = ϕsurf[:,:,1]
GLMakie.surface!(ax, xsurf, ysurf, ϕedge5, transformation = (:xy, -Lz *  zscale), colorrange = clims, colormap = :balance)


# edge 6
ϕedge6 = ϕsurf[:,:,end]
GLMakie.surface!(ax, xsurf, ysurf, ϕedge6, transformation = (:xy, 0 *  zscale), colorrange = clims, colormap = :balance)

display(fig)