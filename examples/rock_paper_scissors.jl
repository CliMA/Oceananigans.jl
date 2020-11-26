# # Rock paper scissors ecological game

using Oceananigans

# # Grid and domain

using Oceananigans.Utils: meters

Nx = Ny = 64
Nz = 32

Lx = Ly = 100meters
Lz = 50meters

topology = (Periodic, Periodic, Bounded)
grid = RegularCartesianGrid(topology=topology, size=(Nx, Ny, Nz),
                            x=(0, Lx), y=(0, Ly), z=(-Lz, 0))

# # Idealized diurnal cycle

using Oceananigans.Utils: hours

N² = 1e-5

Qb(x, y, t, p) = p.Q₀ + p.QΔ * sin(2π * t / p.τ)

b_top_bc = FluxBoundaryCondition(Qb, parameters=(Q₀=2e-8, QΔ=5e-8, τ=24hours))
b_bot_bc = GradientBoundaryCondition(N²)

b_bcs  = TracerBoundaryConditions(grid, top=b_top_bc, bottom=b_bot_bc)

# # Sponge layer

# # Lagrangian microbes

using StructArrays

@enum Species Rock Paper Scissors
display(Species)

struct Microbe{T,S}
          x :: T
          y :: T
          z :: T
    species :: S
end

n_microbes = 100

x₀ = Lx * rand(n_microbes)
y₀ = Ly * rand(n_microbes)
z₀ = -Lz/10 * rand(n_microbes)
s₀ = rand(instances(Species), n_microbes)

microbes = StructArray{Microbe}((x₀, y₀, z₀, s₀))

particles = LagrangianParticles(microbes, 0.5)

# # Model setup

using Oceananigans.Advection

model = IncompressibleModel(
           architecture = CPU(),
                   grid = grid,
            timestepper = :RungeKutta3,
              advection = UpwindBiasedFifthOrder(),
                tracers = :b,
               buoyancy = BuoyancyTracer(),
               coriolis = FPlane(f=-1e-4),
                closure = AnisotropicMinimumDissipation(),
    boundary_conditions = (b=b_bcs,),
              particles = particles
)

# # Setting initial conditions

b₀(x, y, z) = N²*z + 1e-6 * randn()
set!(model, b=b₀)

# # Simulation setup

using Printf
using Oceananigans.Utils: prettytime, second, seconds, hours

function print_progress(simulation)
    model = simulation.model

    @info @sprintf("iteration: %04d, time: %s, Δt: %s",
                   model.clock.iteration,
                   prettytime(model.clock.time),
                   prettytime(wizard.Δt))

    return nothing
end

wizard = TimeStepWizard(cfl=0.7, Δt=1second, min_Δt=0.2seconds, max_Δt=90seconds)

simulation = Simulation(model, Δt=wizard, iteration_interval=10, stop_time=12hours, progress=print_progress)

# # Output writing

using Oceananigans: fields

using Oceananigans.OutputWriters

using Oceananigans.Utils: minutes

simulation.output_writers[:fields] = NetCDFOutputWriter(model, fields(model), filepath="idealized_diurnal_cycle.nc",
                                                        schedule=TimeInterval(5minutes))

simulation.output_writers[:particles] = NetCDFOutputWriter(model, model.particles, filepath="lagrangian_microbes.nc",
                                                           schedule=TimeInterval(5minutes))

run!(simulation)

# # Visualizing the solution

using Plots, GeoData, NCDatasets

using GeoData: GeoXDim, GeoYDim, GeoZDim

@dim xC GeoXDim "x"
@dim xF GeoXDim "x"
@dim yC GeoYDim "y"
@dim yF GeoYDim "y"
@dim zC GeoZDim "z"
@dim zF GeoZDim "z"

stack = NCDstack("idealized_diurnal_cycle.nc")

w, b = stack[:w], stack[:b]
times = dims(b)[4]
Nt = length(times)

anim = @animate for n in 1:Nt
    @info "Plotting idealized diurnal cycle frame $n/$Nt..."

    w_plot = plot(w[Ti=n, xC=32], color=:balance, clims=(-0.1, 0.1), aspect_ratio=:auto,
                  title="Idealized diurnal cycle: $(prettytime(times[Nt]))")

    b_plot = plot(b[Ti=n, xC=32], color=:thermal, aspect_ratio=:auto, title="")

    plot(w_plot, b_plot, layout=(2, 1), size=(1600, 900))
end

mp4(anim, "idealized_diurnal_cycle.mp4", fps=15)

# # Visualizing the particle trajectories

x, y, z = stack_particles[:x], stack_particles[:y], stack_particles[:z]