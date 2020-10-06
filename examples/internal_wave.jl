# # Internal wave example
#
# In this example, we initialize an internal wave packet in two-dimensions
# and watch it propagate. This example illustrates how to set up a two-dimensional
# model, set initial conditions, and how to use `BackgroundField`s.

# ## Numerical, domain, and internal wave parameters
#
# First, we pick a resolution and domain size. We use a two-dimensional domain
# that's periodic in $(x, y, z)$:

using Oceananigans

grid = RegularCartesianGrid(size=(128, 1, 128), x=(-π, π), y=(-π, π), z=(-π, π),
                            topology=(Periodic, Periodic, Periodic))

# Inertia-gravity waves propagate in fluids that are both _(i)_ rotating, and
# _(ii)_ density-stratified. We use Oceananigans' coriolis abstraction
# to implement the background rotation rate:

coriolis = FPlane(f=0.2)

# On an `FPlane`, the domain is idealized as rotating at a constant rate with
# rotation period `2π/f`. `coriolis` is passed to `IncompressibleModel` below.
# Our units are arbitrary.

# We use Oceananigans' `background_fields` abstraction to define a background
# buoyancy field `background_b(z) = N^2 * z`, where `z` is the vertical coordinate
# and `N` is the "buoyancy frequency". This means that the modeled buoyany field
# in Oceananigans will be a perturbation away from the basic state `background_b`.
# We choose

N = 1 ## buoyancy frequency

# and then construct the background buoyancy,

using Oceananigans.Fields: BackgroundField

## Background fields are functions of `x, y, z, t`, and optional parameters.
## Here we have one parameter, the buoyancy frequency
B_func(x, y, z, t, N) = N^2 * z

B = BackgroundField(B_func, parameters=N)

# We are now ready to instantiate our model. We pass `grid`, `coriolis`,
# and `background_b` to the `IncompressibleModel` constructor. In addition,
# we add a small amount of `IsotropicDiffusivity` to keep the model stable,
# during time-stepping, and specify our model to use a single tracer called
# `b` that we identify as buoyancy by setting `buoyancy=BuoyancyTracer()`.
  
model = IncompressibleModel(
                 grid = grid, 
          timestepper = :RungeKutta3,
              closure = IsotropicDiffusivity(ν=1e-6, κ=1e-6),
             coriolis = coriolis,
              tracers = :b,
    background_fields = (b=B,), # `background_fields` is a `NamedTuple`
             buoyancy = BuoyancyTracer()
)

# ## A Gaussian wavepacket
#
# Next, we set up an initial condition that excites an internal wave that propates
# through our rotating, stratified fluid. This internal wave has the pressure field
#
# $ p(x, y, z, t) = a(x, z) \, \cos(kx + mz - ω t) $.
#
# where $m$ is the vertical wavenumber, $k$ is the horizontal wavenumber,
# $ω$ is the wave frequncy, and $a(x, z)$ is a Gaussian envelope.
# The internal wave dispersion relation links the wave numbers $k$ and $m$,
# the Coriolis parameter $f$, and the buoyancy frequency N:

## Non-dimensional internal wave parameters
m = 16      # vertical wavenumber
k = 8       # horizontal wavenumber
f = coriolis.f

## Dispersion relation for inertia-gravity waves
ω² = (N^2 * k^2 + f^2 * m^2) / (k^2 + m^2)

ω = sqrt(ω²)
nothing # hide

# We define a Gaussian envelope for the wave packet so that we can 
# observe wave propagation.

## Some Gaussian parameters
A = 1e-9
δ = grid.Lx / 15

## A Gaussian envelope centered at $(x, z) = (0, 0)$.
a(x, z) = A * exp( -( x^2 + z^2 ) / 2δ^2 )
nothing # hide

# An inertia-gravity wave is a linear solution to the Boussinesq equations.
# In order that our initial condition excites an inertia-gravity wave, we
# initialize the velocity and buoyancy perturbation fields to be consistent
# with the pressure field $p = a \, \cos(kx + mx - ωt)$ at $t=0$.
# These relations are sometimes called the "polarization
# relations". At $t=0$, the polarization relations yield

u₀(x, y, z) = a(x, z) * k * ω   / (ω^2 - f^2) * cos(k*x + m*z)
v₀(x, y, z) = a(x, z) * k * f   / (ω^2 - f^2) * sin(k*x + m*z)
w₀(x, y, z) = a(x, z) * m * ω   / (ω^2 - N^2) * cos(k*x + m*z)
b₀(x, y, z) = a(x, z) * m * N^2 / (ω^2 - N^2) * sin(k*x + m*z)

set!(model, u=u₀, v=v₀, w=w₀, b=b₀)

# Recall that the buoyancy `b` is a perturbation, so that the total buoyancy field
# is $N^2 z + b$.

# ## A wave packet on the loose
#
# We're ready to release the packet. We build a simulation with a constant time-step,

simulation = Simulation(model, Δt = 0.02 * 2π/ω, stop_iteration = 100)
                        
# and add an output writer that saves the vertical velocity field every two iterations:

using Oceananigans.OutputWriters: JLD2OutputWriter

simulation.output_writers[:velocities] = JLD2OutputWriter(model, model.velocities,
                                                          iteration_interval = 2,
                                                                      prefix = "internal_wave",
                                                                       force = true)

# With initial conditions set and an output writer at the ready, we run the simulation

run!(simulation)

# ## Animating a propagating packet
#
# To visualize the solution, we load snapshots of the data and use it to make contour
# plots of vertical velocity.

using JLD2, Plots, Printf, Oceananigans.Grids

# We use coordinate arrays appropriate for the vertical velocity field,

x, y, z = nodes(model.velocities.w)

# open the jld2 file with the data,

file = jldopen(simulation.output_writers[:velocities].filepath)

## Extracts a vector of `iterations` at which data was saved.
iterations = parse.(Int, keys(file["timeseries/t"]))

# and makes an animation with Plots.jl:

anim = @animate for (i, iter) in enumerate(iterations)

    w = file["timeseries/w/$iter"][:, 1, :]
    t = file["timeseries/t/$iter"]

    contourf(x, z, w', title = @sprintf("ωt = %.2f", ω * t),
                      levels = range(-1e-8, stop=1e-8, length=10),
                       clims = (-1e-8, 1e-8),
                      xlabel = "x",
                      ylabel = "z",
                       xlims = (-π, π),
                       ylims = (-π, π),
                   linewidth = 0,
                       color = :balance,
                      legend = false,
                 aspectratio = :equal)
end

mp4(anim, "internal_wave.mp4", fps = 8) # hide
