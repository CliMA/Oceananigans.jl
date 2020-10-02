# # Internal wave example
#
# In this example, we initialize an internal wave packet in two-dimensions
# and watch it propagate. This example illustrates how to set up a two-dimensional
# model, set initial conditions, and how to use `BackgroundField`s.

using Oceananigans, Oceananigans.Grids, Plots, Printf

# ## Numerical, domain, and internal wave parameters
#
# First, we pick a resolution and domain size. We use a two-dimensional domain
# that's periodic in $(x, z)$:

grid = RegularCartesianGrid(size=(128, 128), x=(-π, π), z=(-π, π),
                            topology=(Periodic, Flat, Periodic))

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

## Background fields are functions of `x, y, z, t`, and optional parameters
background_b_func(x, y, z, t, N²) = N² * z

background_b = BackgroundField(background_b_func, parameters=N^2)

# We are now ready to instantiate our model. We pass `grid`, `coriolis`,
# and `background_b` to the `IncompressibleModel` constructor. In addition,
# we add a small amount of `IsotropicDiffusivity` to keep the model stable,
# during time-stepping, and specify our model to use a single tracer called
# `b` that we identify as buoyancy by setting `buoyancy=BuoyancyTracer()`.
  
model = IncompressibleModel(
                 grid = grid, 
              closure = IsotropicDiffusivity(ν=1e-6, κ=1e-6),
             coriolis = coriolis,
              tracers = :b,
    background_fields = (b=background_b,), # `background_fields` is a `NamedTuple`
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

# The internal wave polarization relations follow from the linearized
# Boussinesq equations. They determine the amplitude of `u`, `v`, `w`,
# and the buoyancy perturbation `b`.

## Solution amplitudes
U = k * ω   / (ω^2 - f^2)
V = k * f   / (ω^2 - f^2)
W = m * ω   / (ω^2 - N^2)
B = m * N^2 / (ω^2 - N^2)

## Polarization relations
u₀(x, y, z) = a(x, z) * U * cos(k*x + m*z)
v₀(x, y, z) = a(x, z) * V * sin(k*x + m*z)
w₀(x, y, z) = a(x, z) * W * cos(k*x + m*z)
b₀(x, y, z) = a(x, z) * B * sin(k*x + m*z)
nothing # hide

# We initialize the velocity and buoyancy fields
# with our internal wave initial condition.

set!(model, u=u₀, v=v₀, w=w₀, b=b₀)

# ## A wave packet on the loose
#
# Finally, we release the packet and watch it go!

simulation = Simulation(model, Δt = 0.001 * 2π/ω,
                        #stop_iteration = 2000, iteration_interval = 20)
                        stop_iteration = 0, iteration_interval = 20)

# output...
run!(simulation)

anim = @animate for i=0:100
    x, z = xnodes(Cell, model.grid)[:], znodes(Face, model.grid)[:]
    w = model.velocities.w

    contourf(x, z, w.data[1:Nx, 1, 1:Nx+1]',
                   title = @sprintf("ωt = %.2f", ω * model.clock.time),
                  levels = range(-1e-8, stop=1e-8, length=10),
                   clims = (-1e-8, 1e-8),
                  xlabel = "x",
                  ylabel = "z",
                   xlims = (0, Lx),
                   ylims = (-Lx, 0),
               linewidth = 0,
                   color = :balance,
                  legend = false,
             aspectratio = :equal)

    simulation.stop_iteration += 20
    run!(simulation)
end

mp4(anim, "internal_wave.mp4", fps = 15) # hide
