# # Internal wave example
#
# In this example, we initialize an internal wave packet in two-dimensions
# and watch it propagate. This example illustrates how to set up a two-dimensional
# model, set initial conditions, and how to use `BackgroundField`s.
#
# ## Install dependencies
#
# First let's make sure we have all required packages installed.

# ```julia
# using Pkg
# pkg"add Oceananigans, CairoMakie"
# ```

# ## The physical domain
#
# First, we pick a resolution and domain size. We use a two-dimensional domain
# that's periodic in ``(x, z)`` and is `Flat` in ``y``:

using Oceananigans

grid = RectilinearGrid(size=(128, 128), x=(-π, π), z=(-π, π), topology=(Periodic, Flat, Periodic))

# ## Internal wave parameters
#
# Inertia-gravity waves propagate in fluids that are both _(i)_ rotating, and
# _(ii)_ density-stratified. We use Oceananigans' Coriolis abstraction
# to implement a background rotation rate:

coriolis = FPlane(f=0.2)

# On an `FPlane`, the domain is idealized as rotating at a constant rate with
# rotation period `2π/f`. `coriolis` is passed to `NonhydrostaticModel` below.
# Our units are arbitrary.

# We use Oceananigans' `background_fields` abstraction to define a background
# buoyancy field `B(z) = N^2 * z`, where `z` is the vertical coordinate
# and `N` is the "buoyancy frequency". This means that the modeled buoyancy field
# perturbs the basic state `B(z)`.

## Background fields are functions of `x, y, z, t`, and optional parameters.
## Here we have one parameter, the buoyancy frequency

N = 1 ## buoyancy frequency
B_func(x, y, z, t, N) = N^2 * z
B = BackgroundField(B_func, parameters=N)

# We are now ready to instantiate our model. We pass `grid`, `coriolis`,
# and `B` to the `NonhydrostaticModel` constructor.
# We add a small amount of `IsotropicDiffusivity` to keep the model stable
# during time-stepping, and specify that we're using a single tracer called
# `b` that we identify as buoyancy by setting `buoyancy=BuoyancyTracer()`.

model = NonhydrostaticModel(; grid, coriolis,
                            advection = CenteredFourthOrder(),
                            timestepper = :RungeKutta3,
                            closure = ScalarDiffusivity(ν=1e-6, κ=1e-6),
                            tracers = :b,
                            buoyancy = BuoyancyTracer(),
                            background_fields = (; b=B)) # `background_fields` is a `NamedTuple`

# ## A Gaussian wavepacket
#
# Next, we set up an initial condition that excites an internal wave that propates
# through our rotating, stratified fluid. This internal wave has the pressure field
#
# ```math
# p(x, y, z, t) = a(x, z) \, \cos(kx + mz - ω t) \, .
# ```
#
# where ``m`` is the vertical wavenumber, ``k`` is the horizontal wavenumber,
# ``ω`` is the wave frequncy, and ``a(x, z)`` is a Gaussian envelope.
# The internal wave dispersion relation links the wave numbers ``k`` and ``m``,
# the Coriolis parameter ``f``, and the buoyancy frequency ``N``:

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

## A Gaussian envelope centered at ``(x, z) = (0, 0)``.
a(x, z) = A * exp( -( x^2 + z^2 ) / 2δ^2 )
nothing # hide

# An inertia-gravity wave is a linear solution to the Boussinesq equations.
# In order that our initial condition excites an inertia-gravity wave, we
# initialize the velocity and buoyancy perturbation fields to be consistent
# with the pressure field ``p = a \, \cos(kx + mx - ωt)`` at ``t=0``.
# These relations are sometimes called the "polarization
# relations". At ``t=0``, the polarization relations yield

u₀(x, y, z) = a(x, z) * k * ω   / (ω^2 - f^2) * cos(k*x + m*z)
v₀(x, y, z) = a(x, z) * k * f   / (ω^2 - f^2) * sin(k*x + m*z)
w₀(x, y, z) = a(x, z) * m * ω   / (ω^2 - N^2) * cos(k*x + m*z)
b₀(x, y, z) = a(x, z) * m * N^2 / (ω^2 - N^2) * sin(k*x + m*z)

set!(model, u=u₀, v=v₀, w=w₀, b=b₀)

# Recall that the buoyancy `b` is a perturbation, so that the total buoyancy field
# is ``N^2 z + b``.

# ## A wave packet on the loose
#
# We're ready to release the packet. We build a simulation with a constant time-step,

simulation = Simulation(model, Δt = 0.1 * 2π/ω, stop_iteration = 20)

# and add an output writer that saves the vertical velocity field every two iterations:

simulation.output_writers[:velocities] = JLD2OutputWriter(model, model.velocities,
                                                          schedule = IterationInterval(1),
                                                          filename = "internal_wave.jld2",
                                                          overwrite_existing = true)

# With initial conditions set and an output writer at the ready, we run the simulation

run!(simulation)

# ## Animating a propagating packet
#
# To visualize the solution, we load a `FieldTimeSeries` of `w` and make contour
# plots of vertical velocity.

filename = "internal_wave"

w_timeseries = FieldTimeSeries(filename * ".jld2", "w")

# And build the the ``x, y, z`` grid for plotting purposes.

x, y, z = nodes(w_timeseries)

#-

using CairoMakie

fig = Figure(resolution = (600, 600))

ax = Axis(fig[2, 1];
          xlabel = "x",
          ylabel = "z",
          limits = ((-π, π), (-π, π)),
          aspect = AxisAspect(1))

nothing #hide

# We use Makie's `Observable` to animate the data. To dive into how `Observable`s work we
# refer to [Makie.jl's Documentation](https://makie.juliaplots.org/stable/documentation/nodes/index.html).

n = Observable(1)

# We plot the vertical velocity, ``w``.

w_lim = 1e-8
w = @lift interior(w_timeseries[$n], :, 1, :)

contourf!(ax, x, z, w; 
          levels = range(-w_lim, stop=w_lim, length=10),
          colormap = :balance,
          colorrange = (-w_lim, w_lim),
          extendlow = :auto,
          extendhigh = :auto)

title = @lift "ωt = " * string(round(w_timeseries.times[$n] * ω, digits=2))
fig[1, 1] = Label(fig, title, textsize=24, tellwidth=false)

# And, finally, we record a movie.

frames = 1:length(w_timeseries.times)

@info "Animating a propagating internal wave..."

record(fig, filename * ".mp4", frames, framerate=8) do i
    msg = @sprintf("Plotting frame %d of %d...", i, frames[end])
    print(msg * " \r")
    n[] = i
end
nothing #hide

# ![](internal_wave.mp4)
