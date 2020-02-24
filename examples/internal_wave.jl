# # Internal wave example
#
# In this example, we initialize an internal wave packet in two-dimensions
# and watch it propagate.

using Oceananigans, Plots, Printf

# ## Numerical, domain, and internal wave parameters
#
# First, we pick some numerical and physical parameters for our model
# and its rotation rate.

Nx = 128 # resolution
Lx = 2π  # domain extent
nothing # hide

# We set up an internal wave with the pressure field
#
# $ p(x, y, z, t) = a(x, z) cos(kx + mz - ω t) $.
#
# where `m` is the vertical wavenumber, `k` is the horizontal wavenumber,
# `ω` is the wave frequncy, and `a(x, z)` is a Gaussian envelope.

## Non-dimensional internal wave parameters
m = 16      # vertical wavenumber
k = 1       # horizontal wavenumber
N = 1       # buoyancy frequency
f = 0.2     # inertial frequency
nothing # hide

# ## A Gaussian wavepacket
#
# Next, we set up an initial condition corresponding to a propagating
# wave packet with a Gaussian envelope. The internal wave dispersion relation yields

ω² = (N^2 * k^2 + f^2 * m^2) / (k^2 + m^2)

## and thus
ω = sqrt(ω²)
nothing # hide

# The internal wave polarization relations follow from the linearized
# Boussinesq equations,

U = k * ω   / (ω^2 - f^2)
V = k * f   / (ω^2 - f^2)
W = m * ω   / (ω^2 - N^2)
B = m * N^2 / (ω^2 - N^2)
nothing # hide

# Finally, we set-up a small-amplitude, Gaussian envelope for the wave packet

## Some Gaussian parameters
A, x₀, z₀, δ = 1e-9, Lx/2, -Lx/2, Lx/15

## A Gaussian envelope
a(x, z) = A * exp( -( (x - x₀)^2 + (z - z₀)^2 ) / 2δ^2 )
nothing # hide

# Create initial condition functions
u₀(x, y, z) = a(x, z) * U * cos(k*x + m*z)
v₀(x, y, z) = a(x, z) * V * sin(k*x + m*z)
w₀(x, y, z) = a(x, z) * W * cos(k*x + m*z)
b₀(x, y, z) = a(x, z) * B * sin(k*x + m*z) + N^2 * z
nothing # hide

# We are now ready to instantiate our model on a uniform grid.
# We give the model a constant rotation rate with background vorticity `f`,
# use temperature as a buoyancy tracer, and use a small constant viscosity
# and diffusivity to stabilize the model.

model = Model(
        grid = RegularCartesianGrid(size=(Nx, 1, Nx), length=(Lx, Lx, Lx)),
     closure = ConstantIsotropicDiffusivity(ν=1e-6, κ=1e-6),
    coriolis = FPlane(f=f),
     tracers = :b,
    buoyancy = BuoyancyTracer()
)
nothing # hide

# We initialize the velocity and buoyancy fields
# with our internal wave initial condition.

set!(model, u=u₀, v=v₀, w=w₀, b=b₀)

# ## A wave packet on the loose
#
# Finally, we release the packet and watch it go!

simulation = Simulation(model, Δt = 0.001 * 2π/ω, stop_iteration = 0,
                        progress_frequency = 20)

anim = @animate for i=1:100
    simulation.stop_iteration += 20
    run!(simulation)

    x, z = model.grid.xC, model.grid.zF
    w = model.velocities.w
    heatmap(x, z, w.data[1:Nx, 1, 1:Nx+1]', title=@sprintf("t = %.2f", model.clock.time),
            xlabel="x", ylabel="z", c=:balance, clims=(-1e-8, 1e-8))
end

mp4(anim, "internal_wave.mp4", fps = 15) # hide
