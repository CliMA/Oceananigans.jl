#=

* Example description here *

=#

using Oceananigans, Printf, PyPlot

# Internal wave parameters
Nx, Lx = 128, 2π
A, x₀, z₀, δ = 1e-9, Lx/2, -Lx/2, Lx/15 # Wave envelope parameters
m, k = 16, 1 # Vertical and horizontal wavenumber
N, f = 1, 0.2 # Buoyancy frequency and Coriolis parameter (non-dimensionalized by N)

# Internal wave frequency via the dispersion relation
ω = sqrt( (N^2*k^2 + f^2*m^2) / (k^2 + m^2) )
Δt = 0.001 * 2π/ω

# Polarization relations
U = k * ω   / (ω^2 - f^2)
V = k * f   / (ω^2 - f^2)
W = m * ω   / (ω^2 - ℕ^2)
Θ = m * N^2 / (ω^2 - ℕ^2)

a(x, y, z) = A * exp( -((x-x₀)^2 + (z-z₀)^2) / (2*δ)^2 )

u₀(x, y, z) = a(x, y, z) * U * cos(k*x + m*z)
v₀(x, y, z) = a(x, y, z) * V * sin(k*x + m*z)
w₀(x, y, z) = a(x, y, z) * W * cos(k*x + m*z)
T₀(x, y, z) = a(x, y, z) * Θ * sin(k*x + m*z) + ℕ^2 * z

# Create a model where temperature = buoyancy.
model = Model(
        grid = RegularCartesianGrid(N=(Nx, 1, Nx), L=(Lx, Lx, Lx)),
     closure = ConstantIsotropicDiffusivity(ν=1e-6, κ=1e-6),
    coriolis = FPlane(f=f), 
    buoyancy = BuoyancyTracer()
)

set!(model, u=u₀, v=v₀, w=w₀, T=T₀)

xplot(u) = repeat(dropdims(xnodes(u), dims=2), 1, u.grid.Nz)
zplot(u) = repeat(dropdims(znodes(u), dims=2), u.grid.Nx, 1)

function plot_field!(ax, w, t) 
    pcolormesh(xplot(w), zplot(w), data(model.velocities.w)[:, 1, :])
    xlabel(L"x")
    ylabel(L"z")
    title(@sprintf("\$ \\omega t / 2 \\pi = %.2f\$", t*ω/2π))
    ax.set_aspect(1)
    pause(0.1)
    return nothing
end

close("all")
fig, ax = subplots()

for i = 1:10
    time_step!(model, 200, Δt)
    plot_field!(ax, model.velocities.w, model.clock.time)
end
