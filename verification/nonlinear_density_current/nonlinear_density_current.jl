"""
This example sets up a cold bubble perturbation which develops into a non-linear
density current. This numerical test case is described by Straka et al. (1993).
Also see: http://www2.mmm.ucar.edu/projects/srnwp_tests/density/density.html

Straka et al. (1993). "Numerical Solutions of a Nonlinear Density-Current -
    A Benchmark Solution and Comparisons." International Journal for Numerical
    Methods in Fluids 17, pp. 1-22.
"""

using Printf
using Plots
using Oceananigans
using JULES
using JULES: Π

const km = 1000
const hPa = 100

Lx = 51.2km
Lz = 6.4km

Δ = 200  # grid spacing [m]

Nx = Int(Lx/Δ)
Ny = 1
Nz = Int(Lz/Δ)

grid = RegularCartesianGrid(size=(Nx, Ny, Nz), halo=(2, 2, 2),
                            x=(-Lx/2, Lx/2), y=(-Lx/2, Lx/2), z=(0, Lz))

#####
##### Initial perturbation
#####

xᶜ, xʳ = 0km, 4km
zᶜ, zʳ = 3km, 2km

function ΔT(x, y, z)
    L = √(((x - xᶜ) / xʳ)^2 + ((z - zᶜ) / zʳ)^2)
    L > 1 && return 0
    L ≤ 1 && return -15 * (1 + cos(π*L)) / 2
end

#####
##### Set up model
#####

gas = IdealGas()
Rᵈ, cₚ, cᵥ = gas.Rᵈ, gas.cₚ, gas.cᵥ

g  = 9.80665
pₛ = 1000hPa
θₛ = 300
πₛ = 1

θ₀(x, y, z) = θₛ
π₀(x, y, z) = πₛ - g*z / (cₚ*θₛ)
p₀(x, y, z) = pₛ * π₀(x, y, z)^(cₚ/Rᵈ)
ρ₀(x, y, z) = pₛ / (Rᵈ * θ₀(x, y, z)) * π₀(x, y, z)^(cᵥ/Rᵈ)
Θ₀(x, y, z) = ρ₀(x, y, z) * θ₀(x, y, z)

const τ⁻¹ = 1     # Damping/relaxation time scale [s⁻¹]. This is very strong damping.
const Δμ = 0.1Lz  # Sponge layer width [m] set to 10% of the domain height.
@inline μ(z, Lz) = τ⁻¹ * exp(-(Lz-z) / Δμ)

@inline Fw(i, j, k, grid, t, Ũ, C̃, p) = @inbounds (t <= 500) * -μ(grid.zF[k], grid.Lz) * Ũ.ρw[i, j, k]
forcing = ModelForcing(w=Fw)

model = CompressibleModel(grid=grid, buoyancy=gas, reference_pressure=pₛ,
                          prognostic_temperature=ModifiedPotentialTemperature(),
                          tracers=(:Θᵐ,), forcing=forcing)

set!(model.density, ρ₀)
set!(model.tracers.Θᵐ, Θ₀)

#####
##### Run a dry adiabatic atmosphere to hydrostatic balance
#####

# while model.clock.time < 500
#     @printf("t = %.2f s\n", model.clock.time)
#     time_step!(model, Δt=0.5, Nt=100)
# end

#####
##### Now add the cold bubble perturbation.
#####

ρʰᵈ = model.density.data[1:Nx, 1, 1:Nz]
Θʰᵈ = model.tracers.Θᵐ.data[1:Nx, 1, 1:Nz]

xC, zC = grid.xC, grid.zC
ρ, Θ = model.density, model.tracers.Θᵐ
for k in 1:Nz, i in 1:Nx
    π = Π(i, 1, k, grid, gas, Θ)
    θ = Θ[i, 1, k] / ρ[i, 1, k] + π / ΔT(xC[i], 0, zC[k])

    ρ[i, 1, k] = pₛ / (Rᵈ*θ) * π^(cᵥ/Rᵈ)
    Θ[i, 1, k] = ρ[i, 1, k] * θ
end

ρ_plot = contour(model.grid.xC ./ km, model.grid.zC ./ km, rotr90(ρ.data[1:Nx, 1, 1:Nz] .- ρʰᵈ),
                 fill=true, levels=10, color=:balance, dpi=200)
savefig(ρ_plot, "rho_prime_initial_condition.png")

θ_slice = rotr90(Θ.data[1:Nx, 1, 1:Nz] ./ ρ.data[1:Nx, 1, 1:Nz])
Θ_plot = contour(model.grid.xC ./ km, model.grid.zC ./ km, θ_slice,
                 fill=true, levels=10, color=:thermal, dpi=200)
savefig(Θ_plot, "theta_initial_condition.png")

#####
##### Watch the density current evolve!
#####

# for i = 1:1000
#     @printf("t = %.2f s\n", model.clock.time)
#     time_step!(model, Δt=0.1, Nt=100)
#
#     xC, yC, zC = model.grid.xC ./ km, model.grid.yC ./ km, model.grid.zC ./ km
#     xF, yF, zF = model.grid.xF ./ km, model.grid.yF ./ km, model.grid.zF ./ km
#
#     j = 1
#     U_slice = rotr90(model.momenta.ρu.data[1:Nx, j, 1:Nz])
#     W_slice = rotr90(model.momenta.ρw.data[1:Nx, j, 1:Nz])
#     ρ_slice = rotr90(model.density.data[1:Nx, j, 1:Nz] .- ρ_hd)
#     Θ_slice = rotr90(model.tracers.Θᵐ.data[1:Nx, j, 1:Nz] .- Θ_hd)
#
#     pU = contour(xC, zC, U_slice; fill=true, levels=10, color=:balance, clims=(-10, 10))
#     pW = contour(xC, zC, W_slice; fill=true, levels=10, color=:balance, clims=(-10, 10))
#     pρ = contour(xC, zC, ρ_slice; fill=true, levels=10, color=:balance, clims=(-0.05, 0.05))
#     pΘ = contour(xC, zC, Θ_slice; fill=true, levels=10, color=:thermal)
#
#     display(plot(pU, pW, pρ, pΘ, layout=(4, 1), show=true))
# end
