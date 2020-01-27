"""
This example sets up a dry, warm thermal bubble perturbation in a uniform
lateral mean flow which buoyantly rises.
"""

using Printf
using Plots
using Oceananigans
using JULES
using JULES: Π

const km = 1000
const hPa = 100

Lx = 20km
Lz = 10km

Δ = 125  # grid spacing [m]

Nx = Int(Lx/Δ)
Ny = 1
Nz = Int(Lz/Δ)

grid = RegularCartesianGrid(size=(Nx, Ny, Nz), halo=(2, 2, 2),
                            x=(-Lx/2, Lx/2), y=(-Lx/2, Lx/2), z=(0, Lz))

#####
##### Dry thermal bubble perturbation
#####

xᶜ, zᶜ = 0km, 2km
xʳ, zʳ = 2km, 2km

@inline L(x, y, z) = √(((x - xᶜ) / xʳ)^2 + ((z - zᶜ) / zʳ)^2)
@inline function θ′(x, y, z)
    ℓ = L(x, y, z)
    return (ℓ <= 1) * 2cos(π/2 * ℓ)^2
end

#####
##### Set up model
#####

pₛ = 1000hPa
Tₐ = 300
g  = 9.80665

gas = IdealGas()
Rᵈ, cₚ, cᵥ = gas.Rᵈ, gas.cₚ, gas.cᵥ

H = Rᵈ * Tₐ / g    # Scale height [m]
ρₛ = pₛ / (Rᵈ*Tₐ)  # Surface density [kg/m³]

p₀(x, y, z) = pₛ * exp(-z/H)
ρ₀(x, y, z) = ρₛ * exp(-z/H)

θ₀(x, y, z) = Tₐ * exp(z/H * Rᵈ/cₚ)
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
##### Run an isothermal atmosphere to hydrostatic balance
#####

while model.clock.time <= 500
    @printf("t = %.2f s\n", model.clock.time)
    time_step!(model, Δt=0.2, Nt=100)
end

#####
##### Now add the cold bubble perturbation.
#####

ρʰᵈ = model.density.data[1:Nx, 1, 1:Nz]
Θʰᵈ = model.tracers.Θᵐ.data[1:Nx, 1, 1:Nz]

xC, zC = grid.xC, grid.zC
ρ, Θ = model.density, model.tracers.Θᵐ
for k in 1:Nz, i in 1:Nx
    θ = Θ[i, 1, k] / ρ[i, 1, k] + θ′(xC[i], 0, zC[k])
    π = Π(i, 1, k, grid, gas, Θ)

    ρ[i, 1, k] = pₛ / (Rᵈ*θ) * π^(cᵥ/Rᵈ)
    Θ[i, 1, k] = ρ[i, 1, k] * θ
end

ρ_plot = contour(model.grid.xC, model.grid.zC, rotr90(ρ.data[1:Nx, 1, 1:Nz]), fill=true, levels=10, color=:balance, show=true)
savefig(ρ_plot, "rho.png")

Θ_plot = contour(model.grid.xC, model.grid.zC, rotr90(Θ.data[1:Nx, 1, 1:Nz]), fill=true, levels=10, color=:thermal, show=true)
savefig(Θ_plot, "Theta.png")

#####
##### Watch the thermal bubble rise!
#####

for i = 1:1000
    time_step!(model, Δt=0.1, Nt=10)

    @printf("t = %.2f s\n", model.clock.time)
    xC, yC, zC = model.grid.xC ./ km, model.grid.yC ./ km, model.grid.zC ./ km
    xF, yF, zF = model.grid.xF ./ km, model.grid.yF ./ km, model.grid.zF ./ km

    j = 1
    U_slice = rotr90(model.momenta.ρu.data[1:Nx, j, 1:Nz])
    W_slice = rotr90(model.momenta.ρw.data[1:Nx, j, 1:Nz])
    ρ_slice = rotr90(model.density.data[1:Nx, j, 1:Nz] .- ρʰᵈ)
    Θ_slice = rotr90(model.tracers.Θᵐ.data[1:Nx, j, 1:Nz] .- Θʰᵈ)

    pU = contour(xC, zC, U_slice, fill=true, levels=10, color=:balance, clims=(-4, 4))
    pW = contour(xC, zC, W_slice, fill=true, levels=10, color=:balance, clims=(-4, 4))
    pρ = contour(xC, zC, ρ_slice, fill=true, levels=10, color=:balance, clims=(-0.01, 0.01))
    pΘ = contour(xC, zC, Θ_slice, fill=true, levels=10, color=:thermal)

    display(plot(pU, pW, pρ, pΘ, layout=(2, 2), show=true))
end

