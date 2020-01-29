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

while model.clock.time < 500
    @printf("t = %.2f s\n", model.clock.time)
    time_step!(model, Δt=0.2, Nt=100)
end

#####
##### Now add the warm bubble perturbation.
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

ρ_plot = contour(model.grid.xC ./ km, model.grid.zC ./ km, rotr90(ρ.data[1:Nx, 1, 1:Nz] .- ρʰᵈ),
                 fill=true, levels=10, xlims=(-5, 5), clims=(-0.008, 0.008), color=:balance, dpi=200)
savefig(ρ_plot, "rho_prime_initial_condition.png")

θ_slice = rotr90(Θ.data[1:Nx, 1, 1:Nz] ./ ρ.data[1:Nx, 1, 1:Nz])
Θ_plot = contour(model.grid.xC ./ km, model.grid.zC ./ km, θ_slice,
                 fill=true, levels=10, xlims=(-5, 5), color=:thermal, dpi=200)
savefig(Θ_plot, "theta_initial_condition.png")

#####
##### Watch the thermal bubble rise!
#####

for n in 1:200
    time_step!(model, Δt=0.1, Nt=50)

    @printf("t = %.2f s\n", model.clock.time)
    xC, yC, zC = model.grid.xC ./ km, model.grid.yC ./ km, model.grid.zC ./ km
    xF, yF, zF = model.grid.xF ./ km, model.grid.yF ./ km, model.grid.zF ./ km

    j = 1
    u_slice = rotr90(model.momenta.ρu.data[1:Nx, j, 1:Nz] ./ model.density.data[1:Nx, j, 1:Nz])
    w_slice = rotr90(model.momenta.ρw.data[1:Nx, j, 1:Nz] ./ model.density.data[1:Nx, j, 1:Nz])
    ρ_slice = rotr90(model.density.data[1:Nx, j, 1:Nz] .- ρʰᵈ)
    θ_slice = rotr90(model.tracers.Θᵐ.data[1:Nx, j, 1:Nz] ./ model.density.data[1:Nx, j, 1:Nz])

    u_title = @sprintf("u, t = %d s", round(Int, model.clock.time))
    pu = contour(xC, zC, u_slice, title=u_title, fill=true, levels=10, xlims=(-5, 5), color=:balance, clims=(-10, 10))
    pw = contour(xC, zC, w_slice, title="w", fill=true, levels=10, xlims=(-5, 5), color=:balance, clims=(-10, 10))
    pρ = contour(xC, zC, ρ_slice, title="rho_prime", fill=true, levels=10, xlims=(-5, 5), color=:balance, clims=(-0.006, 0.006))
    pθ = contour(xC, zC, θ_slice, title="theta", fill=true, levels=10, xlims=(-5, 5), color=:thermal, clims=(299.9, 302))

    p = plot(pu, pw, pρ, pθ, layout=(2, 2), dpi=200, show=true)
    savefig(p, @sprintf("thermal_bubble_%03d.png", n))
end

θ_1000 = (model.tracers.Θᵐ.data[1:Nx, 1, 1:Nz] ./ model.density.data[1:Nx, 1, 1:Nz]) .- θₛ
w_1000 = (model.momenta.ρw.data[1:Nx, 1, 1:Nz] ./ model.density.data[1:Nx, 1, 1:Nz])

@printf("θ′: min=%.2f, max=%.2f\n", minimum(θ_1000), maximum(θ_1000))
@printf("w:  min=%.2f, max=%.2f\n", minimum(w_1000), maximum(w_1000))
