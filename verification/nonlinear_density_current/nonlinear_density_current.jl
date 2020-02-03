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

Δ = 100  # grid spacing [m]

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

model = CompressibleModel(
                      grid = grid,
                  buoyancy = gas,
        reference_pressure = pₛ,
    thermodynamic_variable = ModifiedPotentialTemperature(),
                   tracers = (:Θᵐ,),
                   closure = ConstantIsotropicDiffusivity(ν=0.5, κ=0.5),
                   forcing = forcing
)

set!(model.density, ρ₀)
set!(model.tracers.Θᵐ, Θ₀)

#####
##### Run a dry adiabatic atmosphere to hydrostatic balance
#####

while model.clock.time < 500
    @printf("t = %.2f s\n", model.clock.time)
    time_step!(model, Δt=0.25, Nt=100)
end

#####
##### Now add the cold bubble perturbation.
#####

ρʰᵈ = model.density.data[1:Nx, 1, 1:Nz]
Θʰᵈ = model.tracers.Θᵐ.data[1:Nx, 1, 1:Nz]

xC, zC = grid.xC, grid.zC
ρ, Θ = model.density, model.tracers.Θᵐ
for k in 1:Nz, i in 1:Nx
    π = Π(i, 1, k, grid, gas, Θ)

    ΔTᵢₖ = ΔT(xC[i], 0, zC[k])
    if ΔTᵢₖ ≈ 0
        θ = Θ[i, 1, k] / ρ[i, 1, k]
    else
        θ = Θ[i, 1, k] / ρ[i, 1, k] + ΔT(xC[i], 0, zC[k]) / π
    end

    ρ[i, 1, k] = pₛ / (Rᵈ*θ) * π^(cᵥ/Rᵈ)
    Θ[i, 1, k] = ρ[i, 1, k] * θ
end

ρ_plot = contour(model.grid.xC ./ km, model.grid.zC ./ km, rotr90(ρ.data[1:Nx, 1, 1:Nz] .- ρʰᵈ),
                 fill=true, levels=10, ylims=(0, 6.4), clims=(-0.05, 0.05), color=:balance, aspect_ratio=:equal, dpi=200)
savefig(ρ_plot, "rho_prime_initial_condition.png")

θ_slice = rotr90(Θ.data[1:Nx, 1, 1:Nz] ./ ρ.data[1:Nx, 1, 1:Nz])
Θ_plot = contour(model.grid.xC ./ km, model.grid.zC ./ km, θ_slice,
                 fill=true, levels=10, ylims=(0, 6.4), color=:thermal, aspect_ratio=:equal, dpi=200)
savefig(Θ_plot, "theta_initial_condition.png")

#####
##### Watch the density current evolve!
#####

for n = 1:180
    @printf("t = %.2f s\n", model.clock.time)
    time_step!(model, Δt=0.1, Nt=50)

    xC, yC, zC = model.grid.xC ./ km, model.grid.yC ./ km, model.grid.zC ./ km
    xF, yF, zF = model.grid.xF ./ km, model.grid.yF ./ km, model.grid.zF ./ km

    j = 1
    u_slice = rotr90(model.momenta.ρu.data[1:Nx, j, 1:Nz] ./ model.density.data[1:Nx, j, 1:Nz])
    w_slice = rotr90(model.momenta.ρw.data[1:Nx, j, 1:Nz] ./ model.density.data[1:Nx, j, 1:Nz])
    ρ_slice = rotr90(model.density.data[1:Nx, j, 1:Nz] .- ρʰᵈ)
    θ_slice = rotr90(model.tracers.Θᵐ.data[1:Nx, j, 1:Nz] ./ model.density.data[1:Nx, j, 1:Nz])

    u_title = @sprintf("u, t = %d s", round(Int, model.clock.time))
    pu = contour(xC, zC, u_slice, title=u_title, fill=true, levels=10, color=:balance, clims=(-20, 20), linewidth=0)
    pw = contour(xC, zC, w_slice, title="w", fill=true, levels=10, color=:balance, clims=(-20, 20), linewidth=0)
    pρ = contour(xC, zC, ρ_slice, title="rho_prime", fill=true, levels=10, color=:balance, clims=(-0.05, 0.05), linewidth=0)
    pθ = contour(xC, zC, θ_slice, title="theta", fill=true, levels=10, color=:thermal, clims=(284, 300), linewidth=0)

    p = plot(pu, pw, pρ, pθ, layout=(4, 1), dpi=300, show=true)
    savefig(p, @sprintf("density_current_%03d.png", n))
end
