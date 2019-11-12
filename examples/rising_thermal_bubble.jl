using JULES, Oceananigans
using Plots, Printf
using OffsetArrays

Nx = Nz = 32
Ny = 8
L = 100

grid = RegularCartesianGrid(size=(Nx, Ny, Nz), x=(-L/2, L/2), y=(-L/2, L/2), z=(0, L))

const pₛ = 100000
const Tₐ = 293.15
const g  = 9.80665

buoyancy = IdealGas()
const Rᵈ = buoyancy.Rᵈ
const γ  = buoyancy.γ
const ρₛ = pₛ / (Rᵈ*Tₐ)

# Isothermal atmosphere
pᵇ(x, y, z) = pₛ * exp(- g*z / (Rᵈ*Tₐ))
ρᵇ(x, y, z) = ρₛ * exp(- g*z / (Rᵈ*Tₐ))
Tᵇ(x, y, z) = ρᵇ(x, y, z) * Tₐ

ρ₀(x, y, z) = ρᵇ(x, y, z) + 1e-4 * exp(- (x^2 + (z-50)^2) / L)
T₀(x, y, z) = ρ₀(x, y, z) * Tₐ
p₀(x, y, z) = Rᵈ * T₀(x, y, z)

pᵇa = @. pᵇ(0, 0, grid.zC)
pᵇoa = OffsetArray([pᵇa[1]; pᵇa; pᵇa[end]], 0:Nz+1)

base_state = BaseState(p=pᵇoa, ρ=ρᵇ)

model = CompressibleModel(grid=grid, buoyancy=buoyancy, surface_pressure=pₛ,
                          prognostic_temperature=Temperature(), base_state=base_state)

set!(model.density, ρ₀)
set!(model.tracers.Θᵐ, T₀)

Δtp = 1e-3
time_step!(model; Δt=Δtp, nₛ=1)

#anim = @animate for i=1:10
for i = 1:1
    @show i
    time_step!(model; Δt=Δtp, nₛ=1)

    t = @sprintf("%.3f s", model.clock.time)
    xC, yC, zC = model.grid.xC, model.grid.yC, model.grid.zC
    xF, yF, zF = model.grid.xF, model.grid.yF, model.grid.zF

    j = Int(Ny/2)
    U_slice = rotr90(model.momenta.U.data[1:Nx, j, 1:Nz])
    W_slice = rotr90(model.momenta.W.data[1:Nx, j, 1:Nz])
    ρ_slice = rotr90(model.density.data[1:Nx, j, 1:Nz] .- ρᵇ.(xC, 1, zC'))
    Θ_slice = rotr90(model.tracers.Θᵐ.data[1:Nx, j, 1:Nz] .- Tᵇ.(xC, 1, zC'))

    pU = contour(xC, zC, U_slice; xlabel="x", ylabel="z", fill=true, levels=10, color=:balance, clims=(-5e-3, 5e-3))
    pW = contour(xC, zC, W_slice; xlabel="x", ylabel="z", fill=true, levels=10, color=:balance, clims=(-5e-3, 5e-3))
    pρ = contour(xC, zC, ρ_slice; xlabel="x", ylabel="z", fill=true, levels=10, color=:balance, clims=(-5e-5, 5e-5))
    pΘ = contour(xC, zC, Θ_slice; xlabel="x", ylabel="z", fill=true, levels=10, color=:thermal, clims=(0, 0.01))

    display(plot(pU, pW, pρ, pΘ, title=["rho*u, t=$t" "rho*w" "rho_prime" "T_prime"], show=true))
end

# gif(anim, "sad_thermal_bubble.gif", fps=10)
