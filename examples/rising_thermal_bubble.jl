using JULES, Oceananigans
using Plots, Printf
using OffsetArrays

Nx = Nz = 32
Ny = 8
L = 100

grid = RegularCartesianGrid(size=(Nx, Ny, Nz), x=(-L/2, L/2), y=(-L/2, L/2), z=(0, L))

const pₛ = 100000
buoyancy = IdealGas()
const Rᵈ = buoyancy.Rᵈ
const γ  = buoyancy.γ

Θᵇ(x, y, z) = 300 
pᵇ(x, y, z) = pₛ * (Rᵈ * Θᵇ(x, y, z) / pₛ)^γ
ρᵇ(x, y, z) = pᵇ(x, y, z) / (Rᵈ * Θᵇ(x, y, z))

pᵇ_a = @. pᵇ(0, 0, grid.zC)
pᵇ_oa = OffsetArray([pᵇ_a[1]; pᵇ_a; pᵇ_a[end]], 0:Nz+1)
@show pᵇ_oa

base_state = BaseState(p=pᵇ_oa, ρ=ρᵇ, θ=Θᵇ)
model = CompressibleModel(grid=grid, buoyancy=buoyancy, surface_pressure=pₛ, base_state=base_state)

Θ₀(x, y, z) = Θᵇ(x, y, z) * (1 + 0.01 * exp(- (x^2 + (z-50)^2) / L))
p₀(x, y, z) = pₛ * (Rᵈ * Θ₀(x, y, z) / pₛ)^γ
ρ₀(x, y, z) = pᵇ(x, y, z) / (Rᵈ * Θ₀(x, y, z))

set!(model.density, ρ₀)
set!(model.tracers.Θᵐ, Θ₀)

Δtp = 1e-3
time_step!(model; Δt=Δtp, nₛ=1)

#anim = @animate for i=1:10
for i = 1:30
    @show i
    time_step!(model; Δt=Δtp, nₛ=1)

    t = @sprintf("%.3f s", model.clock.time)
    xC, yC, zC = model.grid.xC, model.grid.yC, model.grid.zC
    xF, yF, zF = model.grid.xF, model.grid.yF, model.grid.zF

    j = Int(Ny/2)
    U_slice = rotr90(model.momenta.U.data[1:Nx, j, 1:Nz])
    W_slice = rotr90(model.momenta.W.data[1:Nx, j, 1:Nz])
    ρ_slice = rotr90(model.density.data[1:Nx, j, 1:Nz] .- ρ₀.(xC, 1, zC'))
    Θ_slice = rotr90(model.tracers.Θᵐ.data[1:Nx, j, 1:Nz])

    pU = contour(xC, zC, U_slice; xlabel="x", ylabel="z", fill=true, levels=10, color=:balance, clims=(-5e-3, 5e-3))
    pW = contour(xC, zC, W_slice; xlabel="x", ylabel="z", fill=true, levels=10, color=:balance, clims=(-5e-3, 5e-3))
    pρ = contour(xC, zC, ρ_slice; xlabel="x", ylabel="z", fill=true, levels=10, color=:balance, clims=(-5e-5, 5e-5))
    pΘ = contour(xC, zC, Θ_slice; xlabel="x", ylabel="z", fill=true, levels=10, color=:thermal)

    display(plot(pU, pW, pρ, pΘ, title=["U (m/s), t=$t" "W (m/s)" "rho_prime (kg/m^3)" "Theta_m_prime (K)"], show=true))
end

# gif(anim, "sad_thermal_bubble.gif", fps=10)
