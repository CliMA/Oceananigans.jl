using JULES, Oceananigans
using Plots, Printf
using OffsetArrays

Nx = Ny = 1
Nz = 32
L = 10e3

grid = RegularCartesianGrid(size=(Nx, Ny, Nz), x=(0, L), y=(0, L), z=(0, L))

pₛ = 100000
Tₐ = 293.15
g  = 9.80665

buoyancy = IdealGas()
Rᵈ = buoyancy.Rᵈ
cₚ = buoyancy.cₚ
ρₛ = pₛ / (Rᵈ*Tₐ)

# Isothermal atmosphere
H = Rᵈ * Tₐ / g
p₀(x, y, z) = pₛ * exp(-z/H)
ρ₀(x, y, z) = ρₛ * exp(-z/H)

θ₀(x, y, z) = Tₐ * exp(z/H * Rᵈ/cₚ)
Θ₀(x, y, z) = ρ₀(x, y, z) * θ₀(x, y, z)

model = CompressibleModel(grid=grid, buoyancy=buoyancy, reference_pressure=pₛ)

set!(model.density, ρ₀)
set!(model.tracers.Θᵐ, Θ₀)

Δtp = 0.5
Θ₀_prof = model.tracers.Θᵐ[1, 1, 1:Nz]
ρ₀_prof = model.density[1, 1, 1:Nz]

for i = 1:10
    for _ in 1:10
        time_step!(model; Δt=Δtp, nₛ=1)
    end
 
    Θ_prof = model.tracers.Θᵐ[1, 1, 1:Nz] .- Θ₀_prof
    W_prof = model.momenta.W[1, 1, 1:Nz+1]
    ρ_prof = model.density[1, 1, 1:Nz] .- ρ₀_prof

    Θ_plot = plot(Θ_prof, grid.zC, xlim=(-2.5, 2.5), label="")
    W_plot = plot(W_prof, grid.zF, xlim=(-1.2, 1.2), label="")
    ρ_plot = plot(ρ_prof, grid.zC, xlim=(-0.01, 0.01), label="")

    t_str = @sprintf("t = %d s", model.clock.time)
    display(plot(Θ_plot, W_plot, ρ_plot, title=["Theta_prime" "W" "rho, $t_str"], layout=(1, 3), show=true))

    CFL = maximum(abs, model.momenta.W.data) * Δtp / grid.Δz
    @show CFL
end

