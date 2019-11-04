using JULES, Oceananigans
using Plots, Printf

Nx = Nz = 32
Ny = 8
L = 100

grid = RegularCartesianGrid(size=(Nx, Ny, Nz), x=(-L/2, L/2), y=(-L/2, L/2), z=(-L/2, L/2))
model = CompressibleModel(grid=grid)

p₀ = model.surface_pressure
Rᵈ = model.buoyancy.Rᵈ

# ρ₀(x, y, z) = 1.2 * (1 -  (z - L/2) / 10L)
Θ₀(x, y, z) = 300 + 0.01 * exp(- (x^2 + z^2) / L)
ρ₀(x, y, z) = p₀ / (Rᵈ * Θ₀(x, y, z))

set!(model.density, ρ₀)
set!(model.tracers.Θᵐ, Θ₀)

Δtp = 1e-3
time_step!(model; Δt=Δtp, nₛ=1)

for i=1:100
    time_step!(model; Δt=Δtp, nₛ=1)

    t = @sprintf("%.3f s", model.clock.time)
    xC, yC, zC = model.grid.xC, model.grid.yC, model.grid.zC
    xF, yF, zF = model.grid.xF, model.grid.yF, model.grid.zF

    j = Int(Ny/2)
    U_slice = rotr90(model.momenta.U.data[1:Nx, j, 1:Nz])
    W_slice = rotr90(model.momenta.W.data[1:Nx, j, 1:Nz])
    ρ_slice = rotr90(model.density.data[1:Nx, j, 1:Nz] .- ρ₀.(xC, 1, zC'))
    Θ_slice = rotr90(model.tracers.Θᵐ.data[1:Nx, j, 1:Nz]) .- 300

    pU = contour(xC, zC, U_slice; xlabel="x", ylabel="z", fill=true, levels=10, color=:balance, clims=(-5e-3, 5e-3))
    pW = contour(xC, zC, W_slice; xlabel="x", ylabel="z", fill=true, levels=10, color=:balance, clims=(-5e-3, 5e-3))
    pρ = contour(xC, zC, ρ_slice; xlabel="x", ylabel="z", fill=true, levels=10, color=:balance, clims=(-5e-5, 5e-5))
    pΘ = contour(xC, zC, Θ_slice; xlabel="x", ylabel="z", fill=true, levels=10, color=:thermal, clims=(-0.005, 0.01))

    display(plot(pU, pW, pρ, pΘ, title=["U (m/s), t=$t" "W (m/s)" "rho_prime (kg/m^3)" "Theta_m_prime (K)"], show=true))
end

