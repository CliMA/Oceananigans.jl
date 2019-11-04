using JULES, Oceananigans
using Plots

Nx = Nz = 32
Ny = 8
L = 100

grid = RegularCartesianGrid(size=(Nx, Ny, Nz), x=(-L/2, L/2), y=(-L/2, L/2), z=(-L/2, L/2))
model = CompressibleModel(grid=grid)

Θ₀(x, y, z) = 20 + 0.01 * exp(- (x^2 + z^2) / L)
set!(model.tracers.Θᵐ, Θ₀)
set!(model.density, 1.2)

Δtp = 1e-4
time_step!(model; Δt=Δtp, nₛ=1)

l = @layout [a; b]
for i=1:5
    time_step!(model; Δt=Δtp, nₛ=1)

    j = Int(Ny/2)
    U_slice = model.momenta.U.data[1:Nx, j, 1:Nz]
    W_slice = model.momenta.W.data[1:Nx, j, 1:Nz]
    ρ_slice = model.density.data[1:Nx, j, 1:Nz] .- 1.2
    Θ_slice = model.tracers.Θᵐ.data[1:Nx, j, 1:Nz] .- 20

    x, z = model.grid.xC, model.grid.zC
    pU = contour(x, z, U_slice; fill=true, levels=10, color=:balance, clims=(-1e-3, 1e-3))
    pW = contour(x, z, W_slice; fill=true, levels=10, color=:balance, clims=(-1e-3, 1e-3))
    pρ = contour(x, z, ρ_slice; fill=true, levels=10, color=:balance, clims=(-1e-3, 1e-3))
    pΘ = contour(x, z, Θ_slice; fill=true, levels=10, color=:thermal, clims=(0, 0.01))
    display(plot(pU, pW, pρ, pΘ, title=["U" "W" "rho" "Theta_m"], show=true))
    sleep(0.1)
end

