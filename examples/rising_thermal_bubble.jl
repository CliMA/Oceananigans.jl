using JULES, Oceananigans

Nx = Nz = 32
Ny = 8
L = 100

grid = RegularCartesianGrid(size=(Nx, Ny, Nz), x=(0, L), y=(0, L), z=(0, L))
model = CompressibleModel(grid=grid)

model.tracers.Θᵐ.data .= 20

# Add a cube-shaped warm temperature anomaly that takes up the middle 50% of the domain volume.
i1, i2 = round(Int, Nx/4), round(Int, 3Nx/4)
j1, j2 = round(Int, Ny/4), round(Int, 3Ny/4)
k1, k2 = round(Int, Nz/4), round(Int, 3Nz/4)
model.tracers.Θᵐ.data[i1:i2, j1:j2, k1:k2] .+= 0.01

model.density.data .= 1.2

time_step!(model; Δt=1e-5, nₛ=1)

