using Oceananigans, Printf, CuArrays

 N = (32, 32, 32)
 L = (1, 1, 1)
 a = 1e-1              # noise amplitude for initial condition
Δb = 1.0               # buoyancy differential
 ν = κ = 0.001

arch = GPU()
ArrayType = CuArray

model = Model(arch=arch, N=N, L=L, ν=ν, κ=κ,
      eos = LinearEquationOfState(βT=1., βS=0.),
      constants = PlanetaryConstants(g=1., f=0.))

# Constant buoyancy boundary conditions on "temperature"
model.boundary_conditions.T.z.top = BoundaryCondition(Value, 0.0)
model.boundary_conditions.T.z.bottom = BoundaryCondition(Value, Δb)

# Force salinity as a passive tracer (βS=0)
S★(x, z) = exp(4z) * sin(2π/L[1] * x)
FS(grid, u, v, w, T, S, i, j, k) = 1/10 * (S★(grid.xC[i], grid.zC[k]) - S[i, j, k])
model.forcing = Forcing(FS=FS)

Δt = 0.01 * min(model.grid.Δx, model.grid.Δy, model.grid.Δz)^2 / ν

ξ(z) = a * rand() * z * (L[3] + z) # noise, damped at the walls
b₀(x, y, z) = (ξ(z) - z) / L[3]

x, y, z = model.grid.xC, model.grid.yC, model.grid.zC 
x, y, z = reshape(x, N[1], 1, 1), reshape(y, 1, N[2], 1), reshape(z, 1, 1, N[3])

model.tracers.T.data .= ArrayType(b₀.(x, y, z))

@time time_step!(model, 10, Δt)
