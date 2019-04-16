using Oceananigans

include("utils.jl")

Nx, Ny, Nz = 256, 1, 256
Lx, Ly, Lz = 1e5, 1e5, 4000
Nt, Δt = 500, 2

# We're assuming molecular viscosity and diffusivity.
model = Model(N=(Nx, Ny, Nz), L=(Lx, Ly, Lz), ν=1e-2, κ=1e-2)

# Set a temperature perturbation with a Gaussian profile located at the center
# of the vertical slice. It roughly corresponds to a background temperature of
# T = 282.99 K and a bubble temperature of T = 283.01 K.
xC, zC = model.grid.xC, model.grid.zC
zC = zC'

T0 = model.eos.T₀ # Reference temperature
ρ0 = 1.027e3      # Reference water density
ΔT = 10           # Magnitude of temperature perturbation
αᵥ = model.eos.αᵥ # Coefficient of thermal expansion
σ = Lx/10.0       # Characteristic distance over which anomaly is spread
g = model.constants.g
f = model.constants.f

T = T0 .+ ΔT .* exp.((-(xC .- Lx/2).^2 ) ./ σ^2)
V = ((2g*αᵥ*ΔT .* (zC .+ Lz) ./ (ρ0*f*σ^2)) .* (xC .- Lx/2) .* exp.((-(xC .- Lx/2).^2 ) ./ σ^2))

model.tracers.T.data .= repeat(T, outer=[1, Ny, Nz])
model.velocities.v.data .= reshape(V, (Nx, Ny, Nz))

# No-slip boundary conditions at the bottom for u and v.
model.boundary_conditions.u.z.right = BoundaryCondition(NoSlip, nothing)
model.boundary_conditions.v.z.right = BoundaryCondition(NoSlip, nothing)

nc_writer = NetCDFOutputWriter(dir="netcdf_out", prefix="thermal_wind_", frequency=100)
push!(model.output_writers, nc_writer)

time_step!(model, Nt, Δt)

make_vertical_slice_movie(model, nc_writer, "T", Nt, Δt, T0)
make_vertical_slice_movie(model, nc_writer, "v", Nt, Δt, 0)
