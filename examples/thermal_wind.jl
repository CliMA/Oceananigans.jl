using Oceananigans

include("utils.jl")

Nx, Ny, Nz = 256, 1, 256
Lx, Ly, Lz = 1e6, 1e6, 4000
Nt, Δt = 5000, 10

model = Model(N=(Nx, Ny, Nz), L=(Lx, Ly, Lz))

# Set a temperature perturbation with a Gaussian profile located at the center
# of the vertical slice. It roughly corresponds to a background temperature of
# T = 282.99 K and a bubble temperature of T = 283.01 K.
xC, zC = model.grid.xC, model.grid.zC
zC = zC'

T0 = 283.0 # Reference temperature
rho0 = 1.027e3 # Reference water density
DeltaT = 0.1 # Magnitude of temperature perturbation
alpha = 207e-6 # Coefficient of thermal expansion
sigma = Lx/10.0 # Characteristic distance over which anomaly is spread
g = 9.81
f = 1e-4

T = T0 .+ DeltaT * exp.( ( -(xC .- Lx/2).^2 ) ./ sigma.^2  )
V = -2.0*g*alpha*DeltaT.*zC / (rho0*f*sigma.^2) .* (xC .- Lx/2) .* exp.( ( -(xC .- Lx/2).^2 ) ./ sigma.^2  )

model.tracers.T.data .= repeat(T, outer = [1, Ny, Nz])
model.velocities.v.data .= reshape(V, (Nx, Ny, Nz))
#model.boundary_conditions.v.z.right = BoundaryCondition(Value, 0)


# Add a NetCDF output writer that saves NetCDF files to the current directory
# "." with a filename prefix of "thermal_bubble_" every 10 iterations.
nc_writer = NetCDFOutputWriter(dir="netcdf_out", prefix="thermal_wind_", frequency=100)
push!(model.output_writers, nc_writer)

time_step!(model, Nt, Δt)

make_vertical_slice_movie(model, nc_writer, "T", Nt, Δt, T0)
make_vertical_slice_movie(model, nc_writer, "v", Nt, Δt, 0)
