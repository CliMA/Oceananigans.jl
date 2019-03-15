using Oceananigans
using CuArrays

#physical constants
p0 = 1027
cp = 4181.3

#set simulation parameters
Nx, Ny, Nz = 128, 128, 128
Lx, Ly, Lz = 100, 100, 100
Nt, Δt = 10000, 6
ν, κ = 1e-4, 1e-4
Tz = 0.01
bottom_gradient = Tz
top_flux = 75 / (p0 * cp) #for flux bc this should be positive for cooling

#create the model
model = Model(N=(Nx, Ny, Nz), L=(Lx, Ly, Lz), ν=ν, κ=κ, arch=:GPU, float_type=Float32)

#set boundary conditions
model.boundary_conditions.T.z.left = BoundaryCondition(Flux,top_flux)
model.boundary_conditions.T.z.right  = BoundaryCondition(Gradient, bottom_gradient)
#the default boundary conditions for velocity is free-slip

#set initial condition
T_prof = 273.15 .+ 20 .+ Tz .* model.grid.zC
#make initial condition into 3D array
T_3d = repeat(reshape(T_prof, 1, 1, Nz), Nx, Ny, 1)
@. T_3d[:, :, 1:round(Int, Nz/2)] += 0.01*randn() #add noise to the top half of the domain for convection

model.tracers.T.data .= CuArray(T_3d)

# Write temperature field to disk every 10 time steps.
output_writer = NetCDFOutputWriter(dir=".", prefix="convection", frequency=2500)
push!(model.output_writers, output_writer)

# Time stepping
time_step!(model, Nt, Δt)
