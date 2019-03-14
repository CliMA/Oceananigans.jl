using Oceananigans
using CuArrays

Nx, Ny, Nz = 128, 128, 128
Lx, Ly, Lz = 100, 100, 100
Nt, Δt = 10000, 1
ν, κ = 1e-2, 1e-2

model = Model(N=(Nx, Ny, Nz), L=(Lx, Ly, Lz), ν=ν, κ=κ,arch=:GPU,float_type=Float32)
model.boundary_conditions = BoundaryConditions(:periodic, :periodic, :rigid_lid, :free_slip)

#@inline fT(u,v,w,T,S,Nx,Ny,Nz,dx,dy,dz,i,j,k ) = ifelse(k ==1, -9e-6,0)
#model.forcing = Forcing(nothing,nothing,nothing, fT, nothing)

N² = 2e-5
Tz = 0.01
T_prof = 273.15 .+ 20 .+ Tz .* model.grid.zC

T_3d = repeat(reshape(T_prof, 1, 1, Nz), Nx, Ny, 1)
@. T_3d[:, :, 1:round(Int, Nz/2)] += 0.01*rand()

model.tracers.T.data .= CuArray(T_3d)

# Write temperature field to disk every 10 time steps.
output_writer = NetCDFOutputWriter(dir=".", prefix="convection", frequency=2500)
push!(model.output_writers, output_writer)

# Time stepping
time_step!(model, Nt, Δt)
