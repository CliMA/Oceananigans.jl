using Distributed
addprocs(1)  # For asynchronous output writing.

# use keyword arguments
wmsq = parse(Float64,ARGS[1])
grad = parse(Float64,ARGS[2])
res = parse(Int,ARGS[3])
dt = parse(Float64,ARGS[4])
days = parse(Float64,ARGS[5])
ν = parse(Float64,ARGS[6])
fname = "convection"*"_"*string(wmsq)*"_"*string(grad)*"_"*string(res)*"_"*string(dt)*"_"*string(days)*"_"*string(ν)*"_"

# Apparently I have to do this even when starting julia with "-p 2 --project"
# otherwise we get "ERROR: LoadError: On worker 2: ArgumentError: Package Oceananigans
# not found in current path:"
@everywhere using Pkg
@everywhere Pkg.activate(".");

@everywhere using Oceananigans
@everywhere using CuArrays

# physical constants
p0 = 1027
cp = 4181.3

# set simulation parameters
Nx, Ny, Nz = res, res, res
Lx, Ly, Lz = 100, 100, 100
Nt, Δt = Int(days*60*60*24/dt), dt
κ = ν
Tz = grad
bottom_gradient = Tz
top_flux = wmsq / (p0 * cp) #for flux bc this should be positive for cooling

# create the model
model = Model(N=(Nx, Ny, Nz), L=(Lx, Ly, Lz), ν=ν, κ=κ, arch=GPU(), float_type=Float64)

# set boundary conditions
model.boundary_conditions.T.z.left = BoundaryCondition(Flux, top_flux)
model.boundary_conditions.T.z.right = BoundaryCondition(Gradient, bottom_gradient)
# the default boundary conditions for velocity is free-slip

# set initial condition
T_prof = 273.15 .+ 20 .+ Tz .* model.grid.zC
# make initial condition into 3D array
T_3d = repeat(reshape(T_prof, 1, 1, Nz), Nx, Ny, 1)
@. T_3d[:, :, 1:round(Int, Nz/2)] += 0.01*randn() #add noise to the top half of the domain for convection

model.tracers.T.data .= CuArray(T_3d)

# Write temperature field to disk every frequency time steps.
output_writer = NetCDFOutputWriter(dir=".", prefix=fname, padding = 0,  naming_scheme = :file_number, frequency=3600, async=true)
push!(model.output_writers, output_writer)

#Time stepping
@sync begin
      for i = 1:ceil(Int,Nt/100)
          tic = time_ns()
          print("Time: $(model.clock.time) ")
          time_step!(model, 100, Δt)
          println(prettytime(time_ns() - tic))
      end
end
