using ArgParse

s = ArgParseSettings(description="Run simulations of oceanic free convection.")

@add_arg_table s begin
    "--resolution", "-N"
        arg_type=Int
        required=true
        help="Number of grid points in each dimension (Nx, Ny, Nz) = (N, N, N)."
    "--heat-flux"
        arg_type=Float64
        required=true
        help="Heat flux to impose at the surface [W/m²]. Negative values imply a cooling flux."
    "--dTdz"
        arg_type=Float64
        required=true
        help="Temperature gradient to impose at the bottom [K/m]."
    "--diffusivity"
        arg_type=Float64
        required=true
        help="Diffusivity κ [m²/s]."
    "--dt"
        arg_type=Float64
        required=true
        help="Time step in seconds."
    "--days"
        arg_type=Float64
        required=true
        help="Number of simulation days to run the model."
    "--output-dir", "-d"
        arg_type=AbstractString
        required=true
        help="Base directory to save output to."
end

parsed_args = parse_args(s)
N = parsed_args["resolution"]
Q = parsed_args["heat-flux"]
dTdz = parsed_args["dTdz"]
κ = parsed_args["diffusivity"]
dt = isinteger(parsed_args["dt"]) ? Int(dt) : dt
days = isinteger(parsed_args["days"]) ? Int(days) : days
base_dir = parsed_args["output-dir"]

filename_prefix = "free_convection_N" * string(N) * "_Q" * string(heat_flux) *
                  "_dTdz" * string(dTdz) * "_k" * string(κ) * "_dt" * string(dt) *
                  "_days" * string(days)
output_dir = joinpath(base_dir, filename_prefix)

if !isdir(output_dir)
    println("Creating directory: $output_dir")
    mkpath(output_dir)
end
using Distributed
addprocs(1)  # For asynchronous output writing.

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
Nx, Ny, Nz = 128, 128, 128
Lx, Ly, Lz = 100, 100, 100
Nt, Δt = 1000, 1
ν, κ = 1e-4, 1e-4
Tz = 0.01
bottom_gradient = Tz
top_flux = 75 / (p0 * cp) #for flux bc this should be positive for cooling

# create the model
model = Model(N=(Nx, Ny, Nz), L=(Lx, Ly, Lz), ν=ν, κ=κ, arch=:GPU, float_type=Float32)

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

# Write temperature field to disk every 10 time steps.
output_writer = NetCDFOutputWriter(dir=".", prefix="convection", frequency=200, async=true)
push!(model.output_writers, output_writer)

# Time stepping
for i = 1:Nt
    tic = time_ns()
    print("Time: $(model.clock.time) ")
    time_step!(model, 1, Δt)
    println(prettytime(time_ns() - tic))
end
