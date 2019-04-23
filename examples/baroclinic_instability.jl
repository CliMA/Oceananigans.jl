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

dt, days = parsed_args["dt"], parsed_args["days"]
dt = isinteger(dt) ? Int(dt) : dt
days = isinteger(days) ? Int(days) : days

base_dir = parsed_args["output-dir"]

filename_prefix = "free_convection_N" * string(N) * "_Q" * string(Q) *
                  "_dTdz" * string(dTdz) * "_k" * string(κ) * "_dt" * string(dt) *
                  "_days" * string(days)
output_dir = joinpath(base_dir, filename_prefix)

if !isdir(output_dir)
    println("Creating directory: $output_dir")
    mkpath(output_dir)
end

# Adding a second worker/proccessor for asynchronous NetCDF output writing.
using Distributed
addprocs(1)

# We need to activate the Oceananigans environment on all workers if executing
# this script from the development repository using "julia --project".
@everywhere using Pkg
@everywhere Pkg.activate(".");

@everywhere using Oceananigans
@everywhere using CuArrays
@everywhere using Printf

# Physical constants.
ρ₀ = 1027    # Density of seawater [kg/m³]
cₚ = 4181.3  # Specific heat capacity of seawater at constant pressure [J/(kg·K)]

# Set more simulation parameters.
Nx, Ny, Nz = N, N, N
Lx, Ly, Lz = 100, 100, 100
Δt = dt
Nt = Int(days*60*60*24/dt)
ν = κ  # This is also implicitly setting the Prandtl number Pr = 1.

# To impose a flux boundary condition, the top flux imposed should be negative
# for a heating flux and positive for a cooling flux, thus the minus sign.

top_flux = (-Q + Δ*cos(2π*xC/Lx)) / (ρ₀ * cₚ)
top_flux = repeat(top_flux, 1, Ny)

# Create the model.
model = Model(N=(Nx, Ny, Nz), L=(Lx, Ly, Lz), ν=ν, κ=κ, arch=GPU(), float_type=Float64)

# Set boundary conditions.
model.boundary_conditions.T.z.left = BoundaryCondition(Flux, top_flux)
model.boundary_conditions.T.z.right = BoundaryCondition(Gradient, dTdz)

# Set initial conditions.
T_prof = 273.15 .+ 20 .+ dTdz .* model.grid.zC  # Initial temperature profile.
T_3d = repeat(reshape(T_prof, 1, 1, Nz), Nx, Ny, 1)  # Convert to a 3D array.

# Add small normally distributed random noise to the top half of the domain to
# facilitate numerical convection.
@. T_3d[:, :, 1:round(Int, Nz/2)] += 0.01*randn()

model.tracers.T.data .= CuArray(T_3d)

# Add a NaN checker diagnostic that will check for NaN values in the vertical
# velocity and temperature fields every 1,000 time steps and abort the simulation
# if NaN values are detected.
nan_checker = NaNChecker(1000, [model.velocities.w, model.tracers.T], ["w", "T"])
push!(model.diagnostics, nan_checker)

# Write full output to disk enough times that we end up with 32 outputs.
output_freq = Int(3600/dt)
netcdf_writer = NetCDFOutputWriter(dir=output_dir, prefix=filename_prefix * "_",
                                   padding=0, naming_scheme=:file_number,
                                   frequency=output_freq, async=true)
push!(model.output_writers, netcdf_writer)

# With asynchronous output writing we need to wrap our time-stepping using
# an @sync block so that Julia does not just quit if the model finishes time
# stepping while there are still output files that need to be written to disk
# on another worker/processor.
@sync begin
    # Take Ni "intermediate" time steps at a time and print out the current time
    # and average wall clock time per time step.
    Ni = 100
    for i = 1:ceil(Int, Nt/Ni)
        progress = 100 * (model.clock.iteration / Nt)  # Progress %
        @printf("[%06.2f%%] Time: %.1f / %.1f...", progress, model.clock.time, Nt*Δt)

        tic = time_ns()
        time_step!(model, Ni, Δt)

        @printf("   average wall clock time per iteration: %s\n", prettytime((time_ns() - tic) / Ni))
    end
end
