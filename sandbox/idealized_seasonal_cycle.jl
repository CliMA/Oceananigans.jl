using ArgParse

s = ArgParseSettings(description="Run simulations of a mixed layer over an idealized seasonal cycle.")

@add_arg_table s begin
    "--resolution", "-N"
        arg_type=Int
        required=true
        help="Number of grid points in each dimension (Nx, Ny, Nz) = (N, N, N)."
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
    "--cycles"
        arg_type=Int
        required=true
        help="Number of idealized seasonal cycles."
    "--simulation-time", "-T"
        arg_type=Float64
        required=true
        help="Simulation length in seconds."
    "--output-dir", "-d"
        arg_type=AbstractString
        required=true
        help="Base directory to save output to."
end

parsed_args = parse_args(s)
N = parsed_args["resolution"]
dTdz = parsed_args["dTdz"]
κ = parsed_args["diffusivity"]
dt = parsed_args["dt"]
c = parsed_args["cycles"]
T = parsed_args["simulation-time"]

N  = isinteger(N) ? Int(N) : N
dt = isinteger(dt) ? Int(dt) : dt
c  = isinteger(c) ? Int(c) : c
T  = isinteger(T) ? Int(T) : T

base_dir = parsed_args["output-dir"]

filename_prefix = "seasonal_cycle_N" * string(N) * "_dTdz" * string(dTdz) * "_k" * string(κ) * "_dt" * string(dt) *
                  "_c" * string(c) * "_T" * string(T)
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
@everywhere using CUDAnative
@everywhere using CuArrays
@everywhere using Printf
@everywhere using Statistics: mean

# Physical constants.
ρ₀ = 1027    # Density of seawater [kg/m³]
cₚ = 4181.3  # Specific heat capacity of seawater at constant pressure [J/(kg·K)]

# Set more simulation parameters.
Nx, Ny, Nz = N, N, N
Lx, Ly, Lz = 200, 200, 200
Δt = dt
Nt = Int(T/dt)
ν = κ  # This is also implicitly setting the Prandtl number Pr = 1.

ωy = 2π / T  # Seasonal frequency.
Φavg = dTdz * Lz^2 / (8T)
a = 1.1 * Φavg
@inline Qsurface(t) = (Φavg + a*sin(c*ωs*t)) / (ρ₀*cₚ)

@info "Φavg = $Φavg W/m²"

# Create the model.
model = Model(N=(Nx, Ny, Nz), L=(Lx, Ly, Lz), ν=ν, κ=κ, arch=GPU(), float_type=Float64)

# Set boundary conditions.
model.boundary_conditions.T.z.right = BoundaryCondition(Gradient, dTdz)

# Set initial conditions.
T_prof = 20 .+ dTdz .* model.grid.zC  # Initial temperature profile.
T_3d = repeat(reshape(T_prof, 1, 1, Nz), Nx, Ny, 1)  # Convert to a 3D array.

# Add small normally distributed random noise to the top half of the domain to
# facilitate numerical convection.
@. T_3d[:, :, 1:round(Int, Nz/2)] += 0.00001*randn()

model.tracers.T.data .= CuArray(T_3d)

# Add a NaN checker diagnostic that will check for NaN values in the vertical
# velocity and temperature fields every 1,000 time steps and abort the simulation
# if NaN values are detected.
nan_checker = NaNChecker(1000, [model.velocities.w, model.tracers.T], ["w", "T"])
push!(model.diagnostics, nan_checker)

# Add a checkpointer that saves the entire model state.
# checkpoint_freq = Int(7*spd / dt)  # Every 7 days.
# checkpointer = Checkpointer(dir=joinpath(output_dir, "checkpoints"), prefix="seasonal_cycle_",
#                             frequency=checkpoint_freq, padding=2)
# push!(model.output_writers, checkpointer)

# Write full output to disk every 6 hour.
output_freq = Int(6*3600 / dt)
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

        for j in 1:Ni
            model.boundary_conditions.T.z.left = BoundaryCondition(Flux, Qsurface(model.clock.time))
            time_step!(model, 1, Δt)
        end

        @printf("   average wall clock time per iteration: %s\n", prettytime((time_ns() - tic) / Ni))
    end
end
