using ArgParse

s = ArgParseSettings(description="Run simulations of a mixed layer over diurnal and seasonal cycles.")

@add_arg_table s begin
    "--resolution", "-N"
        arg_type=Int
        required=true
        help="Number of grid points in each dimension (Nx, Ny, Nz) = (N, N, N)."
    "--cooling-factor"
        arg_type=Float64
        required=true
    "--heating-factor"
        arg_type=Float64
        required=true
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
Sc = parsed_args["cooling-factor"]
Sh = parsed_args["heating-factor"]
dTdz = parsed_args["dTdz"]
κ = parsed_args["diffusivity"]
dt = parsed_args["dt"]
days = parsed_args["days"]

N = isinteger(N) ? Int(N) : N
Sc = isinteger(Sc) ? Int(Sc) : Sc
Sh = isinteger(Sh) ? Int(Sh) : Sh

dt = isinteger(dt) ? Int(dt) : dt
days = isinteger(days) ? Int(days) : days

base_dir = parsed_args["output-dir"]

filename_prefix = "seasonal_cycle_N" * string(N) * "_Sc" * string(Sc) * "_Sh" * string(Sh) *
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
@everywhere using CUDAnative
@everywhere using CuArrays
@everywhere using Printf
@everywhere using Statistics: mean

# Physical constants.
ρ₀ = 1027    # Density of seawater [kg/m³]
cₚ = 4181.3  # Specific heat capacity of seawater at constant pressure [J/(kg·K)]

# Defining functions for the surface forcing over the seasonal and diurnal cycles.
spd = 86400  # Seconds per day.
dpy = 365    # Days per year.

ωd = 2π / spd  # Daily frequency.
ωy = ωd / dpy  # Yearly frequency.

@inline C₂(t) = -Sc * (1 + 0.5*cos(ωy*t))
@inline C₁(t) =  Sh * (1 - 0.5*cos(ωy*t)) - C₂(t)

@inline Qsurface(t) = (C₁(t) * max(0, sin(ωd*t - π/2)) + C₂(t)) / (ρ₀*cₚ)

# Set more simulation parameters.
Nx, Ny, Nz = N, N, N
Lx, Ly, Lz = 200, 200, 200
Δt = dt
Nt = Int(days*60*60*24/dt)
ν = κ  # This is also implicitly setting the Prandtl number Pr = 1.

# Create the model.
model = Model(N=(Nx, Ny, Nz), L=(Lx, Ly, Lz), ν=ν, κ=κ, arch=GPU(), float_type=Float64)

# Interior forcing
const δL = 20  # Longwave penetration length scale [m].

@inline radiative_factor(z) = exp(z/δL)
@inline ddz_radiative_factor_cpu(z) = δL*exp(z/δL)
@inline ddz_radiative_factor(z) = δL*CUDAnative.exp(z/δL)

ts = 0:dt:(spd*dpy)
const β = -mean(Qsurface.(ts)) + (κ*dTdz)
const NN = sum(ddz_radiative_factor_cpu.(model.grid.zC)) * model.grid.Δz

@inline Qinterior(grid, u, v, w, T, S, i, j, k) = (β/NN) * ddz_radiative_factor(grid.zC[k])

model.forcing = Forcing(nothing, nothing, nothing, Qinterior, nothing)

# Set boundary conditions.
model.boundary_conditions.T.z.right = BoundaryCondition(Gradient, dTdz)

# Set initial conditions.
T_prof = 273.15 .+ 20 .+ dTdz .* model.grid.zC  # Initial temperature profile.
T_3d = repeat(reshape(T_prof, 1, 1, Nz), Nx, Ny, 1)  # Convert to a 3D array.

# Add small normally distributed random noise to the top half of the domain to
# facilitate numerical convection.
@. T_3d[:, :, 1:round(Int, Nz/2)] += 0.001*randn()

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

# Write full output to disk every 1 hour.
output_freq = Int(2*3600 / dt)
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
