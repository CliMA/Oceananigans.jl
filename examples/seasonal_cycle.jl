using ArgParse

s = ArgParseSettings(description="Run simulations of a mixed layer over diurnal and seasonal cycles.")

@add_arg_table s begin
    "--resolution", "-N"
        arg_type=Int
        required=true
        help="Number of grid points in each dimension (Nx, Ny, Nz) = (N, N, N)."
    "--insolation", "-S"
        arg_type=Float64
        required=false
        default=1400
        help="Maximum solar insolation in the summer (at noon) [W/m²]. 1400 by default. The winter maximum is half this value (700 by default)."
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
S = parsed_args["insolation"]
Q = parsed_args["heat-flux"]
dTdz = parsed_args["dTdz"]
κ = parsed_args["diffusivity"]
dt = parsed_args["dt"]
days = parsed_args["days"]

N = isinteger(N) ? Int(N) : N
S = isinteger(S) ? Int(S) : S
Q = isinteger(Q) ? Int(Q) : Q

dt = isinteger(dt) ? Int(dt) : dt
days = isinteger(days) ? Int(days) : days

base_dir = parsed_args["output-dir"]

filename_prefix = "wind_stress_N" * string(N) * "_tau" * string(τ) * "_Q" * string(Q) *
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
@everywhere using Statistics: mean

# Physical constants.
ρ₀ = 1027    # Density of seawater [kg/m³]
cₚ = 4181.3  # Specific heat capacity of seawater at constant pressure [J/(kg·K)]

# Defining functions for the surface forcing over the seasonal and diurnal cycles.
C₁ = S     # Noon amplitude of insolation [W/m²].
C₂ = C₁/2  # Noon amplitude of winter [W/m²].
C₃ = -144  # Long wave loss number? Unused for now.
C₄ = -117  # Long wave loss winter? Unused for now.
C₅ = 0     # Below horizon?
C₆ = -0.4  # Below horizon winter?

d = 365    # Number of days in a year.
s = 86400  # Number of seconds per day.

@inline  f(t) = cos(π*t / (s*d/2))  # ??
@inline qa(t) = (C₁+C₂)/2 - (C₁-C₂)/2 * f(t)  # Annual cycle
@inline qo(t) = (C₅+C₆)/2 - (C₅-C₆)/2 * f(t)
@inline  h(t) = (1 - qo(t)) * cos(2π*(t/s - 0.5/s)) + qo(t)
@inline qd(t) = max(0, h(t))  # Daily cycle.

@inline Qsurface(t) = qa(t) * qd(t) / (ρ₀*cₚ)

# Set more simulation parameters.
Nx, Ny, Nz = N, N, N
Lx, Ly, Lz = 100, 100, 100
Δt = dt
Nt = Int(days*60*60*24/dt)
ν = κ  # This is also implicitly setting the Prandtl number Pr = 1.

# Create the model.
model = Model(N=(Nx, Ny, Nz), L=(Lx, Ly, Lz), ν=ν, κ=κ, arch=GPU(), float_type=Float64)

# Interior forcing
F₁ = 0.62
F₂ = 1-F₁
δ₁ = 0.6
δ₂ = 20

@inline radiative_factor(z) = F₁*exp(z/δ₁) + F₂*exp(z/δ₂)
@inline ddz_radiative_factor(z) = (F₁/δ₁)*exp(z/δ₁) + (F₂/δ₂)*exp(z/δ₂)

α = model.eos.αᵥ
g = model.constants.g
N² = α*g*dTdz

Z = sum(ddz_radiative_factor.(zC)) * Δz
β = -mean(Qsurface.(ts)) + (κ*N²/(α*g))

@inline Qinterior(z) = (β/Z) * ddz_radiative_factor(z)

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

# Add a checkpointer that saves the entire model state
checkpointer = Checkpointer(dir=".", prefix="test_", frequency=5, padding=1)
push!(checkpointed_model.output_writers, checkpointer)

# Write full output to disk every 1 hour.
n_outputs = 32
output_freq = floor(Int, Nt / n_outputs)
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

        # model.boundary_conditions.T.z.left = BoundaryCondition(Flux, heat_flux)

        tic = time_ns()
        time_step!(model, Ni, Δt)

        @printf("   average wall clock time per iteration: %s\n", prettytime((time_ns() - tic) / Ni))
    end
end
