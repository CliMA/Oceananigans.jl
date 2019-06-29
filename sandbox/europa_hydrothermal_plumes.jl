using ArgParse

s = ArgParseSettings(description="Run simulations of a deep subsurface ocean on Europa with" *
                                 " a heating flux at the seafloor inducing hydrothermal plumes.")

@add_arg_table s begin
    "--horizontal-resolution", "-N"
        arg_type=Int
        required=true
        help="Number of grid points in the horizontal (Nx, Ny) = (N, N)."
    "--vertical-resolution", "-V"
        arg_type=Int
        required=true
        help="Number of grid points in the vertical Nz."
    "--length", "-L"
        arg_type=Float64
        required=true
        help="Horizontal size of the domain (Lx, Ly) = (L, L) [meters] ."
    "--height", "-H"
        arg_type=Float64
        required=true
        help="Vertical height (or depth) of the domain Lz [meters]."
    "--heat-flux"
        arg_type=Float64
        required=true
        help="Heat flux to impose at the bottom [W/m²]. Negative values imply a cooling flux."
    "--dTdz"
        arg_type=Float64
        required=true
        help="Temperature gradient (stratification) to impose [K/m]."
    "--diffusivity"
        arg_type=Float64
        required=true
        help="Vertical diffusivity κv [m²/s]. Horizontal diffusivity κh will be scaled by aspect ratio (L/H)."
    "--dt"
        arg_type=Float64
        required=true
        help="Time step in seconds."
    "--days"
        arg_type=Float64
        required=true
        help="Number of Europa days to run the model."
    "--output-dir", "-d"
        arg_type=AbstractString
        required=true
        help="Base directory to save output to."
end

parsed_args = parse_args(s)
Nh = parsed_args["horizontal-resolution"]
Nz = parsed_args["vertical-resolution"]
L = parsed_args["length"]
H = parsed_args["height"]
Q = parsed_args["heat-flux"]
dTdz = parsed_args["dTdz"]
κ = parsed_args["diffusivity"]

Nh = isinteger(Nh) ? Int(Nh) : Nh
Nz = isinteger(Nz) ? Int(Nz) : Nz
L = isinteger(L) ? Int(L) : L
H = isinteger(H) ? Int(H) : H
Q = isinteger(Q) ? Int(Q) : Q

dt, days = parsed_args["dt"], parsed_args["days"]
dt = isinteger(dt) ? Int(dt) : dt
days = isinteger(days) ? Int(days) : days

base_dir = parsed_args["output-dir"]

filename_prefix = "europa_hydrothermal_plumes_N" * string(Nh) * "_V" * string(Nz) *
                   "_L" * string(L) * "_H" * string(H) *
                  "_Q" * string(Q) * "_dTdz" * string(dTdz) * "_k" * string(κ) *
                  "_dt" * string(dt) * "_days" * string(days)
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

# Seconds per day on Europa.
# See: https://en.wikipedia.org/wiki/Rotation_period#Rotation_period_of_selected_objects
spd = 35430

# Set more simulation parameters.
Nx, Ny, Nz = Nh, Nh, Nz
Lx, Ly, Lz = L, L, H
Δt = dt
Nt = Int(days*spd/dt)
κv, νv = κ, κ  # This is also implicitly setting the Prandtl number Pr = 1.
κh, νh = (L/H)*κv, (L/H)*νv  # Scale horizontal κ and ν by the aspect ratio.

# Create the model.
model = Model(N=(Nx, Ny, Nz), L=(Lx, Ly, Lz), νh=νh, κh=κh, νv=νv, κv=κv,
              constants=Europa(lat=45), arch=GPU(), float_type=Float64)

# Set heating flux at the bottom.
bottom_flux = Q / (ρ₀ * cₚ)
model.boundary_conditions.T.z.right = BoundaryCondition(Flux, bottom_flux)

# Set initial conditions.
T_prof = 273.15 .+ 20 .+ dTdz .* model.grid.zC  # Initial temperature profile.
T_3d = repeat(reshape(T_prof, 1, 1, Nz), Nx, Ny, 1)  # Convert to a 3D array.

# Add small normally distributed random noise to the seafloor to
# facilitate numerical convection.
@. T_3d[:, :, Nz] += 0.001*randn()

model.tracers.T.data .= CuArray(T_3d)

# Add a NaN checker diagnostic that will check for NaN values in the vertical
# velocity and temperature fields every 1,000 time steps and abort the simulation
# if NaN values are detected.
nan_checker = NaNChecker(1000, [model.velocities.w, model.tracers.T], ["w", "T"])
push!(model.diagnostics, nan_checker)

# Write full output to disk every 2500 iterations.
output_freq = 2500
netcdf_writer = NetCDFOutputWriter(dir=output_dir, prefix=filename_prefix * "_",
                                   padding=0, naming_scheme=:file_number,
                                   frequency=output_freq, async=false)
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
