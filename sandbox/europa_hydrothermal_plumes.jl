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

days = parsed_args["days"]
days = isinteger(days) ? Int(days) : days

base_dir = parsed_args["output-dir"]

filename_prefix = "europa_hydrothermal_plumes_N" * string(Nh) * "_V" * string(Nz) *
                   "_L" * string(L) * "_H" * string(H) *
                  "_Q" * string(Q) * "_dTdz" * string(dTdz) * "_k" * string(κ) *
                  "_days" * string(days)
output_dir = joinpath(base_dir, filename_prefix)

if !isdir(output_dir)
    println("Creating directory: $output_dir")
    mkpath(output_dir)
end

using Printf
using CuArrays
using Oceananigans

# Physical constants.
ρ₀ = 1027    # Density of seawater [kg/m³]
cₚ = 4181.3  # Specific heat capacity of seawater at constant pressure [J/(kg·K)]

# Seconds per day on Europa.
# See: https://en.wikipedia.org/wiki/Rotation_period#Rotation_period_of_selected_objects
spd = 35430

# Set more simulation parameters.
Nx, Ny, Nz = Nh, Nh, Nz
Lx, Ly, Lz = L, L, H
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
nan_checker = NaNChecker(1000, [model.velocities.w], ["w"])
push!(model.diagnostics, nan_checker)

Δt_wizard = TimeStepWizard(cfl=0.15, Δt=10.0, max_change=1.2, max_Δt=300.0)

# Take Ni "intermediate" time steps at a time before printing a progress
# statement and updating the time step.
Ni = 50

# Write output to disk every No time steps.
No = 1000

end_time = spd * days
while model.clock.time < end_time
    walltime = @elapsed time_step!(model; Nt=Ni, Δt=Δt_wizard.Δt)

    progress = 100 * (model.clock.time / end_time)

    umax = maximum(abs, model.velocities.u.data.parent)
    vmax = maximum(abs, model.velocities.v.data.parent)
    wmax = maximum(abs, model.velocities.w.data.parent)
    CFL = Δt_wizard.Δt / cell_advection_timescale(model)

    update_Δt!(Δt_wizard, model)

    @printf("[%06.2f%%] i: %d, t: %.3f days, umax: (%6.3g, %6.3g, %6.3g) m/s, CFL: %6.4g, next Δt: %3.2f s, ⟨wall time⟩: %s",
            progress, model.clock.iteration, model.clock.time / spd,
            umax, vmax, wmax, CFL, Δt_wizard.Δt, prettytime(1e9*walltime / Ni))

    if model.clock.iteration % No == 0
        filename = filename_prefix  * "_" * string(model.clock.iteration) * ".jld2"
        io_time = @elapsed save(filename,
            Dict("t" => model.clock.time,
                 "xC" => Array(model.grid.xC),
                 "yC" => Array(model.grid.yC),
                 "zC" => Array(model.grid.zC),
                 "xF" => Array(model.grid.xF),
                 "yF" => Array(model.grid.yF),
                 "zF" => Array(model.grid.zF),
                 "u"  => Array(model.velocities.u.data.parent),
                 "v"  => Array(model.velocities.v.data.parent),
                 "w"  => Array(model.velocities.w.data.parent),
                 "T"  => Array(model.tracers.T.data.parent)))
        @printf(", IO time: %s", prettytime(1e9*io_time))
     end
end
