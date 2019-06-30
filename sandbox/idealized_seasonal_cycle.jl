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
c = parsed_args["cycles"]
T = parsed_args["simulation-time"]

N  = isinteger(N) ? Int(N) : N
c  = isinteger(c) ? Int(c) : c
T  = isinteger(T) ? Int(T) : T

base_dir = parsed_args["output-dir"]

filename_prefix = "idealized_seasonal_cycle_N" * string(N) * "_dTdz" * string(dTdz) *
                  "_k" * string(κ) * "_c" * string(c) * "_T" * string(T)
output_dir = joinpath(base_dir, filename_prefix)

if !isdir(output_dir)
    println("Creating directory: $output_dir")
    mkpath(output_dir)
end

"""
    add_sponge_layer!(model; damping_timescale)

Adds a sponge layer to the bottom layer of the `model`. The purpose of this
sponge layer is to effectively dampen out waves reaching the bottom of the
domain and avoid having waves being continuously generated and reflecting from
the bottom, some of which may grow unrealistically large in amplitude.

Numerically the sponge layer acts as an extra source term in the momentum
equations. It takes on the form Gu[i, j, k] += -u[i, j, k]/τ for each momentum
source term where τ is a damping timescale. Typially, Δt << τ.
"""
function add_sponge_layer!(model; damping_timescale)
    τ = damping_timescale

    @inline damping_u(grid, u, v, w, T, S, i, j, k) = ifelse(k == grid.Nz, -u[i, j, k] / τ, 0)
    @inline damping_v(grid, u, v, w, T, S, i, j, k) = ifelse(k == grid.Nz, -v[i, j, k] / τ, 0)
    @inline damping_w(grid, u, v, w, T, S, i, j, k) = ifelse(k == grid.Nz, -w[i, j, k] / τ, 0)

    model.forcing = Forcing(damping_u, damping_v, wave_damping_w, nothing, nothing)
end

using Statistics, Printf
using CUDAnative, CuArrays, JLD2, FileIO
using Oceananigans

# Physical constants.
ρ₀ = 1027    # Density of seawater [kg/m³]
cₚ = 4181.3  # Specific heat capacity of seawater at constant pressure [J/(kg·K)]

# Set more simulation parameters.
Nx, Ny, Nz = N, N, N
Lx, Ly, Lz = 200, 200, 200
ν = κ  # This is also implicitly setting the Prandtl number Pr = 1.

ωs = 2π / T  # Seasonal frequency.
Φavg = dTdz * Lz^2 / (8T)
a = 1.5 * Φavg
@inline Qsurface(t) = (Φavg + a*sin(c*ωs*t))

@info @sprintf("Φavg = %.3f W/m², Φmax = %.3f W/m², Φmin = %.3f W/m²",
               Φavg*ρ₀*cₚ, (Φavg+a)*ρ₀*cₚ, (Φavg-a)*ρ₀*cₚ)

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

ardata_view(model.tracers.T) .= CuArray(T_3d)

# Add a NaN checker diagnostic that will check for NaN values in the vertical
# velocity field every 1,000 time steps and abort the simulation if NaN values
# are detected.
nan_checker = NaNChecker(1000, [model.velocities.w], ["w"])
push!(model.diagnostics, nan_checker)

# Add a checkpointer that saves the entire model state.
# checkpoint_freq = Int(7*spd / dt)  # Every 7 days.
# checkpointer = Checkpointer(dir=joinpath(output_dir, "checkpoints"), prefix="seasonal_cycle_",
#                             frequency=checkpoint_freq, padding=2)
# push!(model.output_writers, checkpointer)


Δt_wizard = TimeStepWizard(cfl=0.1, Δt=10.0, max_change=1.2, max_Δt=60.0)

# Take Ni "intermediate" time steps at a time before printing a progress
# statement and updating the time step.
Ni = 50

# Write output to disk every No time steps.
No = 1000

while model.clock.time < T
    tic = time_ns()
    for j in 1:Ni
        model.boundary_conditions.T.z.left = BoundaryCondition(Flux, Qsurface(model.clock.time))
        time_step!(model, 1, Δt_wizard.Δt)
    end
    toc = time_ns()

    progress = 100 * (model.clock.time / T)

    umax = maximum(abs, model.velocities.u.data.parent)
    vmax = maximum(abs, model.velocities.v.data.parent)
    wmax = maximum(abs, model.velocities.w.data.parent)
    CFL = Δt_wizard.Δt / cell_advection_timescale(model)

    update_Δt!(Δt_wizard, model)

    @printf("[%06.2f%%] i: %d, t: %.3f days, umax: (%6.3g, %6.3g, %6.3g) m/s, CFL: %6.4g, next Δt: %3.2f s, ⟨wall time⟩: %s",
            progress, model.clock.iteration, model.clock.time / 86400,
            umax, vmax, wmax, CFL, Δt_wizard.Δt, prettytime((toc-tic) / Ni))

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
     @printf("\n")
end
