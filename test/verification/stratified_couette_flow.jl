using Statistics, Printf
using Oceananigans

""" Friction velocity squared. See equation (16) of Vreugdenhil & Taylor (2018). """
function uτ²(model)
    Δz = model.grid.Δz
    ν = model.closure.ν

    U = mean(Array(model.velocities.u.data.parent); dims=[1, 2])
    uτ²⁺ = ν * abs(U[2] - Uw) / Δz
    uτ²⁻ = ν * abs(-Uw - U[end]) / Δz

    uτ²⁺, uτ²⁻
end

""" Heat flux at the wall. See equation (16) of Vreugdenhil & Taylor (2018). """
function qw(model)
    Δz = model.grid.Δz
    κ = model.closure.κ

    Θ = mean(Array(model.tracers.T.data.parent); dims=[1, 2])
    qw⁺ = κ * abs(Θ[1] - Θw) / Δz
    qw⁻ = κ * abs(-Θw - Θ[end]) / Δz

    qw⁺, qw⁻
end

""" Friction temperature. See equation (16) of Vreugdenhil & Taylor (2018). """
function θτ(model)
    Δz = model.grid.Δz
    uτ²⁺, uτ²⁻ = uτ²(model)
    qw⁺, qw⁻ = qw(model)
    uτ⁺, uτ⁻ = √uτ²⁺, √uτ²⁻

    qw⁺ / uτ⁺, qw⁻ / uτ⁻
end

""" Obukov length scale (assuming a linear equation of state). See equation (17) of Vreugdenhil & Taylor (2018). """
function L_Obukov(model)
    kₘ = 0.4  # Von Kármán constant
    uτ²⁺, uτ²⁻ = uτ²(model)
    qw⁺, qw⁻ = qw(model)
    uτ⁺, uτ⁻ = √uτ²⁺, √uτ²⁻

    uτ⁺^3 / (kₘ * qw⁺), uτ⁻^3 / (kₘ * qw⁻)
end

""" Near wall viscous length scale. See line following equation (18) of Vreugdenhil & Taylor (2018). """
function δᵥ(model)
    ν = model.closure.ν
    uτ²⁺, uτ²⁻ = uτ²(model)
    ν / √uτ²⁺, ν / √uτ²⁻
end

"""
    Ratio of length scales that define when the stratified plane Couette flow is turbulent.
    See equation (18) of Vreugdenhil & Taylor (2018).
"""
function L⁺(model)
    δᵥ⁺, δᵥ⁻ = δᵥ(model)
    L_O⁺, L_O⁻ = L_Obukov(model)
    L_O⁺ / δᵥ⁺, L_O⁻ / δᵥ⁻
end

""" Friction Reynolds number. See equation (20) of Vreugdenhil & Taylor (2018). """
function Reτ(model)
    ν = model.closure.ν
    h = model.grid.Lz
    uτ²⁺, uτ²⁻ = uτ²(model)
    h * √uτ²⁺ / ν, h * √uτ²⁻ / ν
end

""" Friction Nusselt number. See equation (20) of Vreugdenhil & Taylor (2018). """
function Nu(model)
    κ = model.closure.κ
    h = model.grid.Lz
    uτ²⁺, uτ²⁻ = uτ²(model)
    qw⁺, qw⁻ = qw(model)

    (qw⁺ * h)/(κ * Θw), (qw⁻ * h)/(κ * Θw)
end

# Non-dimensional parameters chosen to reproduce run 5 from Table 1 of Vreugdenhil & Taylor (2018).
Pr = 0.7   # Prandtl number
Re = 4250  # Reynolds number
Ri = 0.04  # Richardson number
 h = 1.0   # Height
Uw = 1.0   # Wall velocity

# Computed parameters.
 ν = Uw * h / Re    # From Re = Uw h / ν
Θw = Ri * Uw^2 / h  # From Ri = L Θw / Uw²
 κ = ν / Pr         # From Pr = ν / κ

# Impose boundary conditions.
Tbcs = HorizontallyPeriodicBCs(    top = BoundaryCondition(Value,  Θw),
                                bottom = BoundaryCondition(Value, -Θw))

ubcs = HorizontallyPeriodicBCs(    top = BoundaryCondition(Value,  Uw),
                                bottom = BoundaryCondition(Value, -Uw))

vbcs = HorizontallyPeriodicBCs(    top = BoundaryCondition(Value, 0),
                                bottom = BoundaryCondition(Value, 0))

wbcs = HorizontallyPeriodicBCs(    top = BoundaryCondition(Value, 0),
                                bottom = BoundaryCondition(Value, 0))

# Non-dimensional model setup
model = Model(
         arch = HAVE_CUDA ? GPU() : CPU(),
            N = (128, 128, 1024),
            L = (4π*h, 2π*h, 2h),
      # closure = ConstantIsotropicDiffusivity(ν=ν, κ=κ),
      closure = VerstappenAnisotropicMinimumDissipation(ν=ν, κ=κ),
          eos = LinearEquationOfState(βT=1.0, βS=0.0),
    constants = PlanetaryConstants(f=0.0, g=1.0),
          bcs = BoundaryConditions(u=ubcs, v=vbcs, w=wbcs, T=Tbcs)
    )

# Add a bit of surface-concentrated noise to the initial condition
ε(z) = randn() * z/model.grid.Lz * (1 + z/model.grid.Lz)

T₀(x, y, z) = 2Θw * (1/2 + z/model.grid.Lz) * (1 + 1e-6 * ε(z))
u₀(x, y, z) = 2Uw * (1/2 + z/model.grid.Lz) * (1 + 1e-6 * ε(z)) * (1 + 0.1*sin(4π/model.grid.Lx * x))
v₀(x, y, z) = 1e-6 * ε(z)
w₀(x, y, z) = 1e-6 * ε(z)
S₀(x, y, z) = 1e-6 * ε(z)

set_ic!(model, u=u₀, v=v₀, w=w₀, T=T₀, S=S₀)

@printf(
    """
    stratified Couette flow

            N : %d, %d, %d
            L : %.3g, %.3g, %.3g
           Re : %.3f
           Ri : %.3f
           Pr : %.3f
            ν : %.3g
            κ : %.3g
           Uw : %.3f
           Θw : %.3f

    """, model.grid.Nx, model.grid.Ny, model.grid.Nz,
         model.grid.Lx, model.grid.Ly, model.grid.Lz,
         Re, Ri, Pr, ν, κ, Uw, Θw)

base_dir = "stratified_couette_flow_data"
prefix = @sprintf("stratified_couette_Re%d_Ri%.3f_Nz%d", Re, Ri, model.grid.Nz)

# Saving fields.
function init_save_parameters_and_bcs(file, model)
    file["parameters/reynolds_number"] = Re
    file["parameters/richardson_number"] = Ri
    file["parameters/prandtl_number"] = Pr
    file["parameters/viscosity"] = ν
    file["parameters/diffusivity"] = κ
    file["parameters/wall_velocity"] = Uw
    file["parameters/wall_temperature"] = Θw
end

fields = Dict(
    :u => model -> Array(model.velocities.u.data.parent),
    :v => model -> Array(model.velocities.v.data.parent),
    :w => model -> Array(model.velocities.w.data.parent),
    :T => model -> Array(model.tracers.T.data.parent),
    :kappaT => model -> Array(model.diffusivities.κₑ.T.data.parent),
    :nu => model -> Array(model.diffusivities.νₑ.data.parent))


field_writer = JLD2OutputWriter(model, fields; dir=base_dir, prefix=prefix * "_fields",
                                init=init_save_parameters_and_bcs,
                                max_filesize=25GiB, interval=10, force=true, verbose=true)
push!(model.output_writers, field_writer)

# Set up diagnostics.
push!(model.diagnostics, NaNChecker(model))

Δtₚ = 1 # Time interval for computing and saving profiles.

Up = VerticalProfile(model, model.velocities.u; interval=Δtₚ)
Vp = VerticalProfile(model, model.velocities.v; interval=Δtₚ)
Wp = VerticalProfile(model, model.velocities.w; interval=Δtₚ)
Tp = VerticalProfile(model, model.tracers.T;    interval=Δtₚ)
νp = VerticalProfile(model, model.diffusivities.νₑ; interval=Δtₚ)
κp = VerticalProfile(model, model.diffusivities.κₑ.T; interval=Δtₚ)
wT = ProductProfile(model, model.velocities.w, model.tracers.T; interval=Δtₚ)
vc = VelocityCovarianceProfiles(model; interval=Δtₚ)

append!(model.diagnostics, [Up, Vp, Wp, Tp, wT, νp, κp, vc])

profiles = Dict(
     :u => model -> Array(Up.profile),
     :v => model -> Array(Vp.profile),
     :w => model -> Array(Wp.profile),
     :T => model -> Array(Tp.profile),
    :nu => model -> Array(νp.profile),
:kappaT => model -> Array(κp.profile),
    :wT => model -> Array(wT.profile),
    :uu => model -> Array(vc.uu.profile),
    :uv => model -> Array(vc.uv.profile),
    :uw => model -> Array(vc.uw.profile),
    :vv => model -> Array(vc.vv.profile),
    :vw => model -> Array(vc.vw.profile),
    :ww => model -> Array(vc.ww.profile))

profile_writer = JLD2OutputWriter(model, profiles; dir=base_dir, prefix=prefix * "_profiles",
                                  init=init_save_parameters_and_bcs, interval=Δtₚ, force=true, verbose=true)

push!(model.output_writers, profile_writer)

scalars = Dict(
    :u_tau2 => model -> uτ²(model),
        :qw => model -> qw(model),
 :theta_tau => model -> θτ(model),
   :delta_v => model -> δᵥ(model),
    :L_plus => model -> L⁺(model),
    :Re_tau => model -> Reτ(model),
    :Nu_tau => model -> Nu(model))

scalar_writer = JLD2OutputWriter(model, scalars; dir=base_dir, prefix=prefix * "_scalars",
                                  init=init_save_parameters_and_bcs, interval=Δtₚ/2, force=true, verbose=true)
push!(model.output_writers, scalar_writer)

wizard = TimeStepWizard(cfl=0.02, Δt=0.0001, max_change=1.1, max_Δt=0.02)

function cell_diffusion_timescale(model)
    Δ = min(model.grid.Δx, model.grid.Δy, model.grid.Δz)
    max_ν = maximum(model.diffusivities.νₑ.data.parent)
    max_κ = max(Tuple(maximum(κₑ.data.parent) for κₑ in model.diffusivities.κₑ)...)
    return min(Δ^2 / max_ν, Δ^2 / max_κ)
end

# Take Ni "intermediate" time steps at a time before printing a progress
# statement and updating the time step.
Ni = 10

cfl(t) = min(0.01*t, 0.1)

end_time = 1000
while model.clock.time < end_time
    wizard.cfl = cfl(model.clock.time)

    walltime = @elapsed time_step!(model; Nt=Ni, Δt=wizard.Δt)
    progress = 100 * (model.clock.time / end_time)

    umax = maximum(abs, model.velocities.u.data.parent)
    vmax = maximum(abs, model.velocities.v.data.parent)
    wmax = maximum(abs, model.velocities.w.data.parent)
    CFL = wizard.Δt / cell_advection_timescale(model)

    νmax = maximum(model.diffusivities.νₑ.data.parent)
    κmax = maximum(model.diffusivities.κₑ.T.data.parent)
    dCFL = wizard.Δt / cell_diffusion_timescale(model)

    update_Δt!(wizard, model)

    @printf("[%06.2f%%] i: %d, t: %8.5g, umax: (%6.3g, %6.3g, %6.3g), CFL: %6.4g, νκmax: (%6.3g, %6.3g), dCFL: %6.4g, next Δt: %8.5g, ⟨wall time⟩: %s\n",
            progress, model.clock.iteration, model.clock.time,
            umax, vmax, wmax, CFL, κmax, νmax, dCFL, wizard.Δt, prettytime(walltime / Ni))
end
