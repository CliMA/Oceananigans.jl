using Random, Printf, FileIO, Statistics
using PyPlot, JLD2
using Oceananigans

""" Friction velocity """
function uτ²(model)
    Δz = model.grid.Δz
    ν = model.closure.ν

    U = mean(data(model.velocities.u); dims=[1, 2])
    uτ²⁺ = ν * abs(U[2] - Uw) / Δz
    uτ²⁻ = ν * abs(-Uw - U[end]) / Δz

    uτ²⁺, uτ²⁻
end

""" Heat flux at the wall """
function qw(model)
    Δz = model.grid.Δz
    κ = model.closure.κ

    Θ = mean(data(model.tracers.T); dims=[1, 2])
    qw⁺ = κ * abs(Θ[1] - Θw) / Δz
    qw⁻ = κ * abs(-Θw - Θw[end]) / Δz

    qw⁺, qw⁻
end

""" Friction temperature """
function θτ(model)
    Δz = model.grid.Δz
    uτ²⁺, uτ²⁻ = uτ²(model)
    qw⁺, qw⁻ = qw(model)
    uτ⁺, uτ⁻ = √uτ²⁺, √uτ²⁻

    qw⁺ / uτ⁺, qw⁻ / uτ⁻
end

""" Obukov length scale (assuming a linear equation of state) """
function L_Obukov(model)
    kₘ = 0.4  # Von Kármán constant
    uτ²⁺, uτ²⁻ = uτ²(model)
    qw⁺, qw⁻ = qw(model)
    uτ⁺, uτ⁻ = √uτ²⁺, √uτ²⁻

    uτ⁺^3 / (kₘ * qw⁺), uτ⁻^3 / (kₘ * qw⁻)
end

""" Near wall viscous length scale """
function δᵥ(model)
    ν = model.closure.ν
    uτ²⁺, uτ²⁻ = uτ²(model)
    ν / √uτ²⁺, ν / √uτ²⁻
end

""" Ratio of length scales that define when the stratified plane Couette flow is turbulent """
function L⁺(model)
    δᵥ⁺, δᵥ⁻ = δᵥ(model)
    L_O⁺, L_O⁻ = L_Obukov(model)
    L_O⁺ / δᵥ⁺, L_O⁻ / δᵥ⁻
end

""" Friction Reynolds number """
function Reτ(model)
    ν = model.closure.ν
    h = model.grid.Lz
    uτ²⁺, uτ²⁻ = uτ²(model)
    h * √uτ²⁺ / ν, h * √uτ²⁻ / ν
end

""" Friction Nusselt number """
function Nu(model)
    κ = model.closure.κ
    h = model.grid.Lz
    uτ²⁺, uτ²⁻ = uτ²(model)
    qw⁺, qw⁻ = qw(model)

    (qw⁺ * h)/(κ * Θw), (qw⁻ * h)/(κ * Θw)
end

# Initial condition, boundary condition, and tracer forcing
Pr = 0.7
Re = 4250
Ri = 0.01
 h = 1.0
Uw = 0.1

# Computed parameters
# Re = Uw h / ν
# Ri = L Θw / Uw²
# Ri*Re² = Ra = Δb * L³ / νκ = [m⁴/s²] / [m⁴/s²]
 ν = Uw * h / Re
Θw = Ri * Uw^2 / h
 κ = ν / Pr

# Impose boundary conditions.
Tbcs = HorizontallyPeriodicBCs(    top = BoundaryCondition(Value,  Θw),
                                bottom = BoundaryCondition(Value, -Θw))

ubcs = HorizontallyPeriodicBCs(    top = BoundaryCondition(Value,  Uw),
                                bottom = BoundaryCondition(Value, -Uw))

# Non-dimensional model setup
model = Model(
         arch = HAVE_CUDA ? GPU() : CPU(),
            N = (32, 32, 32),
            L = (h, h, h),
      # closure = ConstantIsotropicDiffusivity(ν=ν, κ=κ),
      closure = AnisotropicMinimumDissipation(ν=ν, κ=κ),
          eos = LinearEquationOfState(βT=1.0, βS=0.0),
    constants = PlanetaryConstants(f=0.0, g=1.0),
          bcs = BoundaryConditions(u=ubcs, T=Tbcs)
    )

# Add a bit of surface-concentrated noise to the initial condition
ε(z) = randn() * z/model.grid.Lz * (1 + z/model.grid.Lz)

T₀(x, y, z) = 2Θw * (1/2 + z/model.grid.Lz) * (1 + 1e-4 * ε(z))
u₀(x, y, z) = 2Uw * (1/2 + z/model.grid.Lz) * (1 + 1e-4 * ε(z)) * (1 + 0.1*sin(4π/model.grid.Lx * x))
v₀(x, y, z) = 1e-4 * ε(z)
w₀(x, y, z) = 1e-4 * ε(z)
S₀(x, y, z) = 1e-4 * ε(z)

set_ic!(model, u=u₀, v=v₀, w=w₀, T=T₀, S=S₀)

@printf(
    """
    Crunching stratified Couette flow with

            N : %d, %d, %d
            L : %.3g, %.3g, %.3g
           Re : %.3f
           Ri : %.3f
           Pr : %.3f
           Uw : %.3f
           Θw : %.3f

    Let's spin the gears.

    """, model.grid.Nx, model.grid.Ny, model.grid.Nz, model.grid.Lx, model.grid.Ly, model.grid.Lz, Re, Ri, Pr, Uw, Θw)

wizard = TimeStepWizard(cfl=0.025, Δt=0.01, max_change=1.1, max_Δt=1.0)

# Take Ni "intermediate" time steps at a time before printing a progress
# statement and updating the time step.
Ni = 50

# Write output to disk every No time steps.
No = 1000

close("all")
fig, axs = subplots(nrows=1, ncols=3)

end_time = 1000
while model.clock.time < end_time
    walltime = @elapsed time_step!(model; Nt=Ni, Δt=wizard.Δt)
    progress = 100 * (model.clock.time / end_time)

    umax = maximum(abs, model.velocities.u.data.parent)
    vmax = maximum(abs, model.velocities.v.data.parent)
    wmax = maximum(abs, model.velocities.w.data.parent)
    CFL = wizard.Δt / cell_advection_timescale(model)

    update_Δt!(wizard, model)

    @printf("[%06.2f%%] i: %d, t: %8.5g, umax: (%6.3g, %6.3g, %6.3g), CFL: %6.4g, next Δt: %8.5g, ⟨wall time⟩: %s\n",
            progress, model.clock.iteration, model.clock.time,
            umax, vmax, wmax, CFL, wizard.Δt, prettytime(1e9*walltime / Ni))

    uτ²⁺, uτ²⁻ = uτ²(model)
    qw⁺, qw⁻ = qw(model)
    θτ⁺, θτ⁻ = θτ(model)
    δᵥ⁺, δᵥ⁻ = δᵥ(model)
    L⁺⁺, L⁺⁻ = L⁺(model)
    Reτ⁺, Reτ⁻ = Reτ(model)
    Nu⁺, Nu⁻ = Nu(model)

    @printf("[%06.2f%%] uτ²: (%6.3g, %6.3g),  qw: (%6.3g, %6.3g), θτ: (%6.3g, %6.3g), δᵥ: (%6.3g, %6.3g), L⁺: (%6.3g, %6.3g), Reτ: (%6.3g, %6.3g), Nu: (%6.3g, %6.3g)",
            progress, uτ²⁺, uτ²⁻, qw⁺, qw⁻, θτ⁺, θτ⁻, δᵥ⁺, δᵥ⁻, L⁺⁺, L⁺⁻, Reτ⁺, Reτ⁻, Nu⁺, Nu⁻)

    if model.clock.iteration % No == 0
        subplot(1, 3, 1); cla()
        pcolormesh(model.grid.xC/h, model.grid.zC/h, rotr90(view(data(model.tracers.T), :, 2, :)))
        xlabel("x/h")
        ylabel("z/h")

        U = reshape(mean(data(model.velocities.u); dims=[1, 2]), model.grid.Nz)
        Θ =reshape(mean(data(model.tracers.T); dims=[1, 2]), model.grid.Nz)

        subplot(1, 3, 2); cla()
        plot(U/Uw, model.grid.zC/h)
        xlabel("U/Uw")
        ylabel("z/h")

        subplot(1, 3, 3); cla()
        plot(Θ/Θw, model.grid.zC/h)
        xlabel("Θ/Θw")
        ylabel("z/h")
        show()

        fname = @sprintf("stratified_couette_Re%d_Ri%.3f_Nz%d.jld2", Re, Ri, model.grid.Nz)
        io_time = @elapsed save(fname,
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
