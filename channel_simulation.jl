using Oceananigans
using Oceananigans.Units
using Oceananigans.BoundaryConditions: PerturbationAdvectionOpenBoundaryCondition
using Oceananigans.ImmersedBoundaries: mask_immersed_field!
using Oceananigans.Fields: @compute
using Oceananigans.Solvers: ConjugateGradientPoissonSolver, fft_poisson_solver, DiagonallyDominantPreconditioner

using CUDA: @allowscalar
using Statistics: mean

H, L = 256meters, 1024meters
δ = L / 2
x, y, z = (-3L, 3L), (-L, L), (-H, 0)
Nz = 16

underlying_grid = RectilinearGrid(CPU(); size=(6Nz, 2Nz, Nz), halo=(6, 6, 6),
                                  x, y, z, topology=(Bounded, Bounded, Bounded))

bowl(y) = -H * (1 - (y / L)^2)
bowl(x, y) = bowl(y)
wedge(x, y) = -H * (1 + (y + abs(x)) / δ)
bowl_wedge(x, y) = max(bowl(y), wedge(x, y))

flat_bottom(x, y) = -3H/4

grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bowl_wedge))

T₂ = 12.421hours
U₂ = 0.1 # m/s

@inline U(x, y, z, t, p) = p.U₂ * sin(2π * t / p.T₂)
@inline U(y, z, t, p) = U(zero(y), y, z, t, p)

open_bc_pad = PerturbationAdvectionOpenBoundaryCondition(U; inflow_timescale = 2minutes,
                                                         outflow_timescale = 2minutes,
                                                         parameters=(; U₂, T₂))

u_bcs = FieldBoundaryConditions(east = open_bc_pad, west = open_bc_pad)

@inline ambient_temperature(x, z, t, H) = 12 + 4z/H
@inline ambient_temperature(x, y, z, t, H) = ambient_temperature(x, z, t, H)
ambient_temperature_bc = ValueBoundaryCondition(ambient_temperature; parameters = H)
T_bcs = FieldBoundaryConditions(east = ambient_temperature_bc, west = ambient_temperature_bc)

ambient_salinity_bc = ValueBoundaryCondition(32)
S_bcs = FieldBoundaryConditions(east = ambient_salinity_bc, west = ambient_salinity_bc)

model = NonhydrostaticModel(; grid, tracers = (:T, :S),
                              buoyancy = SeawaterBuoyancy(),
                              advection = WENO(order=5), coriolis = FPlane(latitude=47.5),
                              #pressure_solver = ConjugateGradientPoissonSolver(grid, maxiter=15, preconditioner=nothing),
                              #pressure_solver = ConjugateGradientPoissonSolver(grid, maxiter=15, preconditioner=DiagonallyDominantPreconditioner()),
                              pressure_solver = ConjugateGradientPoissonSolver(grid, maxiter=15, preconditioner=fft_poisson_solver(grid.underlying_grid)),
                              boundary_conditions = (; T=T_bcs, u = u_bcs, S = S_bcs))

Tᵢ(x, y, z) = ambient_temperature(x, y, z, 0, H)

set!(model, T=Tᵢ, S=32, u=U(0, 0, 0, 0, (; U₂, T₂)))

simulation = Simulation(model, Δt=5, stop_time=0.5day)
conjure_time_step_wizard!(simulation, cfl=0.1)

using Printf

wallclock = Ref(time_ns())
T₀ = time_ns()

function progress(sim)
    u, v, w = sim.model.velocities
    ΔT = 1e-9 * (time_ns() - wallclock[])
    ΔT₀ = 1e-9 * (time_ns() - T₀)
    ps = sim.model.pressure_solver
    if ps isa ConjugateGradientPoissonSolver
        msg = @sprintf("(%d) t: %s, Δt: %s, wall time since last print: %s, total wall time : %s
            solver iterations: %d, mean solver residual: %e\n",
                       iteration(sim), prettytime(sim), prettytime(sim.Δt), prettytime(ΔT), prettytime(ΔT₀),
                       ps.conjugate_gradient_solver.iteration, mean(ps.conjugate_gradient_solver.residual))
    else
        msg = @sprintf("(%d) t: %s, Δt: %s, wall time since last print: %s, total wall time : %s",
                       iteration(sim), prettytime(sim), prettytime(sim.Δt), prettytime(ΔT), prettytime(ΔT₀))
    end
    @info msg
    wallclock[] = time_ns()
    return nothing
end

add_callback!(simulation, progress, IterationInterval(10))
#add_callback!(simulation, Oceananigans.Models.NaNChecker(fields=(; model.velocities.u)), IterationInterval(2))

prefix = "channel_$Nz"
u, v, w = model.velocities
ζ = ∂x(v) - ∂y(u)

#+++ Plotting
output_interval_2d = 10minutes
animations_dir = "."

u, v, w = model.velocities
T, S = model.tracers

@compute u_sfc = Field(Field(u + 0), indices=(:, :, grid.Nz))
@compute v_sfc = Field(Field(v + 0), indices=(:, :, grid.Nz))
@compute w_sfc = Field(Field(w + 0), indices=(:, :, grid.Nz))
@compute T_sfc = Field(Field(T + 0), indices=(:, :, grid.Nz))
@compute ω_sfc = Field(Field(∂x(v) - ∂y(u)), indices=(:, :, grid.Nz))

@compute u_ver = Field(Field(u + 0), indices=(:, grid.Ny÷2, :))
@compute v_ver = Field(Field(v + 0), indices=(:, grid.Ny÷2, :))
@compute w_ver = Field(Field(w + 0), indices=(:, grid.Ny÷2, :))
@compute T_ver = Field(Field(T + 0), indices=(:, grid.Ny÷2, :))
@compute ω_ver = Field(Field(∂x(v) - ∂y(u)), indices=(:, grid.Ny÷2, :))

using CairoMakie
fig = Figure(size = (2220, 1080));

xy_axis_kwargs = (
    xlabel = "easting (m)",
    ylabel = "northing (m)",
    xgridvisible = false,
    ygridvisible = false,
    height = 700,
)

xz_axis_kwargs = (
    xlabel = "easting (m)",
    ylabel = "northing (m)",
    xgridvisible = false,
    ygridvisible = false,
    height = 200,
)

ax_1 = Axis(fig[1, 1]; title = "u velocity", xy_axis_kwargs...)
ax_2 = Axis(fig[1, 3]; title = "w velocity", xy_axis_kwargs...)
ax_3 = Axis(fig[1, 5]; title = "temperature", xy_axis_kwargs...)

ax_5 = Axis(fig[2, 1]; title = "u velocity", xz_axis_kwargs...)
ax_5 = Axis(fig[2, 3]; title = "w velocity", xz_axis_kwargs...)
ax_6 = Axis(fig[2, 5]; title = "Temperature", xz_axis_kwargs...)

u_scale = 0.5
w_scale = u_scale / 10
ω_scale = u_scale / mean(xspacings(grid))

function update_plot!(sim)
    mask_immersed_field!(u_sfc, NaN)
    mask_immersed_field!(ω_sfc, NaN)
    mask_immersed_field!(w_sfc, NaN)
    mask_immersed_field!(T_sfc, NaN)

    hm_u = heatmap!(fig[1, 1], u_sfc, colorrange=(-u_scale, u_scale), colormap=:balance, nan_color=:gray)
    hm_w = heatmap!(fig[1, 3], w_sfc, colorrange=(-w_scale, w_scale), colormap=:balance, nan_color=:gray)
    hm_T = heatmap!(fig[1, 5], T_sfc, colorrange=(8, 12), colormap=:lajolla, nan_color=:gray)

    @allowscalar hm_u2 = heatmap!(fig[2, 1], u_ver, colorrange=(-u_scale, u_scale), colormap=:balance, nan_color=:gray)
    @allowscalar hm_w2 = heatmap!(fig[2, 3], w_ver, colorrange=(-w_scale, w_scale), colormap=:balance, nan_color=:gray)
    @allowscalar hm_T2 = heatmap!(fig[2, 5], T_ver, colorrange=(8, 12), colormap=:lajolla, nan_color=:gray)

    if simulation.model.clock.iteration == 0
        Colorbar(fig[1, 2], hm_u, label="m/s")
        Colorbar(fig[1, 4], hm_w, label="m/s")
        Colorbar(fig[1, 6], hm_T, label="°C")

        Colorbar(fig[2, 2], hm_u2, label="m/s")
        Colorbar(fig[2, 4], hm_w2, label="m/s")
        Colorbar(fig[2, 6], hm_T2, label="°C")
    end

    recordframe!(io)
    resize_to_layout!(fig) # Resize figure after everything is done to it, but before recording
end

io = VideoStream(fig, format="mp4", framerate=12, compression=20)
update_plot!(simulation)
add_callback!(simulation, update_plot!, TimeInterval(output_interval_2d))
#---

run!(simulation)

surface_animation_path = joinpath(animations_dir, "$prefix.mp4")
save(surface_animation_path, io)
