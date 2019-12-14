using Statistics, Printf
using Oceananigans

const ν = 1  # Also implicitly setting Re = 1.

@inline ω(t) = 1 + sin(2π*t^2)
@inline ω′(t) = 4π*t * cos(2π*t^2)

@inline u(x, z, t) =             cos(2π * (x - ω(t))) * (3(z+1)^2 - 2(z+1))
@inline w(x, z, t) =        2π * sin(2π * (x - ω(t))) * (z+1)^2 * ((z+1) - 1)
@inline p(x, z, t) = -ω′(t)/2π * sin(2π * (x - ω(t))) * (sin(2π*z) - 2π*z + π) +
                           - ν * cos(2π * (x - ω(t))) * (-2sin(2π*z) + 2π*z - π)

@inline u_top(i, j, grid, t, args...) = u(grid.xF[i], 0, t)

ubcs = HorizontallyPeriodicBCs(   top = BoundaryCondition(Value, u_top),
                               bottom = BoundaryCondition(Value, 0))

wbcs = HorizontallyPeriodicBCs(   top = BoundaryCondition(Value, 0),
                               bottom = BoundaryCondition(Value, 0))

Nx, Ny, Nz = 192, 1, 192
Lx, Ly, Lz = 1, 1, 1
arch = CPU()

model = Model(N = (Nx, Ny, Nz), L = (Lx, Ly, Lz), arch = arch,
              constants = PlanetaryConstants(f=0, g=0),  # Turn off rotation and gravity.
                closure = ConstantIsotropicDiffusivity(ν = ν, κ = 0),  # Turn off tracer diffusivity.
                    eos = NoEquationOfState(),  # Turn off buoyancy.
                    bcs = BoundaryConditions(u=ubcs, w=wbcs))

u₀(x, y, z) = u(x, z, 0)
w₀(x, y, z) = w(x, z, 0)
set!(model; u=u₀, w=w₀)

xC, zC = reshape(model.grid.xC, (Nx, 1, 1)), reshape(model.grid.zC, (1, 1, Nz))
xF, zF = reshape(model.grid.xF[1:end-1], (Nx, 1, 1)), reshape(model.grid.zF[1:end-1], (1, 1, Nz))

Δt = 1e-6

Δ = min(model.grid.Δx, model.grid.Δz)
Δt_max = 1/2 *  (1/2)^2 * (Δ^2 / ν)
@printf("Diffusive CFL: Δt = %.4e, Δt_max = %.4e\n", Δt, Δt_max)

end_time = 1/2
time_step!(model; Δt=Δt, Nt=1)
while model.clock.time < end_time
    walltime = @elapsed time_step!(model; Δt=Δt, Nt=1, init_with_euler=false)

    t = model.clock.time
    i = model.clock.iteration

    umax = maximum(abs, model.velocities.u.data.parent)
    wmax = maximum(abs, model.velocities.w.data.parent)
    CFL = Δt / cell_advection_timescale(model)

    u_model, w_model = data(model.velocities.u), data(model.velocities.w)
    u_analytic, w_analytic = u.(xF, zC, t), w.(xC, zF, t)

    u_err    = abs.(u_model .- u_analytic)
    u_L¹_err = mean(u_err)
    u_L∞_err = maximum(u_err)
    
    w_err    = abs.(w_model .- w_analytic)
    w_L¹_err = mean(w_err)
    w_L∞_err = maximum(w_err)

    @printf(
        "i: %d, t: %.6f, umax: (%06.3g, %06.3g), CFL: %.4f, Δu: (L¹=%.4e, L∞=%.4e), Δw: (L¹=%.4e, L∞=%.4e), ⟨wall time⟩: %s\n",
        model.clock.iteration, model.clock.time, umax, wmax, CFL, u_L¹_err, u_L∞_err, w_L¹_err, w_L∞_err, prettytime(walltime))

    any(isnan.(model.velocities.w.data.parent)) && break
end

