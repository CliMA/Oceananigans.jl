using Statistics, Printf

using Oceananigans
using Oceananigans.Diagnostics

#####
##### Forced flow analytic solution (§6.1, Brown et al., 2001)
#####

@inline ωₐ(t) = 1 + sin(2π*t^2)
@inline ω′ₐ(t) = 4π*t * cos(2π*t^2)

@inline uₐ(x, z, t) =              cos(2π * (x - ωₐ(t))) * (3(z+1)^2 - 2(z+1))
@inline wₐ(x, z, t) =         2π * sin(2π * (x - ωₐ(t))) * (z+1)^2 * ((z+1) - 1)
@inline pₐ(x, z, t) = -ω′ₐ(t)/2π * sin(2π * (x - ωₐ(t))) * (sin(2π*z) - 2π*z + π) +
                             - ν * cos(2π * (x - ωₐ(t))) * (-2sin(2π*z) + 2π*z - π)

#####
##### Boundary conditions
#####

@inline u_top(i, j, grid, t, args...) = @inbounds uₐ(grid.xF[i], 0, t)

ubcs = HorizontallyPeriodicBCs(   top = BoundaryCondition(Value, u_top),
                               bottom = BoundaryCondition(Value, 0))

wbcs = HorizontallyPeriodicBCs(   top = BoundaryCondition(Value, 0),
                               bottom = BoundaryCondition(Value, 0))

#####
##### Construct model
#####

Nx, Ny, Nz = 192, 1, 192
Lx, Ly, Lz = 1, 1, 1

const ν = 1  # Also implicitly setting Re = 1.

model = Model(
           architecture = CPU(),
             float_type = Float64,
                   grid = RegularCartesianGrid(size=(Nx, Ny, Nz), length=(Lx, Ly, Lz)),
                closure = ConstantIsotropicDiffusivity(ν=ν),
                tracers = nothing,
               coriolis = nothing,
               buoyancy = nothing,
    boundary_conditions = BoundaryConditions(u=ubcs, w=wbcs)
)

#####
##### Setting initial conditions
#####

u₀(x, y, z) = uₐ(x, z, 0)
w₀(x, y, z) = wₐ(x, z, 0)
set!(model; u=u₀, w=w₀)

#####
##### Time step!
#####

# Compute time step Δt from explicit diffusive stability criterion.
Δ = min(model.grid.Δx, model.grid.Δz)
Δt = 0.1 * (Δ^2 / ν)

# CFL diagnostics
 cfl = AdvectiveCFL(Δt)
dcfl = DiffusiveCFL(Δt)

# Useful aliases
u, v, w = model.velocities
xC, zC = reshape(model.grid.xC, (Nx, 1, 1)), reshape(model.grid.zC, (1, 1, Nz))
xF, zF = reshape(model.grid.xF[1:end-1], (Nx, 1, 1)), reshape(model.grid.zF[1:end-1], (1, 1, Nz))

end_time = 1/2

while model.clock.time < end_time
    i, t = model.clock.iteration, model.clock.time

    walltime = @elapsed time_step!(model; Δt=Δt, Nt=1, init_with_euler = i == 0 ? true : false)

    u_model, w_model = interior(u), interior(w)

    u_max = maximum(abs, u_model)
    w_max = maximum(abs, w_model)

    u_analytic, w_analytic = uₐ.(xF, zC, t), wₐ.(xC, zF, t)

    u_err    = abs.(u_model .- u_analytic)
    u_L¹_err = mean(u_err)
    u_L∞_err = maximum(u_err)

    w_err    = abs.(w_model .- w_analytic)
    w_L¹_err = mean(w_err)
    w_L∞_err = maximum(w_err)

    @printf("i: %d, t: %.6f, umax: (%06.3g, %06.3g), CFL: %.4f, Δu: (L¹=%.4e, L∞=%.4e), Δw: (L¹=%.4e, L∞=%.4e), ⟨wall time⟩: %s\n",
            i, t, u_max, w_max, cfl(model), u_L¹_err, u_L∞_err, w_L¹_err, w_L∞_err, prettytime(walltime))

    any(isnan.(model.velocities.w.data.parent)) && break
end
