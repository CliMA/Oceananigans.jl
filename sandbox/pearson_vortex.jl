# Pearson vortext test
# See p. 310 of "Nodal Discontinuous Galerkin Methods: Algorithms, Analysis, and Application" by Hesthaven & Warburton.

using Printf, Statistics
using Oceananigans

FT = Float64
Nx, Ny, Nz = 64, 8, 64
Lx, Ly, Lz = 1, 1, 1
end_time = 10
Δt = 1 / (π*Nx^2)

const ν = 1

ubcs = HorizontallyPeriodicBCs(   top = BoundaryCondition(Value, 0),
                               bottom = BoundaryCondition(Value, 0))

wbcs = HorizontallyPeriodicBCs(   top = BoundaryCondition(Gradient, 0),
                               bottom = BoundaryCondition(Gradient, 0))

model = Model(N=(Nx, Ny, Nz), L=(Lx, Ly, Lz),
            constants = PlanetaryConstants(f=0, g=0),
              closure = ConstantIsotropicDiffusivity(FT; ν = 1, κ = 0),
                  eos = LinearEquationOfState(βT=0, βS=0),
                  bcs = BoundaryConditions(u=ubcs, w=wbcs))

u(x, y, z, t) = -sin(2π*z) * exp(-4π^2 * ν * t)
w(x, y, z, t) =  sin(2π*x) * exp(-4π^2 * ν * t)

u₀(x, y, z) = u(x, y, z, 0)
w₀(x, y, z) = w(x, y, z, 0)
T₀(x, y, z) = 0
S₀(x, y, z) = 0

set_ic!(model; u=u₀, w=w₀, T=T₀, S=S₀)

xC, yC, zC = reshape(model.grid.xC, (Nx, 1, 1)), reshape(model.grid.yC, (1, Ny, 1)), reshape(model.grid.zC, (1, 1, Nz))
xF, zF = reshape(model.grid.xF[1:end-1], (Nx, 1, 1)), reshape(model.grid.zF[1:end-1], (1, 1, Nz))
# while model.clock.time < end_time
for i = 1:100
    time_step!(model, 1, Δt)

    t = model.clock.time
    i = model.clock.iteration
    u_rel_err = abs.((data(model.velocities.u) .- u.(xF, yC, zC, t)) ./ u.(xF, yC, zC, t))
    u_rel_err_avg = mean(u_rel_err)
    u_rel_err_max = maximum(u_rel_err)
    w_rel_err = abs.((data(model.velocities.w) .- w.(xC, yC, zF, t)) ./ w.(xC, yC, zF, t))
    w_rel_err_avg = mean(u_rel_err)
    w_rel_err_max = maximum(u_rel_err)
    @printf("i: %d, t: %3.3f, Δu: (avg=%6.3g, max=%6.3g), Δw: (avg=%6.3g, max=%6.3g)\n",
            i, t, u_rel_err_avg, u_rel_err_max, w_rel_err_avg, w_rel_err_max)
end
