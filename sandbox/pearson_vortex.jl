# Pearson vortext test
# See p. 310 of "Nodal Discontinuous Galerkin Methods: Algorithms, Analysis, and Application" by Hesthaven & Warburton.

using Printf, Statistics
using Oceananigans

FT = Float64
Nx, Ny, Nz = 64, 64, 8
Lx, Ly, Lz = 1, 1, 1
end_time = 10
const ν = 1

# Choose a very small time step ~O(1/Δx²) as we are diffusion-limited in this test.
Δt = 1 / (10*π*Nx^2)

# Pearson vortex analytic solution.
@inline u(x, y, z, t) = -sin(2π*y) * exp(-4π^2 * ν * t)
@inline v(x, y, z, t) =  sin(2π*x) * exp(-4π^2 * ν * t)

@inline u_top(i, j, grid, t, args...) = u(grid.xF[i], grid.yC[j], 0, t)
@inline v_top(i, j, grid, t, args...) = v(grid.xC[i], grid.yF[j], 0, t)
@inline u_bot(i, j, grid, t, args...) = u(grid.xF[i], grid.yC[j], model.grid.Lz, t)
@inline v_bot(i, j, grid, t, args...) = v(grid.xC[i], grid.yF[j], model.grid.Lz, t)

ubcs = HorizontallyPeriodicBCs(   top = BoundaryCondition(Value, u_top),
                               bottom = BoundaryCondition(Value, u_bot))

vbcs = HorizontallyPeriodicBCs(   top = BoundaryCondition(Value, v_top),
                               bottom = BoundaryCondition(Value, v_bot))

model = Model(N=(Nx, Ny, Nz), L=(Lx, Ly, Lz),
            constants = PlanetaryConstants(f=0, g=0),
              closure = ConstantIsotropicDiffusivity(FT; ν = 1, κ = 0),
                  eos = LinearEquationOfState(βT=0, βS=0),
                  bcs = BoundaryConditions(u=ubcs, v=vbcs))

u₀(x, y, z) = u(x, y, z, 0)
v₀(x, y, z) = v(x, y, z, 0)
T₀(x, y, z) = 0
S₀(x, y, z) = 0

set_ic!(model; u=u₀, v=v₀, T=T₀, S=S₀)

xC, yC, zC = reshape(model.grid.xC, (Nx, 1, 1)), reshape(model.grid.yC, (1, Ny, 1)), reshape(model.grid.zC, (1, 1, Nz))
xF, yF = reshape(model.grid.xF[1:end-1], (Nx, 1, 1)), reshape(model.grid.yF[1:end-1], (1, Ny, 1))

for i = 1:100
    time_step!(model, 1, Δt)

    t = model.clock.time
    i = model.clock.iteration

    # Calculate relative error between model and analytic solutions for u and v.
    u_rel_err = abs.((data(model.velocities.u) .- u.(xF, yC, zC, t)) ./ u.(xF, yC, zC, t))
    u_rel_err_avg = mean(u_rel_err)
    u_rel_err_max = maximum(u_rel_err)

    v_rel_err = abs.((data(model.velocities.v) .- v.(xC, yF, zC, t)) ./ v.(xC, yF, zC, t))
    v_rel_err_avg = mean(v_rel_err)
    v_rel_err_max = maximum(v_rel_err)

    @printf("i: %d, t: %8.6f, Δu: (avg=%6.3g, max=%6.3g), Δv: (avg=%6.3g, max=%6.3g)\n",
            i, t, u_rel_err_avg, u_rel_err_max, v_rel_err_avg, v_rel_err_max)
end
