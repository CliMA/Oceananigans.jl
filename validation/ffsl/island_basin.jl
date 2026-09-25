# Reproduce: julia --project validation/ffsl/island_basin.jl immersed 2.5   (args: immersed|plain, target Courant number)

using Oceananigans
using Oceananigans.Units
using Oceananigans.Grids: inactive_cell
using Oceananigans.Advection: swept_region, XSweep, YSweep, ffsl_volume_flux
using Printf
using Random
Random.seed!(7)

#####
##### Prescribed flow around an island in a closed basin
#####

const N = 48
const Δ = 1 / N
const rᵢ = 0.12
island(x, y) = (x - 0.5)^2 + (y - 0.5)^2 < rᵢ^2
bottom(x, y) = island(x, y) ? 1.0 : -1.0   # the island pierces the surface

# ψ vanishes on the walls and on every corner of the island cells, with flow along the coast
const r꜀ = rᵢ + 0.75Δ
function basin_ψ(x, y, t)
    r = sqrt((x - 0.5)^2 + (y - 0.5)^2)
    return 2 * max(0, r - r꜀) * sin(π * x) * sin(π * y)
end

u(x, y, z, t) = - (basin_ψ(x, y + Δ/2, t) - basin_ψ(x, y - Δ/2, t)) / Δ
v(x, y, z, t) = + (basin_ψ(x + Δ/2, y, t) - basin_ψ(x - Δ/2, y, t)) / Δ

underlying_grid = RectilinearGrid(size = (N, N, 2), halo = (7, 7, 4), x = (0, 1), y = (0, 1), z = (-1, 0),
                                  topology = (Bounded, Bounded, Bounded))
grid = ARGS[1] == "immersed" ? ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom)) : underlying_grid

model = HydrostaticFreeSurfaceModel(grid; velocities = PrescribedVelocityFields(; u, v),
                                    tracers = (:c, :uniform), tracer_advection = FluxFormSemiLagrangian(),
                                    timestepper = :SplitRungeKutta3, buoyancy = nothing)

bell(x, y, z) = exp(-((x - 0.2)^2 + (y - 0.5)^2) / 0.01) + 0.1 * rand()
set!(model, c = bell, uniform = 1)

wet = [!inactive_cell(i, j, k, grid) & !island((i-0.5)/N, (j-0.5)/N) for i in 1:N, j in 1:N, k in 1:2]
umax = max(maximum(abs, interior(Field(model.velocities.u))), maximum(abs, interior(Field(model.velocities.v))))
Δt = parse(Float64, ARGS[2]) / (N * umax)
uᵢ = Array(interior(Field(model.velocities.u)))[:, :, 1]
vᵢ = Array(interior(Field(model.velocities.v)))[:, :, 1]
lipschitz = Δt * N * max(maximum(abs, diff(uᵢ, dims=1)), maximum(abs, diff(uᵢ, dims=2)), maximum(abs, diff(vᵢ, dims=1)), maximum(abs, diff(vᵢ, dims=2)))
@printf("deformation Courant number Δt |∇u| = %.2f\n", lipschitz)
Cx = Δt * N * maximum(abs, interior(Field(model.velocities.u)))
Cy = Δt * N * maximum(abs, interior(Field(model.velocities.v)))

# Naive truncation loses the part of the swept volume beyond the first dry cell.
# Quantify the resulting uniform-tracer error |δ(lost volume)| / V (the scheme itself renormalises).
function lost_volumes(model, Δt)
    time_step!(model, Δt) # fills the swept Courant numbers
    workspace = model.advection.c.workspace
    sˣ, sʸ, σ⁰ = workspace.geometry.sˣ, workspace.geometry.sʸ, workspace.geometry.σ⁰
    U = model.velocities
    Lx = zeros(N+1, N, 2)
    Ly = zeros(N, N+1, 2)
    for k in 1:2, j in 1:N, i in 1:N+1
        R = swept_region(i, j, k, grid, XSweep(), sˣ, U.u, Δt, σ⁰, Val(3))
        available = sum(R.volumes[m+1] for m in 0:R.n-1; init=0.0) + R.r * R.volumes[R.n+1]
        Lx[i, j, k] = sign(R.F) * (abs(R.F) - available)
    end
    for k in 1:2, j in 1:N+1, i in 1:N
        R = swept_region(i, j, k, grid, YSweep(), sʸ, U.v, Δt, σ⁰, Val(3))
        available = sum(R.volumes[m+1] for m in 0:R.n-1; init=0.0) + R.r * R.volumes[R.n+1]
        Ly[i, j, k] = sign(R.F) * (abs(R.F) - available)
    end
    V = Δ^2 * 0.5
    error = [(Lx[i+1, j, k] - Lx[i, j, k] + Ly[i, j+1, k] - Ly[i, j, k]) / V for i in 1:N, j in 1:N, k in 1:2]
    truncated_faces = count(abs.(Lx) .> 1e-14) + count(abs.(Ly) .> 1e-14)
    return maximum(abs, error[wet]), truncated_faces
end

c₀ = interior(model.tracers.c)[wet]
naive_error, truncated_faces = lost_volumes(model, Δt)
set!(model, c = bell, uniform = 1)
c₀ = interior(model.tracers.c)[wet]

Nsteps = 200
maximum_uniform_error = 0.0
for n in 1:Nsteps
    time_step!(model, Δt)
    global maximum_uniform_error = max(maximum_uniform_error, maximum(abs, interior(model.tracers.uniform)[wet] .- 1))
end
c₁ = interior(model.tracers.c)[wet]

@printf("Prescribed island basin: Cx = %.2f, Cy = %.2f, truncated faces per step = %d\n", Cx, Cy, truncated_faces)
@printf("  naive truncation uniform-tracer error after one step = %.3e\n", naive_error)
@printf("  renormalised: max uniform error = %.2e, mass rel error = %.2e, NaN = %s\n",
        maximum_uniform_error, abs(sum(c₁) - sum(c₀)) / sum(c₀), any(isnan, c₁))
@printf("  min c - min c0 = %.3e, max c - max c0 = %.3e\n", minimum(c₁) - minimum(c₀), maximum(c₁) - maximum(c₀))
