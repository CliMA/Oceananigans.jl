# CPU cost per tracer of FluxFormSemiLagrangian against WENO(order=5), for 1 and 24 tracers.
#
#   julia --project validation/ffsl/cost_per_tracer.jl
#
# Velocities are prescribed as fields. The time step is the same for both schemes (the cost of one step does not depend on it).
# Reported times are the minimum over several repetitions of `Nsteps` time steps, single-threaded.

using Oceananigans
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Printf

const Nsteps = 5
const repetitions = 3

deformational_ψ(x, y, t) = 1 / π * sin(π * x)^2 * sin(π * y)^2 * cos(π * t)

function build_model(N, Nz, scheme, Ntracers)
    grid = RectilinearGrid(size = (N, N, Nz), halo = (6, 6, 3), x = (0, 1), y = (0, 1), z = (-1, 0),
                           topology = (Periodic, Periodic, Bounded))
    Δ = 1 / N
    u = XFaceField(grid)
    v = YFaceField(grid)
    set!(u, (x, y, z) -> - (deformational_ψ(x, y + Δ/2, 0) - deformational_ψ(x, y - Δ/2, 0)) / Δ)
    set!(v, (x, y, z) -> + (deformational_ψ(x + Δ/2, y, 0) - deformational_ψ(x - Δ/2, y, 0)) / Δ)
    fill_halo_regions!((u, v))
    tracers = Tuple(Symbol(:c, n) for n in 1:Ntracers)
    model = HydrostaticFreeSurfaceModel(grid; velocities = PrescribedVelocityFields(; u, v), tracers,
                                        tracer_advection = scheme, timestepper = :SplitRungeKutta3,
                                        buoyancy = nothing)
    for name in tracers
        set!(model.tracers[name], (x, y, z) -> exp(-((x - 0.5)^2 + (y - 0.5)^2) / 0.02))
    end
    return model
end

function time_model(model, Δt)
    time_step!(model, Δt)
    elapsed = Inf
    for _ in 1:repetitions
        repetition_time = @elapsed begin
            for _ in 1:Nsteps
                time_step!(model, Δt)
            end
        end
        elapsed = min(elapsed, repetition_time)
    end
    return elapsed / Nsteps
end

N, Nz = 128, 8
Δt = 1 / N
schemes = (("WENO(order=5)", WENO(order=5)), ("FluxFormSemiLagrangian()", FluxFormSemiLagrangian()))

@printf("Grid %d × %d × %d, %d thread(s)\n", N, N, Nz, Threads.nthreads())
@printf("%-26s %10s %10s %16s %22s\n", "scheme", "1 tracer", "24 tracers", "per tracer (24)", "marginal per tracer")

for (label, scheme) in schemes
    t₁  = time_model(build_model(N, Nz, scheme, 1), Δt)
    t₂₄ = time_model(build_model(N, Nz, scheme, 24), Δt)
    @printf("%-26s %8.1f ms %8.1f ms %13.2f ms %19.2f ms\n", label, 1e3t₁, 1e3t₂₄, 1e3t₂₄ / 24, 1e3(t₂₄ - t₁) / 23)
end
