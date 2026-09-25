# Reproduce: julia --project validation/ffsl/lauritzen_deformational_flow.jl

using Oceananigans
using Printf
using Random
Random.seed!(42)

# Discretely non-divergent velocities from a streamfunction evaluated at cell corners
struct StreamfunctionVelocity{D, P}
    ψ :: P
    Δ :: Float64
end
StreamfunctionVelocity{D}(ψ, Δ) where D = StreamfunctionVelocity{D, typeof(ψ)}(ψ, Δ)
(s::StreamfunctionVelocity{:u})(x, y, z, t) = - (s.ψ(x, y + s.Δ/2, t) - s.ψ(x, y - s.Δ/2, t)) / s.Δ
(s::StreamfunctionVelocity{:v})(x, y, z, t) = + (s.ψ(x + s.Δ/2, y, t) - s.ψ(x - s.Δ/2, y, t)) / s.Δ

function model_2d(N, ψ; topology = (Periodic, Periodic, Bounded), cmax = 3, limiter = :monotone, tracers = (:c, :one), scheme = nothing)
    grid = RectilinearGrid(size = (N, N, 1), halo = (cmax + 3, cmax + 3, 3), x = (0, 1), y = (0, 1), z = (-1, 0); topology)
    Δ = 1 / N
    uψ = StreamfunctionVelocity{:u}(ψ, Δ)
    vψ = StreamfunctionVelocity{:v}(ψ, Δ)
    velocities = PrescribedVelocityFields(u = (x, y, z, t) -> uψ(x, y, z, t), v = (x, y, z, t) -> vψ(x, y, z, t))
    advection = isnothing(scheme) ? FluxFormSemiLagrangian(; maximum_courant_number = cmax, limiter) : scheme
    return HydrostaticFreeSurfaceModel(grid; velocities, tracers, tracer_advection = advection,
                                       timestepper = :SplitRungeKutta3, buoyancy = nothing)
end

# Periodic analogue of Lauritzen et al. (2012) case 4: deformation plus translation, reverses at t = T/2
const T = 1.0
const κ = 1.0
deformational_ψ(x, y, t) = κ / π * sin(π * (x - t / T))^2 * sin(π * y)^2 * cos(π * t / T) - y / T

# Solid body rotation in a disk of radius R inside a closed box
const ω = 2π
const R = 0.45
rotation_ψ(x, y, t) = ω / 2 * min((x - 0.5)^2 + (y - 0.5)^2, R^2)

function cosine_bells(x, y)
    r₁ = sqrt((x - 0.35)^2 + (y - 0.5)^2)
    r₂ = sqrt((x - 0.65)^2 + (y - 0.5)^2)
    rᵇ = 0.15
    b₁ = r₁ < rᵇ ? (1 + cos(π * r₁ / rᵇ)) / 2 : 0.0
    b₂ = r₂ < rᵇ ? (1 + cos(π * r₂ / rᵇ)) / 2 : 0.0
    return 0.1 + 0.9 * (b₁ + b₂)
end

gaussian_hills(x, y) = 0.95 * (exp(-5 * ((x - 0.35)^2 + (y - 0.5)^2) / 0.1^2 * 0.2) + exp(-5 * ((x - 0.65)^2 + (y - 0.5)^2) / 0.1^2 * 0.2))

function slotted_cylinders(x, y)
    r₁ = sqrt((x - 0.35)^2 + (y - 0.5)^2)
    r₂ = sqrt((x - 0.65)^2 + (y - 0.5)^2)
    rᶜ = 0.15
    c₁ = (r₁ < rᶜ) & !((abs(x - 0.35) < rᶜ / 6) & (y - 0.5 > - 5rᶜ / 12))
    c₂ = (r₂ < rᶜ) & !((abs(x - 0.65) < rᶜ / 6) & (y - 0.5 < 5rᶜ / 12))
    return c₁ | c₂ ? 1.0 : 0.1
end

interior2d(f) = Array(interior(f))[:, :, 1]

function lauritzen_errors(q, qᵀ)
    l₁ = sum(abs, q .- qᵀ) / sum(abs, qᵀ)
    l₂ = sqrt(sum(abs2, q .- qᵀ) / sum(abs2, qᵀ))
    l∞ = maximum(abs, q .- qᵀ) / maximum(abs, qᵀ)
    return l₁, l₂, l∞
end

max_courant(model, Δt, N) = Δt * N * max(maximum(abs, interior(Field(model.velocities.u))), maximum(abs, interior(Field(model.velocities.v))))

println("== uniform tracer and mass conservation at C > 1 ==")
for (label, ψ, topology, Nsteps) in (("deformational", deformational_ψ, (Periodic, Periodic, Bounded), 40),
                                     ("solid body rotation", rotation_ψ, (Bounded, Bounded, Bounded), 64),
                                     ("solid body rotation", rotation_ψ, (Bounded, Bounded, Bounded), 100))
    N = 64
    model = model_2d(N, ψ; topology)
    set!(model, c = (x, y, z) -> cosine_bells(x, y) + 0.1 * rand(), one = 1)
    Δt = T / Nsteps
    C = max_courant(model, Δt, N)
    c₀ = interior2d(model.tracers.c)
    maxerr = 0.0
    for n in 1:Nsteps
        time_step!(model, Δt)
        maxerr = max(maxerr, maximum(abs, interior2d(model.tracers.one) .- 1))
    end
    c₁ = interior2d(model.tracers.c)
    @printf("%-20s N = %d  max C = %.2f  max|1 - c| over run = %.2e  mass rel err = %.2e  min c = %.3e\n",
            label, N, C, maxerr, abs(sum(c₁) - sum(c₀)) / sum(c₀), minimum(c₁))
end


println("== Lauritzen deformational flow error table (one period, return to initial condition) ==")
for (label, q) in (("cosine bells", cosine_bells), ("gaussian hills", gaussian_hills), ("slotted cylinders", slotted_cylinders))
    for limiter in (:monotone, nothing)
        for N in (50, 100, 200)
            model = model_2d(N, deformational_ψ; limiter, tracers = :c)
            set!(model, c = (x, y, z) -> q(x, y))
            qᵀ = interior2d(model.tracers.c)
            Nsteps = Int(0.8N)
            Δt = T / Nsteps
            C = max_courant(model, Δt, N)
            for n in 1:Nsteps
                time_step!(model, Δt)
            end
            qₙ = interior2d(model.tracers.c)
            l₁, l₂, l∞ = lauritzen_errors(qₙ, qᵀ)
            @printf("%-18s limiter=%-9s N=%4d  C=%.2f  l1=%.3e l2=%.3e linf=%.3e  min=%.3e max=%.4f\n",
                    label, string(limiter), N, C, l₁, l₂, l∞, minimum(qₙ), maximum(qₙ))
        end
    end
end
