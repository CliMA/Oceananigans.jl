# Reproduce: julia --project validation/ffsl/one_dimensional_advection.jl

using Oceananigans
include(joinpath(@__DIR__, "..", "..", "test", "setup", "volume_integrals.jl"))
using Printf
using Random
Random.seed!(42)

function model_1d(N, u; cmax = 3, limiter = :monotone)
    grid = RectilinearGrid(size = (N, 1), halo = (cmax + 3, 3), x = (0, N), z = (-1, 0),
                           topology = (Periodic, Flat, Bounded))
    velocities = PrescribedVelocityFields(; u)
    advection = FluxFormSemiLagrangian(; maximum_courant_number = cmax, limiter)
    return HydrostaticFreeSurfaceModel(grid; velocities, tracers = :c, tracer_advection = advection,
                                       timestepper = :SplitRungeKutta3, buoyancy = nothing)
end

ctracer(model) = interior(model.tracers.c)[:, 1, 1]

function run!(model, steps; Δt = 1)
    for _ in 1:steps
        time_step!(model, Δt)
    end
end

println("== conservation, uniform, bounds ==")
N = 64
square(x) = 0.25N < x < 0.5N ? 1.0 : 0.0
for C in (0.5, 1.5, 2.7)
    m = model_1d(N, (x, z, t) -> C)
    set!(m, c = (x, z) -> rand())
    ∫c₀ = volume_integral(m.tracers.c); run!(m, 50)
    mass_error = abs(volume_integral(m.tracers.c) - ∫c₀) / ∫c₀

    m = model_1d(N, (x, z, t) -> C)
    set!(m, c = 1); run!(m, 50)
    uniform_error = maximum(abs, ctracer(m) .- 1)

    m = model_1d(N, (x, z, t) -> C)
    set!(m, c = (x, z) -> square(x)); run!(m, 50)
    c = ctracer(m)
    @printf("C = %.1f  mass %.2e  uniform %.2e  square wave min %.3e max-1 %.3e\n", C, mass_error, uniform_error, minimum(c), maximum(c) - 1)
end

println("== exact integer shifts ==")
for C in (1, 2, 3)
    m = model_1d(N, (x, z, t) -> C)
    set!(m, c = (x, z) -> rand())
    c₀ = ctracer(m); run!(m, 1); c₁ = ctracer(m)
    @printf("C = %d  max |c - shift(c₀)| = %.2e\n", C, maximum(abs, c₁ .- circshift(c₀, C)))
end

println("== convergence, smooth sine, one period at C = 1.5 ==")
for limiter in (nothing, :monotone)
    errors = Float64[]
    Ns = (48, 96, 192, 384)
    for N in Ns
        m = model_1d(N, (x, z, t) -> 1.5; limiter)
        f(x) = sin(2π * x / N)
        set!(m, c = (x, z) -> f(x))
        c₀ = ctracer(m)
        run!(m, Int(N / 1.5))
        push!(errors, sum(abs, ctracer(m) .- c₀) / N)
    end
    orders = [log2(errors[i] / errors[i+1]) for i in 1:length(errors)-1]
    println("limiter = $limiter  L1 errors = ", errors, "  orders = ", orders)
end

println("== variable C(x) crossing integers, steady state c = 1/u ==")
for N in (32, 64, 128)
    ufun(x, z, t) = 1.8 + 1.2 * sin(2π * x / N)
    m = model_1d(N, ufun)
    set!(m, c = (x, z) -> 1 / ufun(x, 0, 0))
    c₀ = ctracer(m); ∫c₀ = volume_integral(m.tracers.c); run!(m, N); c₁ = ctracer(m)
    @printf("N = %d  mass %.2e  max rel deviation from steady state %.3e\n", N, abs(volume_integral(m.tracers.c) - ∫c₀) / ∫c₀, maximum(abs, c₁ ./ c₀ .- 1))
end
