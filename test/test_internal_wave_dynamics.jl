function internal_wave_solution(; L, background_stratification=false)
    # Internal wave parameters
     ν = κ = 1e-9
    z₀ = -L/3
     δ = L/20
    a₀ = 1e-3
     m = 16
     k = 1
     f = 0.2
     ℕ = 1.0
     σ = sqrt( (ℕ^2*k^2 + f^2*m^2) / (k^2 + m^2) )

    # Numerical parameters
    Δt = 0.01 * 1/σ

    cᵍ = m * σ / (k^2 + m^2) * (f^2/σ^2 - 1)
     U = a₀ * k * σ   / (σ^2 - f^2)
     V = a₀ * k * f   / (σ^2 - f^2)
     W = a₀ * m * σ   / (σ^2 - ℕ^2)
     B = a₀ * m * ℕ^2 / (σ^2 - ℕ^2)

    a(x, y, z, t) = exp( -(z - cᵍ*t - z₀)^2 / (2*δ)^2 )

    u(x, y, z, t) = a(x, y, z, t) * U * cos(k*x + m*z - σ*t)
    v(x, y, z, t) = a(x, y, z, t) * V * sin(k*x + m*z - σ*t)
    w(x, y, z, t) = a(x, y, z, t) * W * cos(k*x + m*z - σ*t)

    # Buoyancy field depends on whether we use background fields or not
    wavy_b(x, y, z, t) = a(x, y, z, t) * B * sin(k*x + m*z - σ*t)
    background_b(x, y, z, t) = ℕ^2 * z

    if background_stratification # Move stratification to a background field
        background_fields = (; b=background_b)
        b = wavy_b
    else
        background_fields = NamedTuple()
        b(x, y, z, t) = wavy_b(x, y, z, t) + background_b(x, y, z, t)
    end

    u₀(x, y, z) = u(x, y, z, 0)
    v₀(x, y, z) = v(x, y, z, 0)
    w₀(x, y, z) = w(x, y, z, 0)
    b₀(x, y, z) = b(x, y, z, 0)

    solution = (; u, v, w, b)

    # Form model keyword arguments
    closure = IsotropicDiffusivity(ν=ν, κ=κ)
    buoyancy = BuoyancyTracer()
    tracers = :b
    coriolis = FPlane(f=f)
    model_kwargs = (; closure, buoyancy, tracers, coriolis)

    return solution, model_kwargs, background_fields, Δt, σ
end

function internal_wave_dynamics_test(model, solution, Δt)
    # Make initial conditions
    u₀(x, y, z) = solution.u(x, y, z, 0)
    v₀(x, y, z) = solution.v(x, y, z, 0)
    w₀(x, y, z) = solution.w(x, y, z, 0)
    b₀(x, y, z) = solution.b(x, y, z, 0)
    
    set!(model, u=u₀, v=v₀, w=w₀, b=b₀)

    simulation = Simulation(model, stop_iteration=10, Δt=Δt)
    try run!(simulation); catch; end # so the test continues to execute if there's a NaN

    # Tolerance was found by trial and error...
    @test relative_error(model.velocities.u, solution.u, model.clock.time) < 1e-4

    return nothing
end

