function getmodelfield(fieldname, model)
    if fieldname ∈ (:u, :v, :w)
        field = getfield(model.velocities, fieldname)
    else
        field = getfield(model.tracers, fieldname)
    end
    return field
end

function relative_error(u_num, u, time)
    u_ans = Field(location(u_num), architecture(u_num), u_num.grid)
    set!(u_ans, (x, y, z) -> u(x, y, z, time))
    return mean((interior(u_num) .- interior(u_ans)).^2 ) / mean(interior(u_ans).^2)
end

function test_diffusion_simple(fieldname)
    grid = RegularCartesianGrid(size=(1, 1, 16), length=(1, 1, 1))
    closure = ConstantIsotropicDiffusivity(ν=1, κ=1)
    model = Model(grid=grid, closure=closure, buoyancy=nothing)
    field = getmodelfield(fieldname, model)
    value = π
    interior(field) .= value
    time_step!(model, 10, 0.01)
    field_data = interior(field)

    return !any(@. !isapprox(value, field_data))
end

function test_diffusion_budget_default(fieldname)
    grid = RegularCartesianGrid(size=(1, 1, 16), length=(1, 1, 1))
    closure = ConstantIsotropicDiffusivity(ν=1, κ=1)
    model = Model(grid=grid, closure=closure, buoyancy=nothing)
    field = getmodelfield(fieldname, model)
    half_Nz = round(Int, model.grid.Nz/2)
    interior(field)[:, :,   1:half_Nz] .= -1
    interior(field)[:, :, half_Nz:end] .=  1

    return test_diffusion_budget(field, model, model.closure.ν, model.grid.Lz)
end

function test_diffusion_budget_channel(fieldname)
    grid = RegularCartesianGrid(size=(1, 16, 4), length=(1, 1, 1))
    closure = ConstantIsotropicDiffusivity(ν=1, κ=1)
    model = ChannelModel(grid=grid, closure=closure, buoyancy=nothing)
    field = getmodelfield(fieldname, model)
    half_Ny = round(Int, model.grid.Ny/2)
    interior(field)[:, 1:half_Ny,   :] .= -1
    interior(field)[:, half_Ny:end, :] .=  1

    return test_diffusion_budget(field, model, model.closure.ν, model.grid.Ly)
end

function test_diffusion_budget(field, model, κ, L)
    mean_init = mean(interior(field))
    time_step!(model, 100, 1e-4 * L^2 / κ)
    return isapprox(mean_init, mean(interior(field)))
end

function test_diffusion_cosine(fieldname)
    Nz, Lz, κ, m = 128, π/2, 1, 2
    grid = RegularCartesianGrid(size=(1, 1, Nz), length=(1, 1, Lz))
    closure = ConstantIsotropicDiffusivity(ν=κ, κ=κ)
    model = Model(grid=grid, closure=closure, buoyancy=nothing)
    field = getmodelfield(fieldname, model)

    zC = model.grid.zC
    interior(field)[1, 1, :] .= cos.(m*zC)

    diffusing_cosine(κ, m, z, t) = exp(-κ*m^2*t) * cos(m*z)

    time_step!(model, 100, 1e-6 * Lz^2 / κ) # Use small time-step relative to diff. time-scale
    field_numerical = dropdims(interior(field), dims=(1, 2))

    return !any(@. !isapprox(field_numerical, diffusing_cosine(κ, m, zC, model.clock.time), atol=1e-6, rtol=1e-6))
end

function internal_wave_test(; N=128, Nt=10)

    # Internal wave parameters
     ν = κ = 1e-9
     L = 2π
    z₀ = -L/3
     δ = L/20
    a₀ = 1e-3
     m = 16
     k = 1
     f = 0.2
     ℕ = 1.0
     σ = sqrt( (ℕ^2*k^2 + f^2*m^2) / (k^2 + m^2) )

    # Numerical parameters
     N = 128
    Δt = 0.01 * 1/σ

    cᵍ = m * σ / (k^2 + m^2) * (f^2/σ^2 - 1)
     U = a₀ * k * σ   / (σ^2 - f^2)
     V = a₀ * k * f   / (σ^2 - f^2)
     W = a₀ * m * σ   / (σ^2 - ℕ^2)
     B = a₀ * m * ℕ^2 / (σ^2 - ℕ^2)

    a(x, y, z, t) = exp( -(z - cᵍ*t - z₀)^2 / (2*δ)^2 )

    u(x, y, z, t) =           a(x, y, z, t) * U * cos(k*x + m*z - σ*t)
    v(x, y, z, t) =           a(x, y, z, t) * V * sin(k*x + m*z - σ*t)
    w(x, y, z, t) =           a(x, y, z, t) * W * cos(k*x + m*z - σ*t)
    b(x, y, z, t) = ℕ^2 * z + a(x, y, z, t) * B * sin(k*x + m*z - σ*t)

    u₀(x, y, z) = u(x, y, z, 0)
    v₀(x, y, z) = v(x, y, z, 0)
    w₀(x, y, z) = w(x, y, z, 0)
    b₀(x, y, z) = b(x, y, z, 0)

    # Create a model where temperature = buoyancy.
    grid = RegularCartesianGrid(size=(N, 1, N), length=(L, L, L))
    closure = ConstantIsotropicDiffusivity(ν=ν, κ=κ)
    model = Model(grid=grid, closure=closure, buoyancy=BuoyancyTracer(), tracers=:b, coriolis=FPlane(f=f))

    set_ic!(model, u=u₀, v=v₀, w=w₀, b=b₀)

    time_step!(model, Nt, Δt)

    # Tolerance was found by trial and error...
    return relative_error(model.velocities.u, u, model.clock.time) < 1e-4
end

function passive_tracer_advection_test(; N=128, κ=1e-12, Nt=100)
    L, U, V = 1.0, 0.5, 0.8
    δ, x₀, y₀ = L/15, L/2, L/2

    Δt = 0.05 * L/N / sqrt(U^2 + V^2)

    T(x, y, z, t) = exp( -((x - U*t - x₀)^2 + (y - V*t - y₀)^2) / (2*δ^2) )
    u₀(x, y, z) = U
    v₀(x, y, z) = V
    T₀(x, y, z) = T(x, y, z, 0)

    grid = RegularCartesianGrid(size=(N, N, 2), length=(L, L, L))
    closure = ConstantIsotropicDiffusivity(ν=κ, κ=κ)
    model = Model(grid=grid, closure=closure)

    set_ic!(model, u=u₀, v=v₀, T=T₀)
    time_step!(model, Nt, Δt)

    # Error tolerance is a bit arbitrary
    return relative_error(model.tracers.T, T, model.clock.time) < 1e-4
end

"""
Taylor-Green vortex test
See: https://en.wikipedia.org/wiki/Taylor%E2%80%93Green_vortex#Taylor%E2%80%93Green_vortex_solution
     and p. 310 of "Nodal Discontinuous Galerkin Methods: Algorithms, Analysis, and Application" by Hesthaven & Warburton.
"""
function taylor_green_vortex_test(arch; FT=Float64, N=64, Nt=10)
    Nx, Ny, Nz = N, N, 2
    Lx, Ly, Lz = 1, 1, 1
    ν = 1

    # Choose a very small time step as we are diffusion-limited in this test: Δt ≤ Δx² / 2ν
    Δx = Lx / Nx
    Δt = (1/10π) * Δx^2 / ν

    # Taylor-Green vortex analytic solution.
    @inline u(x, y, z, t) = -sin(2π*y) * exp(-4π^2 * ν * t)
    @inline v(x, y, z, t) =  sin(2π*x) * exp(-4π^2 * ν * t)

    model = Model(architecture = arch,
                          grid = RegularCartesianGrid(FT; size=(Nx, Ny, Nz), length=(Lx, Ly, Lz)),
                       closure = ConstantIsotropicDiffusivity(FT; ν=1, κ=0),  # Turn off diffusivity.
                       tracers = nothing,
                      buoyancy = nothing) # turn off buoyancy

    u₀(x, y, z) = u(x, y, z, 0)
    v₀(x, y, z) = v(x, y, z, 0)
    set_ic!(model; u=u₀, v=v₀)

    time_step!(model, Nt, Δt)

    xC, yC, zC = reshape(model.grid.xC, (Nx, 1, 1)), reshape(model.grid.yC, (1, Ny, 1)), reshape(model.grid.zC, (1, 1, Nz))
    xF, yF = reshape(model.grid.xF[1:end-1], (Nx, 1, 1)), reshape(model.grid.yF[1:end-1], (1, Ny, 1))

    t = model.clock.time
    i = model.clock.iteration

    # Calculate relative error between model and analytic solutions for u and v.
    u_rel_err = abs.((interior(model.velocities.u) .- u.(xF, yC, zC, t)) ./ u.(xF, yC, zC, t))
    u_rel_err_avg = mean(u_rel_err)
    u_rel_err_max = maximum(u_rel_err)

    v_rel_err = abs.((interior(model.velocities.v) .- v.(xC, yF, zC, t)) ./ v.(xC, yF, zC, t))
    v_rel_err_avg = mean(v_rel_err)
    v_rel_err_max = maximum(v_rel_err)

    @info "Taylor-Green vortex test ($arch, $FT) with Nx=Ny=$N @ Nt=$Nt: " *
          @sprintf("Δu: (avg=%6.3g, max=%6.3g), Δv: (avg=%6.3g, max=%6.3g)\n",
                   u_rel_err_avg, u_rel_err_max, v_rel_err_avg, v_rel_err_max)

    u_rel_err_max < 5e-6 && v_rel_err_max < 5e-6
end

@testset "Dynamics" begin
    println("Testing dynamics...")

    @testset "Simple diffusion" begin
        println("  Testing simple diffusion...")
        for fieldname in (:u, :v, :T, :S)
            @test test_diffusion_simple(fieldname)
        end
    end

    @testset "Budgets in isotropic diffusion" begin
        println("  Testing default model budgets with isotropic diffusion...")
        for fieldname in (:u, :v, :T, :S)
            @test test_diffusion_budget_default(fieldname)
        end

        for fieldname in (:u, :T, :S)
            @test test_diffusion_budget_channel(fieldname)
        end
    end

    @testset "Diffusion cosine" begin
        println("  Testing diffusion cosine...")
        for fieldname in (:u, :v, :T, :S)
            @test test_diffusion_cosine(fieldname)
        end
    end

    @testset "Passive tracer advection" begin
        println("  Testing passive tracer advection...")
        @test passive_tracer_advection_test()
    end

    @testset "Internal wave" begin
        println("  Testing internal wave...")
        @test internal_wave_test()
    end

    @testset "Taylor-Green vortex" begin
        println("  Testing Taylor-Green vortex...")
        @test taylor_green_vortex_test(CPU())
    end
end
