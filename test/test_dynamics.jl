using Printf

function getmodelfield(fieldname, model)
    if fieldname ∈ (:u, :v, :w)
        field = getfield(model.velocities, fieldname)
    else
        field = getfield(model.tracers, fieldname)
    end
    return field
end

function u_relative_error(model, u)
    u_num = model.velocities.u
    u_ans = FaceFieldX(u.(nodes(u_num)..., model.clock.time), model.grid)
    return mean((data(u_num) .- u_ans.data).^2 ) / mean(u_ans.data.^2)
end

function w_relative_error(model, w)
    w_num = model.velocities.w
    w_ans = FaceFieldZ(w.(nodes(w_num)..., model.clock.time), model.grid)
    return mean((data(w_num) .- w_ans.data).^2 ) / mean(w_ans.data.^2)
end

function T_relative_error(model, T)
    T_num = model.tracers.T
    T_ans = CellField(T.(nodes(T_num)..., model.clock.time), model.grid)
    return mean((data(T_num) .- T_ans.data).^2 ) / mean(T_ans.data.^2)
end

function test_diffusion_simple(fieldname)
    model = BasicModel(N=(1, 1, 16), L=(1, 1, 1), ν=1, κ=1, buoyancy=nothing)
    field = getmodelfield(fieldname, model)
    value = π
    data(field) .= value
    time_step!(model, 10, 0.01)
    field_data = data(field)

    return !any(@. !isapprox(value, field_data))
end

function test_diffusion_budget_default(fieldname)
    model = BasicModel(N=(1, 1, 16), L=(1, 1, 1), ν=1, κ=1, buoyancy=nothing)
    field = getmodelfield(fieldname, model)
    half_Nz = round(Int, model.grid.Nz/2)
    data(field)[:, :,   1:half_Nz] .= -1
    data(field)[:, :, half_Nz:end] .=  1

    return test_diffusion_budget(field, model, model.closure.κ, model.grid.Lz)
end

function test_diffusion_budget_channel(fieldname)
    model = BasicChannelModel(N=(1, 16, 4), L=(1, 1, 1), ν=1, κ=1, buoyancy=nothing)
    field = getmodelfield(fieldname, model)
    half_Ny = round(Int, model.grid.Ny/2)
    data(field)[:, 1:half_Ny,   :] .= -1
    data(field)[:, half_Ny:end, :] .=  1

    return test_diffusion_budget(field, model, model.closure.κ, model.grid.Ly)
end

function test_diffusion_budget(field, model, κ, L)
    mean_init = mean(data(field))
    time_step!(model, 100, 1e-4 * L^2 / κ)
    return isapprox(mean_init, mean(data(field)))
end

function test_diffusion_cosine(fieldname)
    Nz, Lz, κ, m = 128, π/2, 1, 2
    model = BasicModel(N=(1, 1, Nz), L=(1, 1, Lz), ν=κ, κ=κ, buoyancy=nothing)
    field = getmodelfield(fieldname, model)

    zC = model.grid.zC
    data(field)[1, 1, :] .= cos.(m*zC)

    diffusing_cosine(κ, m, z, t) = exp(-κ*m^2*t) * cos(m*z)

    time_step!(model, 100, 1e-6 * Lz^2 / κ) # Use small time-step relative to diff. time-scale
    field_numerical = dropdims(data(field), dims=(1, 2))

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
     Θ = a₀ * m * ℕ^2 / (σ^2 - ℕ^2)

    a(x, y, z, t) = exp( -(z - cᵍ*t - z₀)^2 / (2*δ)^2 )

    u(x, y, z, t) =           a(x, y, z, t) * U * cos(k*x + m*z - σ*t)
    v(x, y, z, t) =           a(x, y, z, t) * V * sin(k*x + m*z - σ*t)
    w(x, y, z, t) =           a(x, y, z, t) * W * cos(k*x + m*z - σ*t)
    T(x, y, z, t) = ℕ^2 * z + a(x, y, z, t) * Θ * sin(k*x + m*z - σ*t)

    u₀(x, y, z) = u(x, y, z, 0)
    v₀(x, y, z) = v(x, y, z, 0)
    w₀(x, y, z) = w(x, y, z, 0)
    T₀(x, y, z) = T(x, y, z, 0)

    # Create a model where temperature = buoyancy.
    model = BasicModel(N=(N, 1, N), L=(L, L, L), ν=ν, κ=κ, buoyancy=BuoyancyTracer(),
                       coriolis=VerticalRotationAxis(f=f))

    set_ic!(model, u=u₀, v=v₀, w=w₀, T=T₀)

    time_step!(model, Nt, Δt)

    # Tolerance was found by trial and error...
    u_relative_error(model, u) < 1e-4
end

function passive_tracer_advection_test(; N=128, κ=1e-12, Nt=100)
    L, U, V = 1.0, 0.5, 0.8
    δ, x₀, y₀ = L/15, L/2, L/2

    Δt = 0.05 * L/N / sqrt(U^2 + V^2)

    T(x, y, z, t) = exp( -((x - U*t - x₀)^2 + (y - V*t - y₀)^2) / (2*δ^2) )
    u₀(x, y, z) = U
    v₀(x, y, z) = V
    T₀(x, y, z) = T(x, y, z, 0)

    model = BasicModel(N=(N, N, 2), L=(L, L, L), ν=κ, κ=κ)

    set_ic!(model, u=u₀, v=v₀, T=T₀)
    time_step!(model, Nt, Δt)

    # Error tolerance is a bit arbitrary
    return T_relative_error(model, T) < 1e-4
end

"""
Pearson vortex test
See p. 310 of "Nodal Discontinuous Galerkin Methods: Algorithms, Analysis, and Application" by Hesthaven & Warburton.
"""
function pearson_vortex_test(arch; FT=Float64, N=64, Nt=10)
    Nx, Ny, Nz = N, N, 2
    Lx, Ly, Lz = 1, 1, 1
    ν = 1

    # Choose a very small time step ~O(1/Δx²) as we are diffusion-limited in this test.
    Δt = 1 / (10*π*Nx^2)

    # Pearson vortex analytic solution.
    @inline u(x, y, z, t) = -sin(2π*y) * exp(-4π^2 * ν * t)
    @inline v(x, y, z, t) =  sin(2π*x) * exp(-4π^2 * ν * t)

    ubcs = HorizontallyPeriodicBCs(   top = BoundaryCondition(Gradient, 0),
                                   bottom = BoundaryCondition(Gradient, 0))

    vbcs = HorizontallyPeriodicBCs(   top = BoundaryCondition(Gradient, 0),
                                   bottom = BoundaryCondition(Gradient, 0))

    model = Model(        architecture = arch,
                                  grid = RegularCartesianGrid(FT; N=(Nx, Ny, Nz), L=(Lx, Ly, Lz)),
                               closure = ConstantIsotropicDiffusivity(FT; ν=1, κ=0),  # Turn off diffusivity.
                              buoyancy = nothing, # turn off buoyancy
                   boundary_conditions = BoundaryConditions(u=ubcs, v=vbcs))

    u₀(x, y, z) = u(x, y, z, 0)
    v₀(x, y, z) = v(x, y, z, 0)
    T₀(x, y, z) = 0
    S₀(x, y, z) = 0

    set_ic!(model; u=u₀, v=v₀, T=T₀, S=S₀)

    time_step!(model, Nt, Δt)

    xC, yC, zC = reshape(model.grid.xC, (Nx, 1, 1)), reshape(model.grid.yC, (1, Ny, 1)), reshape(model.grid.zC, (1, 1, Nz))
    xF, yF = reshape(model.grid.xF[1:end-1], (Nx, 1, 1)), reshape(model.grid.yF[1:end-1], (1, Ny, 1))

    t = model.clock.time
    i = model.clock.iteration

    # Calculate relative error between model and analytic solutions for u and v.
    u_rel_err = abs.((data(model.velocities.u) .- u.(xF, yC, zC, t)) ./ u.(xF, yC, zC, t))
    u_rel_err_avg = mean(u_rel_err)
    u_rel_err_max = maximum(u_rel_err)

    v_rel_err = abs.((data(model.velocities.v) .- v.(xC, yF, zC, t)) ./ v.(xC, yF, zC, t))
    v_rel_err_avg = mean(v_rel_err)
    v_rel_err_max = maximum(v_rel_err)

    @info "Pearson vortex test ($arch, $FT) with Nx=Ny=$N @ Nt=$Nt: " *
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

    @testset "Pearson vortex" begin
        println("  Testing Pearson vortex...")
        @test pearson_vortex_test(CPU())
    end
end
