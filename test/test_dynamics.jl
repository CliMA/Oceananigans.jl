function relative_error(u_num, u, time)
    u_ans = Field(location(u_num), architecture(u_num), u_num.grid, nothing)
    set!(u_ans, (x, y, z) -> u(x, y, z, time))
    return mean((interior(u_num) .- interior(u_ans)).^2 ) / mean(interior(u_ans).^2)
end

function test_diffusion_simple(fieldname)
    model = IncompressibleModel(    grid = RegularCartesianGrid(size=(1, 1, 16), extent=(1, 1, 1)),
                                 closure = IsotropicDiffusivity(ν=1, κ=1),
                                buoyancy = nothing)
    field = get_model_field(fieldname, model)

    value = π
    interior(field) .= value

    for n in 1:10
        time_step!(model, 1, euler= n==1)
    end

    field_data = interior(field)

    return !any(@. !isapprox(value, field_data))
end

function test_isotropic_diffusion_budget(fieldname, model)
    set!(model; u=0, v=0, w=0, T=0, S=0)
    set!(model; Dict(fieldname => (x, y, z) -> rand())...)

    field = get_model_field(fieldname, model)

    return test_diffusion_budget(fieldname, field, model, model.closure.ν, model.grid.Δz)
end

function test_biharmonic_diffusion_budget(fieldname, model)
    set!(model; u=0, v=0, w=0, T=0, S=0)
    set!(model; Dict(fieldname => (x, y, z) -> rand())...)

    field = get_model_field(fieldname, model)

    return test_diffusion_budget(fieldname, field, model, model.closure.νz, model.grid.Δz, 4)
end

function test_diffusion_budget(fieldname, field, model, κ, Δ, order=2)
    init_mean = mean(interior(field))

    for n in 1:100
        # Very small time-steps required to bring error under machine precision
        time_step!(model, 1e-4 * Δ^order / κ, euler= n==1)
    end

    final_mean = mean(interior(field))

    @info @sprintf("    Initial <%s>: %.16f, final <%s>: %.16f, final - initial: %.4e",
                   fieldname, init_mean, fieldname, final_mean, final_mean - init_mean)

    return isapprox(init_mean, final_mean)
end

function test_diffusion_cosine(fieldname)
    Nz, Lz, κ, m = 128, π/2, 1, 2

    grid = RegularCartesianGrid(size=(1, 1, Nz), x=(0, 1), y=(0, 1), z=(0, Lz))

    model = IncompressibleModel(    grid = grid,
                                 closure = IsotropicDiffusivity(ν=κ, κ=κ),
                                buoyancy = nothing)

    field = get_model_field(fieldname, model)

    zC = znodes(Cell, grid, reshape=true)
    interior(field) .= cos.(m * zC)

    diffusing_cosine(κ, m, z, t) = exp(-κ * m^2 * t) * cos(m * z)

    # Step forward with small time-step relative to diff. time-scale
    Δt = 1e-6 * Lz^2 / κ
    for n in 1:100
        time_step!(model, Δt, euler=n==1)
    end

     numerical = interior(field)
    analytical = diffusing_cosine.(κ, m, zC, model.clock.time)

    return !any(@. !isapprox(numerical, analytical, atol=1e-6, rtol=1e-6))
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

    model = IncompressibleModel(    grid = RegularCartesianGrid(size=(N, 1, N), extent=(L, L, L)),
                                 closure = IsotropicDiffusivity(ν=ν, κ=κ),
                                buoyancy = BuoyancyTracer(),
                                 tracers = :b,
                                coriolis = FPlane(f=f))

    set!(model, u=u₀, v=v₀, w=w₀, b=b₀)

    for n in 1:Nt
        time_step!(model, Δt, euler= n==1)
    end

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

    grid = RegularCartesianGrid(size=(N, N, 2), extent=(L, L, L))
    closure = IsotropicDiffusivity(ν=κ, κ=κ)
    model = IncompressibleModel(grid=grid, closure=closure)

    set!(model, u=u₀, v=v₀, T=T₀)

    for n in 1:Nt
        time_step!(model, Δt, euler= n==1)
    end

    # Error tolerance is a bit arbitrary
    return relative_error(model.tracers.T, T, model.clock.time) < 1e-4
end

"""
Taylor-Green vortex test
See: https://en.wikipedia.org/wiki/Taylor%E2%80%93Green_vortex#Taylor%E2%80%93Green_vortex_solution
     and p. 310 of "Nodal Discontinuous Galerkin Methods: Algorithms, Analysis, and Application"
     by Hesthaven & Warburton.
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

    model = IncompressibleModel(
        architecture = arch,
                grid = RegularCartesianGrid(FT, size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz)),
             closure = IsotropicDiffusivity(FT, ν=1, κ=0),  # Turn off diffusivity.
             tracers = nothing,
            buoyancy = nothing)

    u₀(x, y, z) = u(x, y, z, 0)
    v₀(x, y, z) = v(x, y, z, 0)
    set!(model, u=u₀, v=v₀)

    for n in 1:Nt
        time_step!(model, Δt, euler = n==1)
    end

    xF, yC, zC = nodes(model.velocities.u, reshape=true)
    xC, yF, zC = nodes(model.velocities.v, reshape=true)

    t = model.clock.time
    i = model.clock.iteration

    # Calculate relative error between model and analytic solutions for u and v.
    u_rel_err = abs.((interior(model.velocities.u) .- u.(xF, yC, zC, t)) ./ u.(xF, yC, zC, t))
    u_rel_err_avg = mean(u_rel_err)
    u_rel_err_max = maximum(u_rel_err)

    v_rel_err = abs.((interior(model.velocities.v) .- v.(xC, yF, zC, t)) ./ v.(xC, yF, zC, t))
    v_rel_err_avg = mean(v_rel_err)
    v_rel_err_max = maximum(v_rel_err)

    @info "Taylor-Green vortex test [$arch, $FT, Nx=Ny=$N, Nt=$Nt]: " *
          @sprintf("Δu: (avg=%6.3g, max=%6.3g), Δv: (avg=%6.3g, max=%6.3g)",
                   u_rel_err_avg, u_rel_err_max, v_rel_err_avg, v_rel_err_max)

    u_rel_err_max < 5e-6 && v_rel_err_max < 5e-6
end

@testset "Dynamics" begin
    @info "Testing dynamics..."

    @testset "Simple diffusion" begin
        @info "  Testing simple diffusion..."
        for fieldname in (:u, :v, :T, :S)
            @test test_diffusion_simple(fieldname)
        end
    end

    @testset "Budgets in isotropic diffusion" begin
        @info "  Testing model budgets with isotropic diffusion..."
        for topology in ((Periodic, Periodic, Periodic),
                         (Periodic, Periodic, Bounded),
                         (Periodic, Bounded, Bounded),
                         (Bounded, Bounded, Bounded))

            fieldnames = [:T, :S]

            topology[1] === Periodic && push!(fieldnames, :u)
            topology[2] === Periodic && push!(fieldnames, :v)
            #topology[3] === Periodic && push!(fieldnames, :w)

            grid = RegularCartesianGrid(size=(16, 16, 16), extent=(1, 1, 1), topology=topology)

            model = IncompressibleModel(    grid = grid,
                                         closure = IsotropicDiffusivity(ν=1, κ=1),
                                        coriolis = nothing,
                                        buoyancy = nothing)
                
            for fieldname in fieldnames
                @info "    Testing $fieldname budget in a $topology domain with isotropic diffusion..."
                @test test_isotropic_diffusion_budget(fieldname, model)
            end
        end
    end

    @testset "Budgets in biharmonic diffusion" begin
        @info "  Testing model budgets with biharmonic diffusion..."
        for topology in ((Periodic, Periodic, Periodic),
                         (Periodic, Periodic, Bounded),
                         (Periodic, Bounded, Bounded),
                         (Bounded, Bounded, Bounded))

            fieldnames = [:T, :S]

            topology[1] === Periodic && push!(fieldnames, :u)
            topology[2] === Periodic && push!(fieldnames, :v)
            #topology[3] === Periodic && push!(fieldnames, :w)

            grid = RegularCartesianGrid(size=(16, 16, 16), extent=(1, 1, 1), halo=(2, 2, 2), topology=topology)

            model = IncompressibleModel(    grid = grid,
                                         closure = AnisotropicBiharmonicDiffusivity(νh=1, νz=1, κh=1, κz=1),
                                        coriolis = nothing,
                                        buoyancy = nothing)
                
            for fieldname in fieldnames
                @info "    Testing $fieldname budget in a $topology domain with biharmonic diffusion..."
                @test test_biharmonic_diffusion_budget(fieldname, model)
            end
        end
    end

    @testset "Diffusion cosine" begin
        @info "  Testing diffusion cosine..."
        for fieldname in (:u, :v, :T, :S)
            @test test_diffusion_cosine(fieldname)
        end
    end

    @testset "Passive tracer advection" begin
        @info "  Testing passive tracer advection..."
        @test passive_tracer_advection_test()
    end

    @testset "Internal wave" begin
        @info "  Testing internal wave..."
        @test internal_wave_test()
    end

    @testset "Taylor-Green vortex" begin
        @info "  Testing Taylor-Green vortex..."
        @test taylor_green_vortex_test(CPU())
    end
end
