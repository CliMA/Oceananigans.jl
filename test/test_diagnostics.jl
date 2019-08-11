function vertical_profile_is_correct(arch, FT)
    model = Model(N = (16, 16, 16), L = (100, 100, 100), arch=arch, float_type=FT)
    
    # Set a linear stably stratified temperature profile.
    T₀(x, y, z) = 20 + 0.01*z
    set!(model; T=T₀)
    
    T̅ = VerticalProfile(model, model.tracers.T; interval=0.5second)
    push!(model.diagnostics, T̅)

    time_step!(model, 1, 1)
    all(@. T̅.profile[:] ≈ 20 + 0.01 * model.grid.zC)
end

function product_profile_is_correct(arch, FT)
    model = Model(N = (16, 16, 16), L = (100, 100, 100), arch=arch, float_type=FT)

    # Set a linear stably stratified temperature profile and a sinusoidal u(z) profile.
    u₀(x, y, z) = sin(z)
    T₀(x, y, z) = 20 + 0.01*z
    set!(model; u=u₀, T=T₀)

    uT = ProductProfile(model, model.velocities.u, model.tracers.T; interval=0.5second)
    run_diagnostic(model, uT)

    correct_profile = @. sin.(model.grid.zC) * (20 + 0.01 * model.grid.zC)
    Array(uT.profile[:]) ≈ correct_profile
end

function velocity_covariances_initialize(arch, FT)
    model = Model(N = (16, 16, 16), L = (100, 100, 100), arch=arch, float_type=FT)

    u₀(x, y, z) = sin(z)
    v₀(x, y, z) = x
    w₀(x, y, z) = tanh(x*y)
    set!(model; u=u₀, v=v₀, w=w₀)

    vcp = VelocityCovarianceProfiles(model; interval=0.5second) 
    run_diagnostic(model, vcp)

    return true  # Just check that it didn't crash.
end

function nan_checker_aborts_simulation(arch, FT)
    model = Model(N = (16, 16, 2), L = (1, 1, 1), arch=arch, float_type=FT)

    # It checks for NaNs in w by default.
    nc = NaNChecker(model; frequency=1)
    push!(model.diagnostics, nc)

    model.velocities.w[4, 3, 2] = NaN
    
    time_step!(model, 1, 1);
end

@testset "Diagnostics" begin
    println("Testing diagnostics...")

    for arch in archs
        @testset "Vertical profiles [$(typeof(arch))]" begin
            println("  Testing vertical profiles [$(typeof(arch))]")
            @test vertical_profile_is_correct(arch, Float64)
            @test product_profile_is_correct(arch, Float64)
            @test velocity_covariances_initialize(arch, Float64)
        end
    end
    
    for arch in archs
        @testset "NaN Checker [$(typeof(arch))]" begin
            println("  Testing NaN Checker [$(typeof(arch))]")
            @test_throws ErrorException nan_checker_aborts_simulation(arch, Float64)
        end
    end
end

