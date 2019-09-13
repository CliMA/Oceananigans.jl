function horizontal_average_is_correct(arch, FT)
    model = BasicModel(N = (16, 16, 16), L = (100, 100, 100), arch=arch, float_type=FT)

    # Set a linear stably stratified temperature profile.
    T₀(x, y, z) = 20 + 0.01*z
    set!(model; T=T₀)

    T̅ = HorizontalAverage(model, model.tracers.T; interval=0.5second)
    push!(model.diagnostics, T̅)

    time_step!(model, 1, 1)
    correct_profile = @. 20 + 0.01 * collect(model.grid.zC)
    all(Array(T̅.profile[:][2:end-1]) ≈ correct_profile)
end

function product_profile_is_correct(arch, FT)
    model = BasicModel(N = (16, 16, 16), L = (100, 100, 100), arch=arch, float_type=FT)

    # Set a linear stably stratified temperature profile and a sinusoidal u(z) profile.
    u₀(x, y, z) = sin(z)
    T₀(x, y, z) = 20 + 0.01*z
    set!(model; u=u₀, T=T₀)

    uT = HorizontalAverage(model, [model.velocities.u, model.tracers.T]; interval=0.5second)
    run_diagnostic(model, uT)

    correct_profile = @. sin.(model.grid.zC) * (20 + 0.01 * model.grid.zC)
    Array(uT.profile[:][2:end-1]) ≈ correct_profile
end

function nan_checker_aborts_simulation(arch, FT)
    model = BasicModel(N = (16, 16, 2), L = (1, 1, 1), arch=arch, float_type=FT)

    # It checks for NaNs in w by default.
    nc = NaNChecker(model; frequency=1)
    push!(model.diagnostics, nc)

    model.velocities.w[4, 3, 2] = NaN

    time_step!(model, 1, 1);
end

@testset "Diagnostics" begin
    println("Testing diagnostics...")

    for arch in archs
        @testset "Horizontal average [$(typeof(arch))]" begin
            println("  Testing horizontal average [$(typeof(arch))]")
            for FT in float_types
                @test horizontal_average_is_correct(arch, FT)
                @test product_profile_is_correct(arch, FT)
            end
        end
    end

    for arch in archs
        @testset "NaN Checker [$(typeof(arch))]" begin
            println("  Testing NaN Checker [$(typeof(arch))]")
            for FT in float_types
                @test_throws ErrorException nan_checker_aborts_simulation(arch, FT)
            end
        end
    end
end
