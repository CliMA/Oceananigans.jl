function horizontal_average_is_correct(arch, FT)
    model = Model(N = (16, 16, 16), L = (100, 100, 100), arch=arch, float_type=FT)
    
    # Set a linear stably stratified temperature profile.
    T₀(x, y, z) = 20 + 0.01*z
    set!(model; T=T₀)
    
    T̅ = HorizontalAverage(model, model.tracers.T; interval=0.5second)
    push!(model.diagnostics, T̅)

    time_step!(model, 1, 1)
    all(@. T̅.profile[:] ≈ 20 + 0.01 * model.grid.zC)
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
        @testset "Horizontal average [$(typeof(arch))]" begin
            println("  Testing horizontal average [$(typeof(arch))]")
            @test horizontal_average_is_correct(arch, Float64)
        end
    end
    
    for arch in archs
        @testset "NaN Checker [$(typeof(arch))]" begin
            println("  Testing NaN Checker [$(typeof(arch))]")
            @test_throws ErrorException nan_checker_aborts_simulation(arch, Float64)
        end
    end
end

