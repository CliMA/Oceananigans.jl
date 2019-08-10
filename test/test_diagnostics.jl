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

    @testset "NaN Checker" begin
        println("  Testing NaN Checker...")
        for arch in archs, ft in float_types
            @test_throws ErrorException nan_checker_aborts_simulation(arch, ft)
        end
    end
end

