function summarize_regression_test(field_names, fields, correct_fields)
    for (field_name, φ, φ_c) in zip(field_names, fields, correct_fields)
        Δ = Array(φ) .- φ_c

        Δ_min      = minimum(Δ)
        Δ_max      = maximum(Δ)
        Δ_mean     = mean(Δ)
        Δ_abs_mean = mean(abs, Δ)
        Δ_std      = std(Δ)

        @info(@sprintf("Δ%s: min=%.6g, max=%.6g, mean=%.6g, absmean=%.6g, std=%.6g\n",
                       field_name, Δ_min, Δ_max, Δ_mean, Δ_abs_mean, Δ_std))
    end
end

include("regression_tests/thermal_bubble_regression_test.jl")
include("regression_tests/rayleigh_benard_regression_test.jl")
include("regression_tests/ocean_large_eddy_simulation_regression_test.jl")

@testset "Regression" begin
    println("Running regression tests...")

    for arch in archs
        @testset "Thermal bubble [$(typeof(arch))]" begin
            println("  Testing thermal bubble regression [$(typeof(arch))]")
            run_thermal_bubble_regression_test(arch)
        end

        @testset "Rayleigh–Bénard tracer [$(typeof(arch))]" begin
            println("  Testing Rayleigh–Bénard tracer regression [$(typeof(arch))]")
            run_rayleigh_benard_regression_test(arch)
        end

        @testset "Ocean large eddy simulation [$(typeof(arch))]" begin
            for closure in (AnisotropicMinimumDissipation(), ConstantSmagorinsky())
                closurename = string(typeof(closure).name.wrapper)
                println("  Testing oceanic large eddy simulation regression [$closurename, $(typeof(arch))]")
                run_ocean_large_eddy_simulation_regression_test(arch, closure)
            end
        end
    end
end
