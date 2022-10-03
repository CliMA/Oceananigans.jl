include("dependencies_for_runtests.jl")
include("data_dependencies.jl")

include("regression_tests/shallow_water_bickley_jet_regression.jl")

@testset "Shallow Water Regression" begin
    @info "Running shallow water regression tests..."

    for arch in archs
        for formulation in (VectorInvariantFormulation(), ConservativeFormulation())
            @testset "Shallow Water Bickley jet simulation [$(typeof(arch)), $(typeof(formulation))]" begin
                @info "  Testing shallow water Bickley jet simulation regression [$(typeof(arch)), $(typeof(formulation))]"
                run_shallow_water_regression(arch, formulation; regenerate_data = false)
            end
        end
    end
end
