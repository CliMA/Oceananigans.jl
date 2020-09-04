using Pkg
using SafeTestsets

const CONVERGENCE_DIR = joinpath(@__DIR__, "..", "verification", "convergence_tests")

Pkg.activate(CONVERGENCE_DIR)
Pkg.instantiate()
Pkg.develop(PackageSpec(path=joinpath(@__DIR__, "..")))

@testset "Convergence" begin
    @safetestset "0D point exponential decay" begin
        include(joinpath(@__DIR__, "..", "verification", "convergence_tests", "point_exponential_decay.jl"))
    end

    @safetestset "1D cosine advection-diffusion" begin
        include(joinpath(@__DIR__, "..", "verification", "convergence_tests", "one_dimensional_cosine_advection_diffusion.jl"))
    end

    @safetestset "1D Gaussian advection-diffusion" begin
        include(joinpath(@__DIR__, "..", "verification", "convergence_tests", "one_dimensional_gaussian_advection_diffusion.jl"))
    end

    @safetestset "2D diffusion" begin
        include(joinpath(@__DIR__, "..", "verification", "convergence_tests", "two_dimensional_diffusion.jl"))
    end

    @safetestset "2D Taylor-Green" begin
        include(joinpath(@__DIR__, "..", "verification", "convergence_tests", "run_taylor_green.jl"))
        include(joinpath(@__DIR__, "..", "verification", "convergence_tests", "analyze_taylor_green.jl"))
    end

    @safetestset "2D forced free-slip" begin
        include(joinpath(@__DIR__, "..", "verification", "convergence_tests", "run_forced_free_slip.jl"))
        include(joinpath(@__DIR__, "..", "verification", "convergence_tests", "analyze_forced_free_slip.jl"))
    end

    @safetestset "2D forced fixed-slip" begin
        include(joinpath(@__DIR__, "..", "verification", "convergence_tests", "run_forced_fixed_slip.jl"))
        include(joinpath(@__DIR__, "..", "verification", "convergence_tests", "analyze_forced_fixed_slip.jl"))
    end
end
