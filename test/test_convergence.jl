using Pkg
using SafeTestsets

const CONVERGENCE_DIR = joinpath(@__DIR__, "..", "validation", "convergence_tests")

Pkg.activate(CONVERGENCE_DIR)
Pkg.instantiate()
Pkg.develop(PackageSpec(path=joinpath(@__DIR__, "..")))

vt = get(ENV, "VALIDATION_TEST", :all) |> Symbol

@testset "Convergence" begin
    if vt == :point_exponential_decay || vt == :all
        @safetestset "0D point exponential decay" begin
            include(joinpath(@__DIR__, "..", "validation", "convergence_tests", "point_exponential_decay.jl"))
        end
    end

    if vt == :cosine_advection_diffusion || vt == :all
        @safetestset "1D cosine advection-diffusion" begin
            include(joinpath(@__DIR__, "..", "validation", "convergence_tests", "one_dimensional_cosine_advection_diffusion.jl"))
        end
    end

    if vt == :gaussian_advection_diffusion || vt == :all
        @safetestset "1D Gaussian advection-diffusion" begin
            include(joinpath(@__DIR__, "..", "validation", "convergence_tests", "one_dimensional_gaussian_advection_diffusion.jl"))
        end
    end

    if vt == :advection_schemes || vt == :all
        @safetestset "1D advection schemes" begin
            include(joinpath(@__DIR__, "..", "validation", "convergence_tests", "one_dimensional_advection_schemes.jl"))
        end
    end

    if vt == :diffusion || vt == :all
        @safetestset "2D diffusion" begin
            include(joinpath(@__DIR__, "..", "validation", "convergence_tests", "two_dimensional_diffusion.jl"))
        end
    end

    if vt == :taylor_green || vt == :all
        @safetestset "2D Taylor-Green" begin
            include(joinpath(@__DIR__, "..", "validation", "convergence_tests", "run_taylor_green.jl"))
            include(joinpath(@__DIR__, "..", "validation", "convergence_tests", "analyze_taylor_green.jl"))
        end
    end

    if vt == :forced_free_slip || vt == :all
        @safetestset "2D forced free-slip" begin
            include(joinpath(@__DIR__, "..", "validation", "convergence_tests", "run_forced_free_slip.jl"))
            include(joinpath(@__DIR__, "..", "validation", "convergence_tests", "analyze_forced_free_slip.jl"))
        end
    end

    if vt == :forced_fixed_slip || vt == :all
        @safetestset "2D forced fixed-slip" begin
            include(joinpath(@__DIR__, "..", "validation", "convergence_tests", "run_forced_fixed_slip.jl"))
            include(joinpath(@__DIR__, "..", "validation", "convergence_tests", "analyze_forced_fixed_slip.jl"))
        end
    end
end
