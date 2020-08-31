using Pkg

const CONVERGENCE_DIR = joinpath(@__DIR__, "..", "verification", "convergence_tests")

Pkg.activate(CONVERGENCE_DIR)
Pkg.instantiate()
Pkg.develop(path=joinpath(@__DIR__, ".."))

@testset "Convergence" begin
    include(joinpath(CONVERGENCE_DIR, "point_exponential_decay.jl"))
    include(joinpath(CONVERGENCE_DIR, "one_dimensional_cosine_advection_diffusion.jl"))
    include(joinpath(CONVERGENCE_DIR, "one_dimensional_gaussian_advection_diffusion.jl"))
    include(joinpath(CONVERGENCE_DIR, "two_dimensional_diffusion.jl"))
    include(joinpath(CONVERGENCE_DIR, "run_taylor_green.jl"))
    include(joinpath(CONVERGENCE_DIR, "analyze_taylor_green.jl"))
    include(joinpath(CONVERGENCE_DIR, "run_forced_free_slip.jl"))
    include(joinpath(CONVERGENCE_DIR, "analyze_forced_free_slip.jl"))
    include(joinpath(CONVERGENCE_DIR, "run_forced_fixed_slip.jl"))
    include(joinpath(CONVERGENCE_DIR, "analyze_forced_fixed_slip.jl"))
end
