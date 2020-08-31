using Pkg

const CONVERGENCE_DIR = joinpath(@__DIR__, "..", "verification", "convergence_tests")

Pkg.activate(CONVERGENCE_DIR)

@testset "Convergence" begin
    include(joinpath(CONVERGENCE_DIR, "point_exponential_decay.jl"))
end
