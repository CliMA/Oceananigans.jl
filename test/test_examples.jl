examplespath = normpath(joinpath(@__FILE__, "..", "..", "examples"))

function run_example(examplename)
    println("    Running example $examplename")
    include(joinpath(examplespath, examplename))
    return true
end

@testset "Examples" begin
    @test run_example("deepening_mixed_layer.jl")
end
