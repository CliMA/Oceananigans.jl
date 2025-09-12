include("dependencies_for_runtests.jl")

using PythonCall

@testset "PythonCall extension" begin
    @test 1 == 1
end
