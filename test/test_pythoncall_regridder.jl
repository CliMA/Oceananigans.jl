include("dependencies_for_runtests.jl")

using PythonCall
using CondaPkg

@testset "PythonCall extension" begin
    xesmf = add_import_pkg("xesmf")
    @test 1 == 1
end
