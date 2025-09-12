include("dependencies_for_runtests.jl")

using PythonCall
using CondaPkg

@testset "PythonCall extension" begin
    tg = TripolarGrid(size=(360, 170, 1), z=(0, 1))
    ll = LatitudeLongitudeGrid(size=(360, 180, 1), longitude=(0, 360), latitude=(-90, 90), z=(0, 1))

    ctg = CenterField(tg)
    cll = CenterField(ll)

    W = Oceananigans.Fields.regridding_weights(cll, ctg)
    @test W isa SparseMatrixCSC
end
