include("dependencies_for_runtests.jl")

using XESMF
using SparseArrays
using LinearAlgebra

@testset "XESMF extension" begin
    Nz = 2
    z = (-1, 0)
    southernmost_latitude = -30
    radius = 3.4

    ll = LatitudeLongitudeGrid(; size=(360, 180, Nz),
                               longitude=(0, 360),
                               latitude=(southernmost_latitude, 90),
                               z, radius)

    tg = TripolarGrid(; size=(360, 170, Nz), z, southernmost_latitude, radius)

    cll = CenterField(ll)
    ctg = CenterField(tg)

    set!(cll, 1)

    # ∫ cll dA = 3πR²
    @show Field(Integral(cll, dims=(1, 2)))[1, 1, Nz]
    @test Field(Integral(cll, dims=(1, 2)))[1, 1, Nz] ≈ 3π * radius^2

    W = Oceananigans.Fields.regridding_weights(ctg, cll)

    @test W isa SparseMatrixCSC

    for k in 1:Nz
        LinearAlgebra.mul!(vec(interior(ctg, :, :, k)), W, vec(interior(cll, :, :, k)))
    end

    # ∫ ctg dA = ∫ cll dA
    @show Field(Integral(ctg, dims=(1, 2)))[1, 1, Nz]
    @test Field(Integral(ctg, dims=(1, 2)))[1, 1, Nz] ≈ 3π * radius^2
end
