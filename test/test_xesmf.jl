include("dependencies_for_runtests.jl")

using XESMF
using SparseArrays
using LinearAlgebra

function regrid_conservatively!(dst_field, weights, src_field)
    @assert src_field.grid.Nz == dst_field.grid.Nz

    for k in 1:src_field.grid.Nz
        LinearAlgebra.mul!(vec(interior(dst_field, :, :, k)), weights, vec(interior(src_field, :, :, k)))
    end

    return nothing
end

z = (-1, 0)
southernmost_latitude = -30
radius = 1

llg_coarse = LatitudeLongitudeGrid(; size=(24, 24, Nz),
                                   longitude=(0, 360),
                                   latitude=(southernmost_latitude, 90),
                                   z, radius)

llg_fine = LatitudeLongitudeGrid(; size=(360, 180, 1),
                                 longitude=(0, 360),
                                 latitude=(southernmost_latitude, 90),
                                 z, radius)

tg = TripolarGrid(; size=(360, 170, 1), z, southernmost_latitude, radius)

@testset "XESMF extension" begin

    for (src_grid, dst_grid) in ((llg_coarse, llg_fine),
                                 (llg_fine, llg_coarse),
                                 (llg_coarse, tg))

        @info "  Regridding from $(nameof(typeof(src_grid))) to $(nameof(typeof(dst_grid)))"

        src_field = CenterField(src_grid)
        dst_field = CenterField(dst_grid)

        set!(src_field, 1)

        # ∫ src_field dA = 3πR²
        @test Field(Integral(src_field, dims=(1, 2)))[1, 1, 1] ≈ 3π * radius^2

        W = Oceananigans.Fields.regridding_weights(dst_field, src_field)
        @test W isa SparseMatrixCSC

        regrid_conservatively!(dst_field, W, src_field)

        # ∫ dst_field dA = ∫ src_field dA
        @test first(Field(Integral(dst_field, dims=(1, 2)))) ≈ first(Field(Integral(src_field, dims=(1, 2))))
    end
end
