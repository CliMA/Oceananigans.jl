include("dependencies_for_runtests.jl")

using XESMF
using SparseArrays
using LinearAlgebra

z = (-1, 0)
southernmost_latitude = -80
radius = Oceananigans.Grids.R_Earth

llg_coarse = LatitudeLongitudeGrid(; size=(176, 88, 1),
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
                                 (tg, llg_fine))

        @info "  Regridding from $(nameof(typeof(src_grid))) to $(nameof(typeof(dst_grid)))"

        src_field = CenterField(src_grid)
        dst_field = CenterField(dst_grid)

        λ₀, φ₀ = 150, 30.  # degrees
        width = 12         # degrees
        set!(src_field, (λ, φ, z) -> exp(-((λ - λ₀)^2 + (φ - φ₀)^2) / 2width^2))

        regridder = XESMF.Regridder(dst_field, src_field)
        @test regridder.weights isa SparseMatrixCSC

        regrid!(dst_field, regridder, src_field)

        # ∫ dst_field dA ≈ ∫ src_field dA
        @test isapprox(first(Field(Integral(dst_field, dims=(1, 2)))),
                       first(Field(Integral(src_field, dims=(1, 2)))), rtol=1e-4)
    end
end
