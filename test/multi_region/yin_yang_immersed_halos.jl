include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using Oceananigans.MultiRegion: yin_yang_interpolation
using Oceananigans.Utils: getregion

@testset "Impermeable immersed vector halos" for arch in archs, LZ in (Center, Nothing)
    sphere = YinYangGrid(arch, Float32; size=(48, 16, 4), z=(-4, 0))
    grid = ImmersedBoundaryGrid(sphere, GridFittedBottom((λ, φ) -> 0))
    u = Field{Face, Center, LZ}(grid)
    v = Field{Center, Face, LZ}(grid)
    set!(u, 1)
    set!(v, 1)
    fill_halo_regions!((u, v))

    tables = on_architecture(CPU(), yin_yang_interpolation(grid))
    for region in 1:2, (field, table) in ((u, tables.face_center), (v, tables.center_face))
        data = on_architecture(CPU(), getregion(field, region).data)
        receivers = CartesianIndex.(table.receivers.i, table.receivers.j, 1)
        available = field === u ? table.receivers.i .> 1-grid.Hx : table.receivers.j .> 1-grid.Hy
        @test all(iszero, data[receivers[available]])
        @test maximum(interior(getregion(field, region))) == 1
    end
end

@testset "Only bottoms are immersed on a YinYangGrid" begin
    sphere = YinYangGrid(CPU(); size=(48, 16, 2), z=(-4, 0))
    @test_throws "GridFittedBoundary is not supported" ImmersedBoundaryGrid(sphere, GridFittedBoundary((λ, φ, z) -> z < -2))
end
