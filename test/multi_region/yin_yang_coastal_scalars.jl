include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using Oceananigans.MultiRegion: yin_yang_interpolation
using Oceananigans.ImmersedBoundaries: mask_immersed_field!
using Oceananigans.Utils: getregion

@testset "Uniform wet tracer across Yin-Yang coastlines" for arch in archs, stencil_width in (2, 4)
    sphere = YinYangGrid(arch, Float32; size=(48, 16, 4), z=(-4, 0), stencil_width)
    grid = ImmersedBoundaryGrid(sphere, GridFittedBottom((λ, φ) -> ifelse(φ > 0, 0, -4)))
    tracer = CenterField(grid)
    set!(tracer, 1)
    for region in 1:2
        mask_immersed_field!(getregion(tracer, region))
    end
    fill_halo_regions!(tracer)

    table = on_architecture(CPU(), yin_yang_interpolation(grid).center_center)
    for region in 1:2
        regional_grid = getregion(grid, region)
        bottom = on_architecture(CPU(), regional_grid.immersed_boundary.bottom_height)
        bottom_receivers = CartesianIndex.(table.receivers.i, table.receivers.j, 1)
        wet = bottom[bottom_receivers] .< -1/2
        tracer_receivers = CartesianIndex.(table.receivers.i, table.receivers.j, grid.Nz)
        data = on_architecture(CPU(), getregion(tracer, region).data)
        @test any(wet)
        @test all(value -> isapprox(value, 1; atol=2f-6), data[tracer_receivers[wet]])
    end
end
