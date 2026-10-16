include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using Oceananigans.Utils: getregion

@testset "Grid-fitted Yin-Yang bathymetry halos" for arch in archs, stencil_width in (2, 4)
    faces = Float32[-4, -2, -1/2, 0]
    sphere = YinYangGrid(arch, Float32; size=(48, 16, 3), z=faces, stencil_width)
    grid = ImmersedBoundaryGrid(sphere, GridFittedBottom((λ, φ) -> ifelse(φ > 0, 0, -4)))
    for region in 1:2
        bottom = on_architecture(CPU(), getregion(grid, region).immersed_boundary.bottom_height)
        @test minimum(bottom) == first(faces)
        @test maximum(bottom) == last(faces)
        @test all(height -> height ∈ faces, bottom)
    end
end

# A partial bottom cell is at least ϵ times as tall as its full cell
valid_partial_bottom(h, faces, ϵ) = faces[1] ≤ h ≤ faces[end] &&
    all(k -> !(faces[k] ≤ h < faces[k+1]) || h ≤ faces[k+1] - ϵ * (faces[k+1] - faces[k]), 1:length(faces)-1)

@testset "Partial-cell Yin-Yang bathymetry halos" for arch in archs
    faces = Float32[-4, -2, -1/2, 0]
    ϵ = 0.2f0
    sphere = YinYangGrid(arch, Float32; size=(48, 16, 3), z=faces)
    bottom = PartialCellBottom((λ, φ) -> ifelse(φ > 0, 0, -1.9f0); minimum_fractional_cell_height=ϵ)
    grid = ImmersedBoundaryGrid(sphere, bottom)
    for region in 1:2
        heights = on_architecture(CPU(), getregion(grid, region).immersed_boundary.bottom_height)
        @test all(h -> valid_partial_bottom(h, faces, ϵ), heights)
    end
end
