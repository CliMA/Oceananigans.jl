# include("dependencies_for_runtests.jl")

using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBottom
using Oceananigans.ImmersedBoundaries: masked_length
using Statistics: mean, norm
using CUDA: @allowscalar

@testset "Masked Reductions" begin

    for arch in archs
        grid = RectilinearGrid(arch, size = (10, 10, 10), extent = (1, 1, 1))
        ibg  = ImmersedBoundaryGrid(grid, GridFittedBottom((x, y) -> - grid.Lz/2))

        fful = Field{Center, Center, Center}(grid)
        fimm = Field{Center, Center, Center}(ibg)

        @test immersed_length(fimm) == length(fimm) / 2

        fful .= 1
        fimm .= 1

        @allowscalar fimm[:, :, 4:5] .= 1e6
        @allowscalar fimm[:, :, 1:3] .= -1e4

        @test norm(fful) ≈ √2 * norm(fimm)

        for reduc in (mean, maximum, minimum, prod)
            @test reduc(fful) == reduc(fimm)

            @test all(reduc(fful, dims=1)[:, 1, 6:end] .== reduc(fimm, dims=1)[:, 1, 6:end])
            @test all(reduc(fful, dims=2)[1, :, 6:end] .== reduc(fimm, dims=2)[1, :, 6:end])
            @test all(reduc(fful, dims=3)[:, :, 1]     .== reduc(fimm, dims=3)[:, :, 1])
        end
        @test sum(fful) == sum(fimm) / 2

        @test all(sum(fful, dims=1)[:, 1, 6:end] .== sum(fimm, dims=1)[:, 1, 6:end] ./ 2)
        @test all(sum(fful, dims=2)[1, :, 6:end] .== sum(fimm, dims=2)[1, :, 6:end] ./ 2)
        @test all(sum(fful, dims=3)[:, :, 1]     .== sum(fimm, dims=3)[:, :, 1] ./ 2)
    end
end