# include("dependencies_for_runtests.jl")

using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBoundary
using Oceananigans.ImmersedBoundaries: conditional_length
using Statistics: mean, norm
using CUDA: @allowscalar

@testset "Field broadcasting" begin
    @info "    Testing Reductions on Immersed fields"
    for arch in archs
        grid = RectilinearGrid(arch, size = (6, 1, 1), extent = (1, 1, 1))
        ibg  = ImmersedBoundaryGrid(grid, GridFittedBoundary((x, y, z) -> (x < 0.5)))

        fful = Field{Center, Center, Center}(grid)
        fimm = Field{Center, Center, Center}(ibg)

        @test conditional_length(fimm) == length(fimm) / 2

        fful .= 2
        fimm .= 2

        @allowscalar fimm[1, :, :] .= 1e6
        @allowscalar fimm[2, :, :] .= -1e4
        @allowscalar fimm[3, :, :] .= -12.5

        @test norm(fful) ≈ √2 * norm(fimm)

        for reduc in (mean, maximum, minimum)
            @show reduc
            @show @test reduc(fful) == reduc(fimm)
            @show @test all(reduc(fful, dims=1)[1, :, :] .== reduc(fimm, dims=1)[1, :, :])
        end
        @test sum(fful) == sum(fimm) * 2
        @test all(sum(fful, dims=1)[1, :, :] .== sum(fimm, dims=1)[1, :, :] .* 2)

        @test prod(fful) == prod(fimm) * 8
        @test all(prod(fful, dims=1)[1, :, :] .== prod(fimm, dims=1)[1, :, :] .* 8)
    end
end