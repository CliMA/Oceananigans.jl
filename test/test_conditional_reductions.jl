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
        @test mean(fful) ≈      mean(fimm)

        for reduc in (maximum, minimum)
            @test reduc(fful) == reduc(fimm)
            @test all(Array(interior(reduc(fful, dims=1)) .== interior(reduc(fimm, dims=1))))
        end
        @test sum(fful) == sum(fimm) * 2
        @test all(Array(interior(sum(fful, dims=1)) .== interior(sum(fimm, dims=1)) .* 2))

        @test prod(fful) == prod(fimm) * 8
        @test all(Array(interior(prod(fful, dims=1)) .== interior(prod(fimm, dims=1)) .* 8))
    end

    @info "    Testing Reductions in Standard fields"
    for arch in archs
        grid = RectilinearGrid(arch, size = (6, 1, 1), extent = (1, 1, 1))

        fimm = Field{Center, Center, Center}(grid)
        fful = Field{Center, Center, Center}(grid)
        
        fful .= 2
        fimm .= 2

        @allowscalar fimm[1, :, :] .= 1e6
        @allowscalar fimm[2, :, :] .= -1e4
        @allowscalar fimm[3, :, :] .= -12.5

        @test norm(fful) ≈ √2 * norm(fimm, condition = (i, j, k, x, y) -> i > 3) 
        @test mean(fful) ≈      mean(fimm, condition = (i, j, k, x, y) -> i > 3) 

        for reduc in (maximum, minimum)
            @test reduc(fful) == reduc(fimm, condition = (i, j, k, x, y) -> i > 3)
            @test all(Array(interior(reduc(fful, dims=1)) .== interior(reduc(fimm, condition = (i, j, k, x, y) -> i > 3, dims=1))))
        end
        @test sum(fful) == sum(fimm, condition = (i, j, k, x, y) -> i > 3) * 2
        @test all(Array(interior(sum(fful, dims=1)) .== interior(sum(fimm, condition = (i, j, k, x, y) -> i > 3, dims=1)) .* 2))

        @test prod(fful) == prod(fimm, condition = (i, j, k, x, y) -> i > 3) * 8
        @test all(Array(interior(prod(fful, dims=1)) .== interior(prod(fimm, condition = (i, j, k, x, y) -> i > 3, dims=1)) .* 8))
    end
end