# include("dependencies_for_runtests.jl")

using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBoundary
using Oceananigans.ImmersedBoundaries: conditional_length
using Statistics: mean, mean!, norm
using CUDA: @allowscalar

@testset "Field broadcasting" begin
    for arch in archs
        @info "    Testing Reductions on Immersed fields"

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

        for reduc in (mean, maximum, minimum)
            @test reduc(fful) == reduc(fimm)
            @test all(Array(interior(reduc(fful, dims=1)) .== interior(reduc(fimm, dims=1))))
        end
        @test sum(fful) == sum(fimm) * 2
        @test all(Array(interior(sum(fful, dims=1)) .== interior(sum(fimm, dims=1)) .* 2))

        @test prod(fful) == prod(fimm) * 8
        @test all(Array(interior(prod(fful, dims=1)) .== interior(prod(fimm, dims=1)) .* 8))
    
        @info "    Testing Reductions in Standard fields"
        
        fcon = Field{Center, Center, Center}(grid)
        
        fcon .= 2

        @allowscalar fcon[1, :, :] .= 1e6
        @allowscalar fcon[2, :, :] .= -1e4
        @allowscalar fcon[3, :, :] .= -12.5

        @test norm(fful) ≈ √2 * norm(fcon, condition = (i, j, k, x, y) -> i > 3) 

        for reduc in (mean, maximum, minimum)
            @test reduc(fful) == reduc(fcon, condition = (i, j, k, x, y) -> i > 3)
            @test all(Array(interior(reduc(fful, dims=1)) .== interior(reduc(fcon, condition = (i, j, k, x, y) -> i > 3, dims=1))))
        end
        @test sum(fful) == sum(fcon, condition = (i, j, k, x, y) -> i > 3) * 2
        @test all(Array(interior(sum(fful, dims=1)) .== interior(sum(fcon, condition = (i, j, k, x, y) -> i > 3, dims=1)) .* 2))

        @test prod(fful) == prod(fcon, condition = (i, j, k, x, y) -> i > 3) * 8
        @test all(Array(interior(prod(fful, dims=1)) .== interior(prod(fcon, condition = (i, j, k, x, y) -> i > 3, dims=1)) .* 8))
    
        @info "    Testing in-place conditional reductions"
    
        red = Field{Nothing, Center, Center}(grid)
        for (reduc, reduc!) in zip((mean, maximum, minimum, sum, prod), (mean!, maximum!, minimum!, sum!, prod!))
            @test reduc!(red, fimm)[1, 1 , 1] == reduc(fcon, condition = (i, j, k, x, y) -> i > 3, dims = 1)[1, 1, 1]
        end
    end
end