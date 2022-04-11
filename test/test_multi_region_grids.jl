using Oceananigans.MultiRegion
using Oceananigans.MultiRegion: reconstruct_global_grid, getname
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBottom, GridFittedBoundary

devices(::CPU, num) = nothing
devices(::GPU, num) = Tuple(0 for i in 1:num)

@testset "Testing multi region grids" begin
    for arch in archs
        grids = [LatitudeLongitudeGrid(arch, size=(20, 20, 1), latitude=collect(range(-80, 80, length=21)), 
                                                            longitude=(-180, 180), z=(0, 1)),
                RectilinearGrid(arch, size=(20, 20, 1), x=(0, 1), y=collect(range(0, 1, length=21)), z=(0, 1))]

        immersed_boundaries = [GridFittedBottom((x, y)->0.5),
                               GridFittedBottom([0.5 for i in 1:20, j in 1:20]),
                               GridFittedBoundary((x, y, z)->z>0.5),
                               GridFittedBoundary([false for i in 1:20, j in 1:20, k in 1:1])]
        
        for grid in grids, P in [XPartition, YPartition], regions in [2, 4, 5, 10]
            @info "Testing multi region $(getname(grid)) on $regions $P"
            mrg = MultiRegionGrid(grid, partition = P(regions), devices = devices(arch, regions))
            @test reconstruct_global_grid(mrg) == grid

            for immersed_boundary in immersed_boundaries
                ibg = ImmersedBoundaryGrid(grid, immersed_boundary)
                mrg = MultiRegionGrid(ibg, partition = P(regions), devices = devices(arch, regions))

                @test reconstruct_global_grid(mrg) == ibg
            end
        end
    end
end