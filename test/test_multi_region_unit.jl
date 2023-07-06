include("dependencies_for_runtests.jl")

using Oceananigans.MultiRegion
using Oceananigans.MultiRegion: reconstruct_global_grid, reconstruct_global_field, getnamewrapper
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBottom, GridFittedBoundary

devices(::CPU, num) = nothing
devices(::GPU, num) = Tuple(0 for i in 1:num)

@testset "Testing multi region grids" begin
    for arch in archs

        regions = [2, 4, 5]
        partition_types = [XPartition]

        lat_lon_grid = LatitudeLongitudeGrid(arch,
                                             size = (20, 20, 1),
                                             latitude = (-80, 80),
                                             longitude = collect(range(-180, 180, length=21)),
                                             z = (0, 1))

        rectilinear_grid = RectilinearGrid(arch,
                                           size = (20, 20, 1),
                                           x = (0, 1),
                                           y = collect(range(0, 1, length=21)),
                                           z = (0, 1))
                 
        grids = [lat_lon_grid, rectilinear_grid]

        immersed_boundaries = [GridFittedBottom((x, y) -> 0.5),
                               GridFittedBottom(arch_array(arch, [0.5 for i in 1:20, j in 1:20])),
                               GridFittedBoundary((x, y, z) -> z>0.5),
                               GridFittedBoundary(arch_array(arch, [false for i in 1:20, j in 1:20, k in 1:1]))]
        
        for grid in grids, Partition in partition_types, region in regions
            @info "Testing multi region $(getnamewrapper(grid)) on $regions $(Partition)s"
            mrg = MultiRegionGrid(grid, partition = Partition(region), devices = devices(arch, region))

            @test reconstruct_global_grid(mrg) == grid

            for FieldType in [CenterField, XFaceField, YFaceField]
                @info "Testing multi region $(FieldType) on $(getnamewrapper(grid)) on $regions $(Partition)s"

                multi_region_field = FieldType(mrg)
                single_region_field = FieldType(grid)

                set!(single_region_field, (x, y, z) -> x)
                @apply_regionally set!(multi_region_field, (x, y, z) -> x)

                fill_halo_regions!(single_region_field)
                fill_halo_regions!(multi_region_field)

                reconstructed_field = reconstruct_global_field(multi_region_field)

                @test parent(reconstructed_field) ≈ parent(single_region_field)
            end

            for immersed_boundary in immersed_boundaries
                @info "Testing multi region immersed boundaries on $(getnamewrapper(grid)) on $regions $(Partition)s"
                ibg = ImmersedBoundaryGrid(grid, immersed_boundary)
                mrg = MultiRegionGrid(ibg, partition = Partition(region), devices = devices(arch, region))

                @test on_architecture(arch, reconstruct_global_grid(mrg)) == ibg
            end
        end
    end
end

