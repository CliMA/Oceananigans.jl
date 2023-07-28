include("dependencies_for_runtests.jl")

using Oceananigans.MultiRegion
using Oceananigans.MultiRegion: reconstruct_global_grid, reconstruct_global_field, getnamewrapper
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBottom, GridFittedBoundary

devices(::CPU, num) = nothing
devices(::GPU, num) = Tuple(0 for i in 1:num)

@testset "Testing multi region grids" begin
    for arch in archs

        region_num   = [2, 4, 5]
        partitioning = [XPartition]
        @warn "Need to add tests for YPartition"

        grids = [LatitudeLongitudeGrid(arch, size = (20, 20, 1), latitude = (-80, 80), longitude = collect(range(-180, 180, length=21)), z = (0, 1)),
                 RectilinearGrid(arch, size = (20, 20, 1), x = (0, 1), y = collect(range(0, 1, length=21)), z = (0, 1))]

        immersed_boundaries = [GridFittedBottom((x, y)->0.5),
                               GridFittedBottom(arch_array(arch, [0.5 for i in 1:20, j in 1:20])),
                               GridFittedBoundary((x, y, z)->z>0.5),
                               GridFittedBoundary(arch_array(arch, [false for i in 1:20, j in 1:20, k in 1:1]))]
        
        for grid in grids, P in partitioning, regions in region_num
            @info "Testing multi region $(getnamewrapper(grid)) on $regions $(P)s"
            mrg = MultiRegionGrid(grid, partition = P(regions), devices = devices(arch, regions))

            @test reconstruct_global_grid(mrg) == grid

            for FieldType in [CenterField, XFaceField, YFaceField]
                @info "Testing multi region $(FieldType) on $(getnamewrapper(grid)) on $regions $(P)s"

                par_field = FieldType(mrg)
                ser_field = FieldType(grid)

                set!(ser_field, (x, y, z) -> x)
                @apply_regionally set!(par_field, (x, y, z) -> x)

                fill_halo_regions!(ser_field)
                fill_halo_regions!(par_field)

                rec_field = reconstruct_global_field(par_field)

                @test all(Array(rec_field.data.parent) .≈ Array(ser_field.data.parent))
            end

            for immersed_boundary in immersed_boundaries
                @info "Testing multi region immersed boundaries on $(getnamewrapper(grid)) on $regions $(P)s"
                ibg = ImmersedBoundaryGrid(grid, immersed_boundary)
                mrg = MultiRegionGrid(ibg, partition = P(regions), devices = devices(arch, regions))

                @test on_architecture(arch, reconstruct_global_grid(mrg)) == ibg
            end
        end
    end
end
