include("dependencies_for_runtests.jl")

# To be extended as we find new use cases
@testset "Test @apply_regionally macro" begin
    a = 1
    b = 2
    @apply_regionally a = b + 1

    @test a == 3
    a = MultiRegionObject((1, 2, 3))
    b = MultiRegionObject((4, 5, 6))

    @apply_regionally a = b + 1
    @test a == MultiRegionObject((5, 6, 7))
end

@kernel function _test_kernel!(a)
    i, j = @index(Global, NTuple)
    @inbounds a[i, j, 1] = 1
end

@testset "Test @apply_regionally with regional kernels" begin
    for arch in archs
        grid = RectilinearGrid(arch, size=(10, 10, 1), x=(0, 1), y=(0, 1), z=(0, 1))
        mrg = MultiRegionGrid(grid, partition=XPartition(2))
        field = CenterField(mrg)

        # Configure regional kernels with different work sizes per region
        # Region 1: 5x10, Region 2: 5x10 (each region has half the x-extent)
        regional_sizes = MultiRegionObject(((5, 10), (5, 5)))
        @apply_regionally loops, sizes = configure_kernel(arch, mrg, regional_sizes, _test_kernel!)

        @test loops isa MultiRegionObject  # The loop
        @test sizes isa MultiRegionObject  # The worksize

        # Launch the configured kernels using @apply_regionally with MultiRegionObject function
        @apply_regionally loops(field)

        regional_field = getregion(field, 1)
        interior_data = interior(regional_field)
        @test all(isone, interior_data)

        regional_field = getregion(field, 2)
        interior_data = interior(regional_field)
        @test !(all(isone, interior_data))
        @test any(isone, interior_data)
    end
end

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
                               GridFittedBoundary((x, y, z) -> z>0.5)]

        for grid in grids, Partition in partition_types, region in regions
            @info "Testing multi region $(getnamewrapper(grid)) on $regions $(Partition)s"
            mrg = MultiRegionGrid(grid, partition = Partition(region))

            @test minimum_xspacing(mrg) == minimum(minimum_xspacing(mrg[r]) for r in 1:length(mrg.region_grids))
            @test minimum_yspacing(mrg) == minimum(minimum_yspacing(mrg[r]) for r in 1:length(mrg.region_grids))
            @test minimum_zspacing(mrg) == minimum(minimum_zspacing(mrg[r]) for r in 1:length(mrg.region_grids))

            @test reconstruct_global_grid(mrg) == grid

            for FieldType in [CenterField, XFaceField, YFaceField]
                @info "Testing multi region $(FieldType) on $(getnamewrapper(grid)) on $regions $(Partition)s"

                multi_region_field  = FieldType(mrg)
                single_region_field = FieldType(grid)

                set!(single_region_field, (x, y, z) -> x)
                set!(multi_region_field,  (x, y, z) -> x)

                fill_halo_regions!(single_region_field)
                fill_halo_regions!(multi_region_field)

                # Remember that fields are reconstructed on the CPU!!
                reconstructed_field = reconstruct_global_field(multi_region_field)

                @test parent(reconstructed_field) ≈ Array(parent(single_region_field))
            end

            for immersed_boundary in immersed_boundaries
                @info "Testing multi region immersed boundaries on $(getnamewrapper(grid)) on $regions $(Partition)s"
                ibg = ImmersedBoundaryGrid(grid, immersed_boundary)
                mrg = MultiRegionGrid(grid, partition = Partition(region))
                mribg = ImmersedBoundaryGrid(mrg, immersed_boundary)

                @test reconstruct_global_grid(mribg) == ibg
            end
        end
    end
end
