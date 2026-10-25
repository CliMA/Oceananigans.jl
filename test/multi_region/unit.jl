include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using Oceananigans.Diagnostics: hasnan
using Oceananigans.Fields: dot!
using Oceananigans.Utils: getregion

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

        field_types = [CenterField, XFaceField, YFaceField]

        for grid in grids
            single_region_fields = map(field_types) do FieldType
                single_region_field = FieldType(grid)
                set!(single_region_field, (x, y, z) -> x)
                fill_halo_regions!(single_region_field)
                return single_region_field
            end

            @testset "Multi region $(nameof(typeof(grid))) on $regions $(Partition)s" for Partition in partition_types, region in regions
                mrg = MultiRegionGrid(grid, partition = Partition(region))

                @test minimum_xspacing(mrg) == minimum(minimum_xspacing(mrg[r]) for r in 1:length(mrg.region_grids))
                @test minimum_yspacing(mrg) == minimum(minimum_yspacing(mrg[r]) for r in 1:length(mrg.region_grids))
                @test minimum_zspacing(mrg) == minimum(minimum_zspacing(mrg[r]) for r in 1:length(mrg.region_grids))

                @test reconstruct_global_grid(mrg) == grid

                @testset "Multi region $(FieldType)" for (FieldType, single_region_field) in zip(field_types, single_region_fields)
                    multi_region_field = FieldType(mrg)
                    set!(multi_region_field, (x, y, z) -> x)
                    fill_halo_regions!(multi_region_field)

                    # Remember that fields are reconstructed on the CPU!!
                    reconstructed_field = reconstruct_global_field(multi_region_field)

                    @test parent(reconstructed_field) ≈ Array(parent(single_region_field))
                end

                @testset "Multi region immersed boundaries" for immersed_boundary in immersed_boundaries
                    ibg = ImmersedBoundaryGrid(grid, immersed_boundary)
                    mribg = ImmersedBoundaryGrid(mrg, immersed_boundary)

                    @test reconstruct_global_grid(mribg) == ibg
                end
            end
        end
    end
end

# A region's parent array, halos included, is a contiguous slab of the single-region parent array
# along the partitioned dimension, so comparing them checks both the seams and the outer halos.
function regional_slab(single_region_field, multi_region_field, region)
    mrg = multi_region_field.grid
    dim = mrg.partition isa XPartition ? 1 : 2
    offset = sum(size(mrg[r], dim) for r in 1:region-1; init=0)
    total = size(parent(getregion(multi_region_field, region)), dim)
    return selectdim(Array(parent(single_region_field)), dim, offset+1:offset+total)
end

@testset "Testing multi region halo filling" begin
    for arch in archs
        rectilinear_grids = [RectilinearGrid(arch, size=(12, 12, 1), x=(0, 1), y=(0, 1), z=(0, 1), topology=(TX, TY, Bounded))
                             for TX in (Periodic, Bounded), TY in (Periodic, Bounded)]

        lat_lon_grid = LatitudeLongitudeGrid(arch, size=(12, 12, 1), longitude=(-180, 180), latitude=(-60, 60), z=(0, 1))

        for grid in (rectilinear_grids..., lat_lon_grid), Partition in (XPartition, YPartition), regions in (2, 3)
            @testset "Halo filling on $(nameof(typeof(grid))) with topology $(topology(grid)) on $regions $(Partition)s on $arch" begin
                mrg = MultiRegionGrid(grid, partition = Partition(regions))

                for FieldType in (CenterField, XFaceField, YFaceField)
                    single_region_field = FieldType(grid)
                    set!(single_region_field, (x, y, z) -> x + 10y)
                    fill_halo_regions!(single_region_field)

                    multi_region_field = FieldType(mrg)
                    set!(multi_region_field, (x, y, z) -> x + 10y)
                    fill_halo_regions!(multi_region_field)

                    for region in 1:regions
                        @test Array(parent(getregion(multi_region_field, region))) ≈ regional_slab(single_region_field, multi_region_field, region)
                    end
                end
            end
        end
    end
end

struct OrderedRegions{O, R}
    objects :: O
    region_identifiers :: R
end

Oceananigans.Utils.isregional(::OrderedRegions) = true
Oceananigans.Utils.regions(ordered::OrderedRegions) = ordered.region_identifiers
Oceananigans.Utils.getregion(ordered::OrderedRegions, r) = ordered.objects[r]
Oceananigans.Utils._getregion(ordered::OrderedRegions, r) = ordered.objects[r]

@testset "apply_regionally! keeps region identifiers and order" begin
    objects = (10, 20, 30, 40)
    for region_identifiers in ([2, 1], [2, 4], [1, 2], 1:3, 1:4, Base.OneTo(3), 2:3, Iterators.filter(isodd, 1:4))
        visited = Int[]
        Oceananigans.Utils.apply_regionally!(x -> push!(visited, x), OrderedRegions(objects, region_identifiers))
        @test visited == [objects[r] for r in region_identifiers]
    end
end

@testset "Multi region halo filling with parameterized, windowed and reduced fields" begin
    for arch in archs
        grid = RectilinearGrid(arch, size=(12, 12, 4), x=(0, 1), y=(0, 1), z=(0, 1), topology=(Bounded, Periodic, Bounded))

        top_value(i, j, grid, clock, fields, p) = p.a * xnode(i, j, 1, grid, Center(), Center(), Center()) +
                                                  ynode(i, j, 1, grid, Center(), Center(), Center())
        top = ValueBoundaryCondition(top_value, discrete_form = true, parameters = (; a = 3))
        bottom = GradientBoundaryCondition(2)
        loc = (Center(), Center(), Center())
        clock = Clock(time = 0.0)

        for Partition in (XPartition, YPartition), regions in (2, 3)
            mrg = MultiRegionGrid(grid, partition = Partition(regions))

            three_dimensional_values(x, y, z) = x + 10y + 100z
            two_dimensional_values(x, y) = x + 10y

            field_cases = ((CenterField(grid; boundary_conditions = FieldBoundaryConditions(grid, loc; top, bottom)),
                            CenterField(mrg; boundary_conditions = FieldBoundaryConditions(mrg, loc; top, bottom)),
                            three_dimensional_values),
                           (CenterField(grid; indices = (:, :, 2:3)),
                            CenterField(mrg; indices = (:, :, 2:3)),
                            three_dimensional_values),
                           (Field{Center, Center, Nothing}(grid),
                            Field{Center, Center, Nothing}(mrg),
                            two_dimensional_values))

            for (single_region_field, multi_region_field, initial_values) in field_cases
                for field in (single_region_field, multi_region_field)
                    set!(field, initial_values)
                    fill_halo_regions!(field, clock, (;))
                end

                for region in 1:regions
                    @test Array(parent(getregion(multi_region_field, region))) ≈ regional_slab(single_region_field, multi_region_field, region)
                end
            end
        end
    end
end

@testset "Multi region dot products [$(summary(arch))]" for arch in archs
    grid = RectilinearGrid(arch; size=(8, 8, 4), extent=(1, 1, 1))
    mrg = MultiRegionGrid(grid; partition=XPartition(2))

    a = CenterField(mrg)
    b = CenterField(mrg)
    set!(a, 2)
    set!(b, 3)

    Ncells = prod(size(grid))
    @test dot(a, b) == 2 * 3 * Ncells
    @test norm(a) == 2 * sqrt(Ncells)

    # `dot!` sums the regional dot products into the one-element array `r`
    r = zeros(mrg, 1)
    @test dot!(r, a, b) === r
    @test Array(r)[1] == 2 * 3 * Ncells

    # The condition is evaluated on each regional grid
    western_half(i, j, k, grid, c) = i <= size(grid, 1) ÷ 2
    @test dot(a, b; condition=western_half) == 2 * 3 * Ncells ÷ 2
    @test Array(dot!(r, a, b; condition=western_half))[1] == 2 * 3 * Ncells ÷ 2
end

@testset "NaN in one region of a multi-region field [$(summary(arch))]" for arch in archs
    c = CenterField(MultiRegionGrid(RectilinearGrid(arch; size=(8, 8, 1), extent=(1, 1, 1)); partition=XPartition(2)))
    @test !hasnan(c)
    set!(getregion(c, 2), NaN)
    @test hasnan(c)
end
