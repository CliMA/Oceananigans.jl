include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using Oceananigans.MultiRegion: MultiRegionGrids, number_of_regions
using Oceananigans.Grids: xnode, znode

quiet_multi_region_grid(grid, partition) =
    @test_logs (:warn, r"^MultiRegion functionalities are experimental") MultiRegionGrid(grid; partition)

partition_dimension(grid) = grid.partition isa XPartition ? 1 : 2

# Interior data of `r` on the single-region grid: regional results are concatenated along the partitioned dimension
function global_interior(r)
    r.grid isa MultiRegionGrids || return Array(interior(r))
    regional_interiors = [Array(interior(getregion(r, region))) for region in 1:number_of_regions(r.grid)]
    return cat(regional_interiors...; dims=partition_dimension(r.grid))
end

# Split a global array into a `MultiRegionObject` of regional arrays
function regional_arrays(array, mrf)
    mrg = mrf.grid
    dim = partition_dimension(mrg)
    regional_sizes = [size(getregion(mrf, region), dim) for region in 1:number_of_regions(mrg)]
    last_indices = cumsum(regional_sizes)
    first_indices = last_indices .- regional_sizes .+ 1
    return MultiRegionObject(Tuple(copy(selectdim(array, dim, i:j)) for (i, j) in zip(first_indices, last_indices)))
end

reduction_test_function(x, y, z) = 1 + x / 7 + (y / 5)^2 + z * x / 11

# A condition that depends on the regional grid, so it is only correct if it is evaluated region by region
eastern_upper(i, j, k, grid, operand) = (xnode(i, j, k, grid, Center(), Center(), Center()) > 0.3) & (k > 1)

function test_reduction_matches(reduction, single_region_field, multi_region_field; kwargs...)
    single_region_result = reduction(single_region_field; kwargs...)
    multi_region_result = reduction(multi_region_field; kwargs...)

    if single_region_result isa Number
        return isapprox(multi_region_result, single_region_result; nans=true)
    else
        return isapprox(global_interior(multi_region_result), Array(interior(single_region_result)); nans=true)
    end
end

# Grids stretched in the horizontal, so that the spacings at the seams between regions matter
function reduction_test_grids(arch)
    z = [-(1 - k / 4)^2 for k in 0:4]

    rectilinear_grid = RectilinearGrid(arch, size = (12, 8, 4),
                                       x = [(i / 12)^1.3 for i in 0:12],
                                       y = [(j / 8)^1.5 for j in 0:8],
                                       z = z,
                                       topology = (Bounded, Bounded, Bounded))

    lat_lon_grid = LatitudeLongitudeGrid(arch, size = (12, 8, 4),
                                         longitude = [-180 + 360 * (i / 12)^1.2 for i in 0:12],
                                         latitude = [-60 + 120 * (j / 8)^1.5 for j in 0:8],
                                         z = z)

    return (rectilinear_grid, lat_lon_grid)
end

# Uneven partitions are only supported in x
function reduction_test_cases(arch)
    rectilinear_grid, lat_lon_grid = reduction_test_grids(arch)
    return ((rectilinear_grid, XPartition([4, 8])), (lat_lon_grid, YPartition(2)))
end

@testset "Multi region reductions [$(summary(arch))]" for arch in archs
    for (grid, partition) in reduction_test_cases(arch)
        mrg = quiet_multi_region_grid(grid, partition)
        partitioned_dimension = partition_dimension(mrg)

        c = CenterField(grid)
        m = CenterField(mrg)
        set!(c, reduction_test_function)
        set!(m, reduction_test_function)

        @testset "Reductions on $(getnamewrapper(grid)) with $(summary(partition))" begin
            # dims = 3 is reduced region by region, the others combine regional results
            for reduction in (sum, mean, maximum), dims in (:, 3, partitioned_dimension, (1, 2))
                @test test_reduction_matches(reduction, c, m; dims)
            end

            @test test_reduction_matches(c -> any(>(2), c), c, m)
            @test test_reduction_matches(c -> all(>(0), c), c, m)
            @test test_reduction_matches(c -> sum(abs2, c), c, m)
            @test all(extrema(m) .≈ extrema(c))

            # Integrals of face fields are checked across the partition only, since the metrics of the faces at
            # the seams depend on the spacings of halo cells that each region extrapolates from its own cells
            face_dimension = 3 - partitioned_dimension
            cᶠ = face_dimension == 1 ? XFaceField(grid) : YFaceField(grid)
            mᶠ = face_dimension == 1 ? XFaceField(mrg) : YFaceField(mrg)
            set!(cᶠ, reduction_test_function)
            set!(mᶠ, reduction_test_function)
            @test test_reduction_matches(sum, cᶠ, mᶠ)
            @test test_reduction_matches(maximum, cᶠ, mᶠ; dims=partitioned_dimension)
            @test global_interior(Field(Integral(mᶠ; dims=partitioned_dimension))) ≈ Array(interior(Field(Integral(cᶠ; dims=partitioned_dimension))))
        end

        @testset "Conditional reductions on $(getnamewrapper(grid)) with $(summary(partition))" begin
            # Conditions can exclude whole columns or levels, whose means are NaN
            threshold = mean(c)
            global_condition = interior(c) .> threshold

            single_region_conditions = (eastern_upper, global_condition, global_condition, global_condition)
            multi_region_conditions = (eastern_upper, global_condition, regional_arrays(global_condition, m), m > threshold)

            for (single_region_condition, multi_region_condition) in zip(single_region_conditions, multi_region_conditions)
                @test sum(m; condition=multi_region_condition) ≈ sum(c; condition=single_region_condition)
                @test isapprox(global_interior(mean(m; condition=multi_region_condition, dims=(1, 2))),
                               Array(interior(mean(c; condition=single_region_condition, dims=(1, 2)))); nans=true)
            end

            ∫c = Field(Integral(c; condition=eastern_upper, dims=3))
            ∫m = Field(Integral(m; condition=eastern_upper, dims=3))
            @test global_interior(∫m) ≈ Array(interior(∫c))

            # The mask replaces the points that do not satisfy the condition
            @test sum(m; condition=eastern_upper, mask=1) ≈ sum(c; condition=eastern_upper, mask=1)
        end

        @testset "Scans on $(getnamewrapper(grid)) with $(summary(partition))" begin
            for (Reduction, dims) in ((Integral, (1, 2, 3)), (Integral, partitioned_dimension), (Integral, 3), (Average, (1, 2)))
                single_region_result = Field(Reduction(c; dims))
                multi_region_result = Field(Reduction(m; dims))
                @test global_interior(multi_region_result) ≈ Array(interior(single_region_result))

                # Reductions spanning the partition live on the grid of the first region
                @test (multi_region_result.grid isa MultiRegionGrids) == (partitioned_dimension ∉ dims)
            end

            # Recomputing gives the same result
            ∫m = Field(Integral(m; dims=(1, 2)))
            ∫c = Field(Integral(c; dims=(1, 2)))
            set!(m, (x, y, z) -> 2 * reduction_test_function(x, y, z))
            set!(c, (x, y, z) -> 2 * reduction_test_function(x, y, z))
            compute!(∫m)
            compute!(∫c)
            @test Array(interior(∫m)) ≈ Array(interior(∫c))

            # Weighted reductions of binary operations between multi-region fields
            w = CenterField(grid)
            wm = CenterField(mrg)
            set!(w, (x, y, z) -> 1 + z^2)
            set!(wm, (x, y, z) -> 1 + z^2)
            @test sum(wm * m) ≈ sum(w * c)
            @test Array(interior(Field(Integral(wm * m)))) ≈ Array(interior(Field(Integral(w * c))))

            # Accumulations along an unpartitioned dimension are computed region by region
            @test global_interior(Field(Accumulation(cumsum!, m, dims=3))) ≈ Array(interior(Field(Accumulation(cumsum!, c, dims=3))))
            @test_throws ArgumentError Field(Accumulation(cumsum!, m, dims=partitioned_dimension))
        end

        @testset "In-place reductions on $(getnamewrapper(grid)) with $(summary(partition))" begin
            for reduction! in (sum!, maximum!)
                single_region_column = Field{Nothing, Nothing, Center}(grid)
                reduction!(single_region_column, c)

                # A column on the grid of the first region holds the combined result
                column = Field{Nothing, Nothing, Center}(getregion(mrg, 1))
                fill!(column, NaN)
                @test reduction!(column, m) === column
                @test Array(interior(column)) ≈ Array(interior(single_region_column))

                # A column on the multi-region grid holds the combined result in every region
                multi_region_column = Field{Nothing, Nothing, Center}(mrg)
                fill!(multi_region_column, NaN)
                reduction!(multi_region_column, m)
                for region in 1:number_of_regions(mrg)
                    @test Array(interior(getregion(multi_region_column, region))) ≈ Array(interior(single_region_column))
                end

                # Reductions that do not span the partition are computed region by region
                single_region_surface = Field{Center, Center, Nothing}(grid)
                multi_region_surface = Field{Center, Center, Nothing}(mrg)
                reduction!(single_region_surface, c)
                reduction!(multi_region_surface, m)
                @test global_interior(multi_region_surface) ≈ Array(interior(single_region_surface))
            end

            single_region_column = Field{Nothing, Nothing, Center}(grid)
            column = Field{Nothing, Nothing, Center}(getregion(mrg, 1))
            mean!(single_region_column, c; condition=eastern_upper)
            mean!(column, m; condition=eastern_upper)
            @test isapprox(Array(interior(column)), Array(interior(single_region_column)); nans=true)

            # `init=false` accumulates on top of the existing values
            sum!(single_region_column, c)
            sum!(column, m)
            sum!(column, m; init=false)
            @test Array(interior(column)) ≈ 2 .* Array(interior(single_region_column))

            # A single-region result requires a reduction that spans the partition
            @test_throws ArgumentError sum!(Field{Center, Center, Nothing}(getregion(mrg, 1)), m)
        end
    end
end

@testset "Multi region reductions on immersed grids [$(summary(arch))]" for arch in archs
    grid = first(reduction_test_grids(arch))
    bottom(x, y) = -0.5 - 0.3x
    ibg = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom))
    mrg = quiet_multi_region_grid(grid, XPartition(2))
    mribg = ImmersedBoundaryGrid(mrg, GridFittedBottom(bottom))

    c = CenterField(ibg)
    m = CenterField(mribg)
    set!(c, reduction_test_function)
    set!(m, reduction_test_function)

    for reduction in (sum, mean, maximum)
        @test test_reduction_matches(reduction, c, m)
        @test test_reduction_matches(reduction, c, m; dims=(1, 2))
    end

    @test test_reduction_matches(mean, c, m; condition=eastern_upper)
    @test Array(interior(Field(Integral(m)))) ≈ Array(interior(Field(Integral(c))))
end
