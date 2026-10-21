include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using Oceananigans.MultiRegion: MultiRegionGrids, number_of_regions
using Oceananigans.Grids: xnode

quiet_multi_region_grid(grid, partition) =
    @test_logs (:warn, r"^MultiRegion functionalities are experimental") MultiRegionGrid(grid; partition)

partition_dimension(grid) = grid.partition isa XPartition ? 1 : 2

# Interior data of `r` on the single-region grid: regional results are concatenated along the partitioned dimension
function global_interior(r)
    r.grid isa MultiRegionGrids || return Array(interior(r))
    regional_interiors = [Array(interior(getregion(r, region))) for region in 1:number_of_regions(r.grid)]
    return cat(regional_interiors...; dims=partition_dimension(r.grid))
end

reduction_test_function(x, y, z) = 1 + x / 7 + (y / 5)^2 + z * x / 11

# A condition that depends on the regional grid, so it is only correct if it is evaluated region by region
eastern_upper(i, j, k, grid, operand) = (xnode(i, j, k, grid, Center(), Center(), Center()) > 0.3) & (k > 1)

# Whether the reductions of the single-region field `c` and of the multi-region field `m` agree
plain(result) = result isa Number ? result : global_interior(result)
matches(reduction, c, m; kwargs...) = isapprox(plain(reduction(m; kwargs...)), plain(reduction(c; kwargs...)); nans=true)

# Grids stretched in the horizontal, so that the spacings at the seams between regions matter.
# Uneven partitions are only supported in x.
function reduction_test_cases(arch)
    z = [-(1 - k / 3)^2 for k in 0:3]
    rectilinear_grid = RectilinearGrid(arch; size=(12, 6, 3), x=[(i / 12)^1.3 for i in 0:12], y=[(j / 6)^1.5 for j in 0:6], z,
                                       topology=(Bounded, Bounded, Bounded))
    lat_lon_grid = LatitudeLongitudeGrid(arch; size=(12, 6, 3), longitude=[-180 + 360 * (i / 12)^1.2 for i in 0:12],
                                         latitude=[-60 + 120 * (j / 6)^1.5 for j in 0:6], z)
    immerse(grid) = ImmersedBoundaryGrid(grid, GridFittedBottom((x, y) -> -0.5 - 0.3x))
    return ((rectilinear_grid, XPartition([4, 8]), identity), (lat_lon_grid, YPartition(2), identity),
            (rectilinear_grid, XPartition(2), immerse))
end

@testset "Multi region reductions on $(summary(immerse(grid))) with $(summary(partition)) [$(summary(arch))]" for arch in archs,
                                                                   (grid, partition, immerse) in reduction_test_cases(arch)
    grid, mrg = immerse(grid), immerse(quiet_multi_region_grid(grid, partition))
    pdim = partition_dimension(mrg)
    c, m = CenterField(grid), CenterField(mrg)
    set!(c, reduction_test_function)
    set!(m, reduction_test_function)

    # dims = 3 is reduced region by region, the others combine regional results
    for reduction in (sum, mean, maximum), dims in (:, 3, pdim, (1, 2))
        @test matches(reduction, c, m; dims)
    end
    @test matches(c -> any(>(2), c), c, m)
    @test matches(c -> all(>(0), c), c, m)
    @test matches(c -> sum(abs2, c), c, m)
    @test all(extrema(m) .≈ extrema(c))

    # Scans spanning the partition live on the grid of the first region
    for (Reduction, dims) in ((Integral, (1, 2, 3)), (Integral, pdim), (Integral, 3), (Average, (1, 2)))
        @test matches(f -> Field(Reduction(f; dims)), c, m)
        @test (Field(Reduction(m; dims)).grid isa MultiRegionGrids) == (pdim ∉ dims)
    end
    @test matches(f -> Field(Accumulation(cumsum!, f, dims=3)), c, m) # computed region by region
    @test_throws ArgumentError Field(Accumulation(cumsum!, m, dims=pdim))

    # The metrics of the faces at the seams depend on the spacings of halo cells that each region extrapolates
    FaceField = pdim == 1 ? YFaceField : XFaceField
    cᶠ, mᶠ = FaceField(grid), FaceField(mrg)
    set!(cᶠ, reduction_test_function)
    set!(mᶠ, reduction_test_function)
    @test matches(f -> Field(Integral(f; dims=pdim)), cᶠ, mᶠ)

    # Conditions are evaluated region by region; a mask replaces the excluded points, whose means are NaN
    @test matches(f -> sum(f; condition = f > 1.2), c, m)
    @test matches(f -> mean(f; condition = f > 1.2, dims=(1, 2)), c, m)
    @test matches(f -> Field(Integral(f; condition=eastern_upper, dims=3)), c, m)
    @test matches(sum, c, m; condition=eastern_upper, mask=1)

    # In-place means divide by the number of counted points of every column, on the grid of the first region
    expected, r = Field{Nothing, Nothing, Center}(grid), Field{Nothing, Nothing, Center}(getregion(mrg, 1))
    mean!(expected, c; condition=eastern_upper)
    @test isapprox(Array(interior(mean!(r, m; condition=eastern_upper))), Array(interior(expected)); nans=true)

    # Reduced fields exclude whole immersed columns
    η, ηᵐ = Field{Center, Center, Nothing}(grid), Field{Center, Center, Nothing}(mrg)
    set!(η, (x, y) -> 1 + x)
    set!(ηᵐ, (x, y) -> 1 + x)
    @test matches(mean, η, ηᵐ)
end
