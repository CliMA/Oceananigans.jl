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
matches(reduction, c, m, (cc, mc)) = isapprox(plain(reduction(m, mc)), plain(reduction(c, cc)); nans=true)

# Grids stretched in the horizontal, so that the spacings at the seams between regions matter.
# Uneven partitions are only supported in x.
function reduction_test_cases(arch)
    z = [-(1 - k / 3)^2 for k in 0:3]
    rectilinear_grid = RectilinearGrid(arch; size=(12, 6, 3), x=[(i / 12)^1.3 for i in 0:12], y=[(j / 6)^1.5 for j in 0:6], z,
                                       topology=(Bounded, Bounded, Bounded))
    lat_lon_grid = LatitudeLongitudeGrid(arch; size=(12, 6, 3), longitude=[-180 + 360 * (i / 12)^1.2 for i in 0:12],
                                         latitude=[-60 + 120 * (j / 6)^1.5 for j in 0:6], z)
    return ((rectilinear_grid, XPartition([4, 8])), (lat_lon_grid, YPartition(2)))
end

column(grid) = Field{Nothing, Nothing, Center}(grid)
surface(grid) = Field{Center, Center, Nothing}(grid)

@testset "Multi region reductions on $(nameof(typeof(grid))) with $(summary(partition)) [$(summary(arch))]" for arch in archs,
                                                                                  (grid, partition) in reduction_test_cases(arch)
    mrg = quiet_multi_region_grid(grid, partition)
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
    @test matches(f -> sum(f * f), c, m)
    @test matches(f -> Field(Integral(f * f)), c, m)

    # Face fields are only integrated across the partition: the metrics of the faces at the seams depend on
    # the spacings of halo cells that each region extrapolates from its own cells
    FaceField = pdim == 1 ? YFaceField : XFaceField
    cᶠ, mᶠ = FaceField(grid), FaceField(mrg)
    set!(cᶠ, reduction_test_function)
    set!(mᶠ, reduction_test_function)
    @test matches(sum, cᶠ, mᶠ)
    @test matches(maximum, cᶠ, mᶠ; dims=pdim)
    @test matches(f -> Field(Integral(f; dims=pdim)), cᶠ, mᶠ)

    # Conditions can exclude whole columns or levels, whose means are NaN; a mask replaces the excluded points
    threshold = mean(c)
    global_condition = interior(c) .> threshold
    regional_conditions = MultiRegionObject(Tuple(interior(getregion(m, region)) .> threshold
                                                  for region in 1:number_of_regions(mrg)))
    for conditions in ((eastern_upper, eastern_upper), (global_condition, global_condition),
                       (global_condition, regional_conditions), (global_condition, m > threshold))
        @test matches((f, condition) -> sum(f; condition), c, m, conditions)
        @test matches((f, condition) -> mean(f; condition, dims=(1, 2)), c, m, conditions)
    end
    @test matches(f -> Field(Integral(f; condition=eastern_upper, dims=3)), c, m)
    @test matches(sum, c, m; condition=eastern_upper, mask=1)

    # A column on the grid of the first region, or on the multi-region grid, holds the combined in-place result,
    # which `init=false` accumulates on top of; reductions that do not span the partition are computed region by region
    for reduction! in (sum!, maximum!)
        expected = column(grid)
        reduction!(expected, c)
        r = column(getregion(mrg, 1))
        fill!(r, NaN)
        @test reduction!(r, m) === r
        @test Array(interior(r)) ≈ Array(interior(expected))
        mr = reduction!(column(mrg), m)
        @test all(Array(interior(getregion(mr, region))) ≈ Array(interior(expected)) for region in 1:number_of_regions(mrg))
        sᶜ, sᵐ = surface(grid), surface(mrg)
        reduction!(sᶜ, c)
        reduction!(sᵐ, m)
        @test global_interior(sᵐ) ≈ Array(interior(sᶜ))
    end
    expected, r = column(grid), column(getregion(mrg, 1))
    mean!(expected, c; condition=eastern_upper)
    mean!(r, m; condition=eastern_upper)
    @test isapprox(Array(interior(r)), Array(interior(expected)); nans=true)
    sum!(expected, c)
    sum!(r, m)
    sum!(r, m; init=false)
    @test Array(interior(r)) ≈ 2 .* Array(interior(expected))
    @test_throws ArgumentError sum!(surface(getregion(mrg, 1)), m) # a single-region result must span the partition
end

@testset "Multi region reductions on immersed grids [$(summary(arch))]" for arch in archs
    grid = first(first(reduction_test_cases(arch)))
    bottom(x, y) = -0.5 - 0.3x
    c = CenterField(ImmersedBoundaryGrid(grid, GridFittedBottom(bottom)))
    m = CenterField(ImmersedBoundaryGrid(quiet_multi_region_grid(grid, XPartition(2)), GridFittedBottom(bottom)))
    set!(c, reduction_test_function)
    set!(m, reduction_test_function)
    for reduction in (sum, mean, maximum), dims in (:, (1, 2))
        @test matches(reduction, c, m; dims)
    end
    @test matches(mean, c, m; condition=eastern_upper)
    @test matches(f -> Field(Integral(f)), c, m)
end
