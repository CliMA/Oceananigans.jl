include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))
include(joinpath(@__DIR__, "..", "setup", "yin_yang_test_utils.jl"))

using Oceananigans.DistributedComputations: Partition
using Oceananigans.MultiRegion: MultiRegionPartition, SubregionPartition, YinYangPartition, remapped_field
using Oceananigans.Utils: regions

# The panel of `grid` holding subregion `r` of `partitioned`, and the window of the panel's parent that it covers
function panel_window(field, partitioned, r)
    layout = partitioned.partition.layout
    (ox, oy) = layout[r].offsets
    Nx, Ny, Nz = size(parent(getregion(partitioned_like(field, partitioned), r)))
    return layout[r].region, (ox .+ (1:Nx), oy .+ (1:Ny), 1:Nz)
end

partitioned_like(field, grid) = Field(instantiated_location(field), grid)

# Copy the interior of the unpartitioned `field` into a new field on `grid`
function partitioned_copy(field, grid)
    copy = partitioned_like(field, grid)
    for r in regions(grid)
        panel, window = panel_window(field, grid, r)
        parent(getregion(copy, r)) .= parent(getregion(field, panel))[window...]
        Hx, Hy, _ = halo_size(grid)
        # Only the interior is copied: the halos must be filled
        parent(getregion(copy, r))[1:Hx, :, :] .= NaN
        parent(getregion(copy, r))[end-Hx+1:end, :, :] .= NaN
        parent(getregion(copy, r))[:, 1:Hy, :] .= NaN
        parent(getregion(copy, r))[:, end-Hy+1:end, :] .= NaN
    end
    return copy
end

# Every horizontal halo of the subregions of `copy` equals the corresponding node of the panels of `field`
function same_as_panels(copy, field)
    return all(regions(copy.grid)) do r
        panel, window = panel_window(field, copy.grid, r)
        Hz = halo_size(copy.grid)[3]
        ks = 1+Hz:size(parent(getregion(copy, r)), 3)-Hz
        parent(getregion(copy, r))[:, :, ks] == parent(getregion(field, panel))[window[1:2]..., ks]
    end
end

@testset "Yin-Yang subregion partitions" begin
    grid = YinYangGrid(CPU(); size=(48, 16, 2), z=(-1, 0))

    @testset "Layout" begin
        partition = SubregionPartition(YinYangPartition(), ((3, 2), (1, 2)), (54, 22))
        @test length(partition) == 8
        layout = partition.layout
        for panel in 1:2
            owned = zeros(Int, 54, 22)
            for s in layout
                s.region == panel || continue
                owned[s.offsets[1] .+ (1:s.size[1]), s.offsets[2] .+ (1:s.size[2])] .+= 1
            end
            @test all(==(1), owned)
        end
        @test summary(grid.partition) == "YinYangPartition with (1: Yin, 2: Yang)"
        @test_throws ArgumentError YinYangGrid(CPU(); size=(48, 16, 2), z=(-1, 0), partition = Partition(1, 1, 2))
    end

    for (subregions, partition, R) in (((2, 2), Partition(2, 2), 8), ((3, 1), Partition(3), 6), ((1, 3), Partition(1, 3), 6),
                                       ((1, 2), MultiRegionPartition(Partition(1), Partition(2)), 3))
        partitioned = YinYangGrid(CPU(); size=(48, 16, 2), z=(-1, 0), partition)

        @testset "Halos with $subregions subregions" begin
            @test length(partitioned) == R
            for ℓ in yin_yang_locations
                c = location_field(ℓ..., grid)
                set!(c, (λ, φ, z) -> spherical_harmonics(λ, φ) + z)
                cᵖ = partitioned_copy(c, partitioned)
                fill_halo_regions!(c)
                fill_halo_regions!(cᵖ)
                @test same_as_panels(cᵖ, c)
            end

            u, v = XFaceField(grid), YFaceField(grid)
            set_panel_components!(u, v, solid_body_velocity)
            uᵖ, vᵖ = partitioned_copy(u, partitioned), partitioned_copy(v, partitioned)
            fill_halo_regions!((u, v))
            fill_halo_regions!((uᵖ, vᵖ))
            @test same_as_panels(uᵖ, u)
            @test same_as_panels(vᵖ, v)
        end

        @testset "Reductions and remapping with $subregions subregions" begin
            c = CenterField(grid)
            set!(c, (λ, φ, z) -> 2 + spherical_harmonics(λ, φ))
            cᵖ = partitioned_copy(c, partitioned)
            fill_halo_regions!(c)
            fill_halo_regions!(cᵖ)
            @test sum(cᵖ) ≈ sum(c)
            @test maximum(cᵖ) == maximum(c)
            @test minimum(cᵖ) == minimum(c)
            @test Field(Integral(cᵖ))[1, 1, 1] ≈ Field(Integral(c))[1, 1, 1]
            @test interior(remapped_field(cᵖ)) == interior(remapped_field(c))
        end
    end
end
