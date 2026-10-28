include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using Oceananigans.Fields: indices
using Oceananigans.Utils: MultiRegionObject

# The inferred type of `indices` depends on what is already in the inference cache, so this file runs
# in a dedicated worker (see runtests.jl), and reordering the checks below can hide a regression.

single_argument_kernel(i, j, k, grid, a) = @inbounds a[i, j, k]

@testset "Inference of operation indices [$(summary(arch))]" for arch in archs
    grid = RectilinearGrid(arch; size = (4, 4, 4), extent = (1, 1, 1))
    c = CenterField(grid)
    d = CenterField(grid)
    s = Field(c; indices = (:, :, 2))
    Δz = zspacings(grid, Center(), Center(), Center())
    K(a) = KernelFunctionOperation{Center, Center, Center}(single_argument_kernel, grid, a)

    @test only(Base.return_types(indices, (typeof(c * Δz),))) == NTuple{3, Colon}

    R = c * Δz
    integrand = d * (R / sum(R; dims = 3)) / Δz
    @test only(Base.return_types(axes, (typeof(integrand),))) == NTuple{3, Base.OneTo{Int}}

    for (operation, expected_type) in ((K(K(c * Δz)), NTuple{3, Colon}),
                                       (K(c * d),     NTuple{3, Colon}),
                                       (s + s,        Tuple{Colon, Colon, UnitRange{Int}}))
        @test only(Base.return_types(indices, (typeof(operation),))) == expected_type
    end
end

@testset "Indices of multi-region operations [$(summary(arch))]" for arch in archs
    grid = RectilinearGrid(arch; size = (8, 8, 4), extent = (1, 1, 1))
    multi_region_log = (:warn, "MultiRegion functionalities are experimental: help the development by reporting bugs or non-implemented features!")
    mrg = @test_logs multi_region_log MultiRegionGrid(grid; partition = XPartition(2))
    c = CenterField(mrg)
    s = Field(c; indices = (:, :, 2))
    Δz = zspacings(mrg, Center(), Center(), Center())

    for (operation, expected) in ((c * Δz, (:, :, :)), (s * c, (:, :, 2:2)), (c + s + Δz, (:, :, 2:2)))
        operation_indices = indices(operation)
        @test operation_indices isa MultiRegionObject
        @test all(==(expected), operation_indices.regional_objects)
    end
end
