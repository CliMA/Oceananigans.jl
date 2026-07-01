include("dependencies_for_runtests.jl")

using Oceananigans
using Oceananigans.TurbulenceClosures: CATKEVerticalDiffusivity
using Oceananigans.DistributedComputations: @handshake
using Oceananigans.Utils: pretty_filesize
using Oceananigans.Fields: flattened_unique_values
using Oceananigans.OutputReaders: extract_field_time_series, FieldTimeSeries

function allocation_grid(arch, FT=Float64; immersed_mode, size, extent=(1, 1, 1), halo=(7, 7, 7), topology=(Periodic, Periodic, Bounded))
    grid = RectilinearGrid(arch, FT; size, extent, halo, topology)
    immersed_mode == :flat && return grid
    return ImmersedBoundaryGrid(grid, GridFittedBottom((x, y) -> -0.5 ); active_cells_map = immersed_mode == :active_immersed)
end

hydrostatic_allocation_model(grid) = HydrostaticFreeSurfaceModel(grid;
                                                                 momentum_advection = WENOVectorInvariant(),
                                                                 tracer_advection   = WENO(),
                                                                 buoyancy           = SeawaterBuoyancy(),
                                                                 coriolis           = FPlane(f=1e-4),
                                                                 closure            = CATKEVerticalDiffusivity(),
                                                                 free_surface       = SplitExplicitFreeSurface(grid; substeps=8),
                                                                 tracers            = (:T, :S))

nonhydrostatic_allocation_model(grid) = NonhydrostaticModel(grid;
                                                            advection = WENO(),
                                                            coriolis = FPlane(f=1e-4),
                                                            buoyancy = BuoyancyTracer(),
                                                            tracers = :b)

const Models = (hydrostatic    = (build = hydrostatic_allocation_model,    Δt = 1.0),
                nonhydrostatic = (build = nonhydrostatic_allocation_model, Δt = 1e-3))

function time_step_allocations(model, Δt; samples=10)
    time_step!(model, Δt)
    time_step!(model, Δt)
    time_step!(model, Δt)
    time_step!(model, Δt)
    time_step!(model, Δt)

    Δt = convert(eltype(model.grid), Δt)
    return minimum(@allocated(time_step!(model, Δt)) for _ in 1:samples)
end

#TODO: Fix allocations in the nonhydrostatic model

const serial_memory_cpu = Dict(
    (:hydrostatic,    :flat)            => 1.1e6,
    (:hydrostatic,    :immersed)        => 1.1e6,
    (:hydrostatic,    :active_immersed) => 1.3e6,
    (:nonhydrostatic, :flat)            => 8.7e5,
    (:nonhydrostatic, :immersed)        => 9.8e5,
    (:nonhydrostatic, :active_immersed) => 1.1e6,
)

const serial_memory_gpu = Dict(
    (:hydrostatic,    :flat)            => 2.6e6,
    (:hydrostatic,    :immersed)        => 2.8e6,
    (:hydrostatic,    :active_immersed) => 2.9e6,
    (:nonhydrostatic, :flat)            => 1.9e6,
    (:nonhydrostatic, :immersed)        => 2.2e6,
    (:nonhydrostatic, :active_immersed) => 2.3e6,
)

const distributed_memory_cpu = Dict(
    (:hydrostatic,    :flat)            => 8.0e6,
    (:hydrostatic,    :immersed)        => 9.6e6,
    (:hydrostatic,    :active_immersed) => 1.3e7,
    (:nonhydrostatic, :flat)            => 6.6e6,
    (:nonhydrostatic, :immersed)        => 8.0e6,
    (:nonhydrostatic, :active_immersed) => 1.0e7,
)

const distributed_memory_gpu = Dict(
    (:hydrostatic,    :flat)            => 1.5e7,
    (:hydrostatic,    :immersed)        => 1.7e7,
    (:hydrostatic,    :active_immersed) => 2.2e7,
    (:nonhydrostatic, :flat)            => 1.2e7,
    (:nonhydrostatic, :immersed)        => 1.4e7,
    (:nonhydrostatic, :active_immersed) => 1.7e7,
)

# For distributed this includes only (4, 1), (1, 4) and (2, 2)
archs = nonhydrostatic_regression_test_architectures()

@testset "flattened_unique_values: correctness, inference, allocations" begin
    grid = RectilinearGrid(CPU(), size=(4, 4, 4), extent=(1, 1, 1))
    u = XFaceField(grid)
    c = CenterField(grid)

    nt = (velocities = (u = u, v = c), tracers = (T = c, extra = u))
    result = flattened_unique_values(nt)

    @test result isa Tuple
    @test length(result) == 2
    @test any(f -> f === u, result)
    @test any(f -> f === c, result)

    @test @inferred(flattened_unique_values(())) === ()
    @test @inferred(flattened_unique_values((u = u, v = c))) === (u, c)

    # De-duplication by identity: a repeated entry is dropped. The result length depends on runtime
    # identity when entries share a type, so this case is intentionally not `@inferred` (see the
    # type-stable, no-dedup `extract_field_time_series` used by `update_model_field_time_series!`).
    @test flattened_unique_values((u = u, v = c, w = u)) === (u, c)
end

@testset "extract_field_time_series: tuple convention, inference" begin
    grid = RectilinearGrid(CPU(), size=(4, 4, 4), extent=(1, 1, 1))

    plain = (a = 1, b = (2.0, "x"), c = grid)
    @test extract_field_time_series(plain) === ()
    @test @inferred(extract_field_time_series(plain)) === ()
    @test extract_field_time_series(3.0) === ()
    @test extract_field_time_series(nothing) === ()

    times = 0:0.5:2
    fts = FieldTimeSeries{Center, Center, Center}(grid, times)
    @test extract_field_time_series(fts) === (fts,)

    nested = (u = 1, deep = (grid = grid, series = fts, tag = "t"))
    got = extract_field_time_series(nested)
    @test got == (fts,)
end

@testset "Memory allocation regression tests" begin
    for arch in archs
        @testset "Testing time-stepping memory allocations [$(summary(arch))]..." begin
            for (name, (build, Δt)) in pairs(Models)
                for immersed in (:flat, :immersed, :active_immersed)
                    grid  = allocation_grid(arch; immersed_mode=immersed, size=(48, 48, 8))
                    model = build(grid)
                    allocations = time_step_allocations(model, Δt)
                    baseline = if arch isa Distributed{<:GPU}
                        distributed_memory_gpu
                    elseif arch isa Distributed{<:CPU}
                        distributed_memory_cpu
                    elseif arch isa GPU
                        serial_memory_gpu
                    else
                        serial_memory_cpu
                    end
                    bound = ceil(Int, 1.1 * baseline[(name, immersed)])
                    @handshake @info "  $name | $(summary(arch)) | $immersed : $(pretty_filesize(allocations)) (≤ $(pretty_filesize(bound)))"
                    @test allocations ≤ bound
                end
            end
        end
    end
end
