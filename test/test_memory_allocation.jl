include("dependencies_for_runtests.jl")

using Oceananigans
using Oceananigans.TurbulenceClosures: CATKEVerticalDiffusivity
using Oceananigans.DistributedComputations: @handshake
using Oceananigans.Fields: flattened_unique_values
using Oceananigans.OutputReaders: extract_field_time_series, FieldTimeSeries
using Oceananigans.Utils: pretty_filesize, work_layout, interior_work_layout

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

hydrostatic_split_rk3_allocation_model(grid) = HydrostaticFreeSurfaceModel(grid;
                                                                           momentum_advection = WENOVectorInvariant(),
                                                                           tracer_advection   = WENO(),
                                                                           buoyancy           = SeawaterBuoyancy(),
                                                                           coriolis           = FPlane(f=1e-4),
                                                                           closure            = CATKEVerticalDiffusivity(),
                                                                           free_surface       = SplitExplicitFreeSurface(grid; substeps=8),
                                                                           tracers            = (:T, :S),
                                                                           timestepper        = :SplitRungeKutta3)

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

const serial_memory_cpu = Dict(
    (:hydrostatic,    :flat)            => 560,
    (:hydrostatic,    :immersed)        => 592,
    (:hydrostatic,    :active_immersed) => 640,
    (:nonhydrostatic, :flat)            => 8.6e5,
    (:nonhydrostatic, :immersed)        => 8.9e5,
    (:nonhydrostatic, :active_immersed) => 6.9e5,
)

const serial_memory_gpu = Dict(
    (:hydrostatic,    :flat)            => 4496,
    (:hydrostatic,    :immersed)        => 4544,
    (:hydrostatic,    :active_immersed) => 4544,
    (:nonhydrostatic, :flat)            => 601760,
    (:nonhydrostatic, :immersed)        => 654512,
    (:nonhydrostatic, :active_immersed) => 716144,
)

const distributed_memory_cpu = Dict(
    (:hydrostatic,    :flat)            => 7.5e5,
    (:hydrostatic,    :immersed)        => 8.7e5,
    (:hydrostatic,    :active_immersed) => 9.1e5,
    (:nonhydrostatic, :flat)            => 6.6e6,
    (:nonhydrostatic, :immersed)        => 8.0e6,
    (:nonhydrostatic, :active_immersed) => 1.0e7,
)

const distributed_memory_gpu = Dict(
    (:hydrostatic,    :flat)            => 8.8e6,
    (:hydrostatic,    :immersed)        => 1.0e7,
    (:hydrostatic,    :active_immersed) => 1.2e7,
    (:nonhydrostatic, :flat)            => 8.4e6,
    (:nonhydrostatic, :immersed)        => 9.5e6,
    (:nonhydrostatic, :active_immersed) => 1.2e7,
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
    @test flattened_unique_values((u = u, v = c, w = u, d = deepcopy(u))) == (u, c, u)
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

                    @test (@inferred work_layout(grid, Val(:xyz), ())) isa Tuple
                    @test (@inferred interior_work_layout(grid, Val(:xyz), (Center(), Center(), Center()))) isa Tuple

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

# The `SplitRungeKutta3` timestepper takes 3 substeps per time step, so we allow 3× the single-step hydrostatic bound.
if !mpi_test
    @testset "SplitRungeKutta3 hydrostatic memory allocations" begin
        for arch in archs
            baseline = arch isa GPU ? serial_memory_gpu : serial_memory_cpu
            for immersed in (:flat, :immersed, :active_immersed)
                grid  = allocation_grid(arch; immersed_mode=immersed, size=(48, 48, 8))
                model = hydrostatic_split_rk3_allocation_model(grid)
                allocations = time_step_allocations(model, 1.0)
                bound = ceil(Int, 1.1 * 3 * baseline[(:hydrostatic, immersed)])
                @info "  hydrostatic_split_rk3 | $(summary(arch)) | $immersed : $(pretty_filesize(allocations)) (≤ $(pretty_filesize(bound)))"
                @test allocations ≤ bound
            end
        end
    end
end
