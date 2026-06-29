include("dependencies_for_runtests.jl")

using Oceananigans
using Oceananigans.TurbulenceClosures: CATKEVerticalDiffusivity
using Oceananigans.DistributedComputations: @handshake
using Oceananigans.Utils: pretty_filesize

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

const serial_memory = Dict(
    (:hydrostatic,    :flat)            => 2e10,
    (:hydrostatic,    :immersed)        => 2e10,
    (:hydrostatic,    :active_immersed) => 2e10,
    (:nonhydrostatic, :flat)            => 2e10,
    (:nonhydrostatic, :immersed)        => 2e10,
    (:nonhydrostatic, :active_immersed) => 2e10,
)

const distributed_memory = Dict(
    (:hydrostatic,    :flat)            => 2e10,
    (:hydrostatic,    :immersed)        => 2e10,
    (:hydrostatic,    :active_immersed) => 2e10,
    (:nonhydrostatic, :flat)            => 2e10,
    (:nonhydrostatic, :immersed)        => 2e10,
    (:nonhydrostatic, :active_immersed) => 2e10,
)

# For distributed this includes only (4, 1), (1, 4) and (2, 2)
archs = nonhydrostatic_regression_test_architectures()

@testset "Memory allocation regression tests" begin
    for arch in archs
        @testset "Testing time-stepping memory allocations [$(summary(arch))]..." begin
            for (name, (build, Δt)) in pairs(Models)
                for immersed in (:flat, :immersed, :active_immersed)
                    grid  = allocation_grid(arch; immersed_mode=immersed, size=(48, 48, 8))
                    model = build(grid)
                    allocations = time_step_allocations(model, Δt)
                    baseline = arch isa Distributed ? distributed_memory : serial_memory
                    bound = ceil(Int, 1.1 * baseline[(name, immersed)])
                    @handshake @info "  $name | $(summary(arch)) | $immersed : $(pretty_filesize(allocations)) (≤ $(pretty_filesize(bound)))"
                    @test allocations ≤ bound
                end
            end
        end
    end
end
