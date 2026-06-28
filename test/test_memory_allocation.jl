include("dependencies_for_runtests.jl")

using Oceananigans
using Oceananigans.TurbulenceClosures: CATKEVerticalDiffusivity
using Oceananigans.DistributedComputations: @handshake
using Oceananigans.Utils: pretty_filesize

function allocation_grid(arch, FT=Float64; immersed_mode, size, extent=(1, 1, 1), halo=(7, 7, 7), topology=(Periodic, Periodic, Bounded))

    grid = RectilinearGrid(arch, FT; size, extent, halo, topology)
    immersed_mode == :flat && return grid

    bump(x, y) = - extent[3] * (1 - convert(FT, 0.3) * exp(-((x - extent[1]/2)^2 + (y - extent[2]/2)^2)))
    return ImmersedBoundaryGrid(grid, GridFittedBottom(bump); active_cells_map = immersed_mode == :active_immersed)
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
    # Compile first
    time_step!(model, Δt)
    time_step!(model, Δt)
    time_step!(model, Δt)
    time_step!(model, Δt)
    time_step!(model, Δt)

    # Then measure allocation
    return minimum(@allocated(time_step!(model, convert(eltype(model.grid), Δt))) for _ in 1:samples)
end

const serial_memory = Dict(
    (:hydrostatic,    :flat)            => 22000,
    (:hydrostatic,    :immersed)        => 19000,
    (:hydrostatic,    :active_immersed) => 25000,
    (:nonhydrostatic, :flat)            => 1031984,
    (:nonhydrostatic, :immersed)        => 2011088,
    (:nonhydrostatic, :active_immersed) => 2011088,
)

const distributed_memory = Dict(
    (:hydrostatic,    :flat)            => 50000,
    (:hydrostatic,    :immersed)        => 60000,
    (:hydrostatic,    :active_immersed) => 45000,
    (:nonhydrostatic, :flat)            => 1031984,
    (:nonhydrostatic, :immersed)        => 2011088,
    (:nonhydrostatic, :active_immersed) => 2011088,
)

@testset "Memory allocation regression tests" begin
    for arch in archs
        @testset "Testing time-stepping memory allocations [$(summary(arch))]..." begin
            for (name, (build, Δt)) in pairs(Models)
                for immersed in (:flat, :immersed, :active_immersed)
                    grid  = allocation_grid(arch; immersed_mode=immersed, size=(32, 32, 8))
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
