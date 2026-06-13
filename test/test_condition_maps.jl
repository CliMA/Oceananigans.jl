include("dependencies_for_runtests.jl")

using MPI
using Oceananigans.Models: InteriorBoundarySet, generate_condition_maps
using Oceananigans.Utils: get_active_cells_map

const N = 16
const H = 4

bottom_bump(x, y) = -1 + 0.4 * exp(-((x - 0.5) / 0.1)^2 - ((y - 0.5) / 0.1)^2)

active_advection() = (momentum = WENOVectorInvariant(order=3),
                      c        = WENO(order=3))

function check_split_consistency(ibs::InteriorBoundarySet, expected_total)
    @test length(ibs.interior) + length(ibs.boundary) == expected_total
end

@testset "Condition maps" begin
    @info "Testing condition_maps construction across grid types..."

    for arch in archs
        @testset "Serial grids" begin

            advection = active_advection()

            # condition_*_advection = false: cond = nothing for every key
            grid = RectilinearGrid(arch; size=(N, N, N), halo=(H, H, H), extent=(1, 1, 1), topology=(Bounded, Bounded, Bounded))
            cmaps = generate_condition_maps(grid, advection;
                                            condition_momentum_advection=false,
                                            condition_tracer_advection=false)
            @test isnothing(cmaps.momentum)
            @test isnothing(cmaps.c)

            # advection = nothing: cond = nothing
            cmaps = generate_condition_maps(grid, (; momentum=nothing, c=nothing);
                                            condition_momentum_advection=true,
                                            condition_tracer_advection=true)
            @test isnothing(cmaps.momentum)
            @test isnothing(cmaps.c)

            # Serial non-immersed, fully periodic: boundary is nothing
            grid = RectilinearGrid(arch; size=(N, N, N), halo=(H, H, H), extent=(1, 1, 1), topology=(Periodic, Periodic, Periodic))
            cmaps = generate_condition_maps(grid, advection;
                                            condition_momentum_advection=true,
                                            condition_tracer_advection=true)
            @test isnothing(cmaps.momentum.boundary)
            @test length(cmaps.momentum.interior) == N^3
            @test isnothing(cmaps.c.boundary)
            @test length(cmaps.c.interior) == N^3
            @test isnothing(get_active_cells_map(grid, Val(:core)))

            # Serial non-immersed, bounded: cond = InteriorBoundarySet
            grid = RectilinearGrid(arch; size=(N, N, N), halo=(H, H, H),
                                   extent=(1, 1, 1), topology=(Bounded, Bounded, Bounded))
            cmaps = generate_condition_maps(grid, advection;
                                            condition_momentum_advection=true,
                                            condition_tracer_advection=true)
            for key in (:momentum, :c)
                @test cmaps[key] isa InteriorBoundarySet
                check_split_consistency(cmaps[key], N^3)
            end
            @test isnothing(get_active_cells_map(grid, Val(:core)))

            # Serial immersed (active_cells_map = true): cond = IBS, active = Array
            underlying = RectilinearGrid(arch; size=(N, N, N), halo=(H, H, H),
                                         extent=(1, 1, 1), topology=(Bounded, Bounded, Bounded))
            ibg = ImmersedBoundaryGrid(underlying, GridFittedBottom(bottom_bump);
                                       active_cells_map=true)
            cmaps = generate_condition_maps(ibg, advection;
                                            condition_momentum_advection=true,
                                            condition_tracer_advection=true)
            active = get_active_cells_map(ibg, Val(:core))
            @test active isa AbstractArray
            for key in (:momentum, :c)
                @test cmaps[key] isa InteriorBoundarySet
                check_split_consistency(cmaps[key], length(active))
            end
        end
    end
end

#####
##### Distributed cases — spawn an MPI worker that runs the checks
#####

distributed_condition_maps_script = """
    using MPI
    MPI.Init()
    using Test

    include(raw"$(joinpath(@__DIR__, "dependencies_for_runtests.jl"))")

    using Oceananigans.Models: InteriorBoundarySet, generate_condition_maps
    using Oceananigans.Utils: get_active_cells_map
    using Oceananigans.DistributedComputations: Distributed, Partition, AsynchronousDistributed

    const N = 16
    const H = 4

    bottom_bump(x, y) = -1 + 0.4 * exp(-((x - 0.5) / 0.1)^2 - ((y - 0.5) / 0.1)^2)

    function check_split_consistency(ibs::InteriorBoundarySet, expected_total)
        @test length(ibs.interior) + length(ibs.boundary) == expected_total
    end

    advection   = (momentum = WENOVectorInvariant(order=3), c = WENO(order=3))
    region_keys = (:halo_independent_cells,
                   :west_halo_dependent_cells,
                   :east_halo_dependent_cells,
                   :south_halo_dependent_cells,
                   :north_halo_dependent_cells)

    async_arch = Distributed(CPU(); partition=Partition(2, 1, 1), synchronized_communication=false)
    sync_arch  = Distributed(CPU(); partition=Partition(2, 1, 1), synchronized_communication=true)

    @assert async_arch isa AsynchronousDistributed
    @assert !(sync_arch isa AsynchronousDistributed)

    @testset "Distributed non-immersed, async: cond = NamedTuple of IBS, active = NA" begin
        grid = RectilinearGrid(async_arch; size=(N, N, N), halo=(H, H, H), extent=(1, 1, 1), topology=(Bounded, Bounded, Bounded))
        cmaps = generate_condition_maps(grid, advection;
                                        condition_momentum_advection=true,
                                        condition_tracer_advection=true)
        for key in (:momentum, :c)
            cm = cmaps[key]
            @test cm isa NamedTuple
            @test keys(cm) == region_keys
            @test cm.halo_independent_cells isa InteriorBoundarySet
        end
        @test isnothing(get_active_cells_map(grid, Val(:core)))
    end

    @testset "Distributed non-immersed, sync: cond = single IBS, active = NA (like serial)" begin
        grid = RectilinearGrid(sync_arch; size=(N, N, N), halo=(H, H, H), extent=(1, 1, 1), topology=(Bounded, Bounded, Bounded))
        cmaps = generate_condition_maps(grid, advection;
                                        condition_momentum_advection=true,
                                        condition_tracer_advection=true)
        for key in (:momentum, :c)
            @test cmaps[key] isa InteriorBoundarySet
            check_split_consistency(cmaps[key], prod(size(grid)))
        end
        @test isnothing(get_active_cells_map(grid, Val(:core)))
    end

    @testset "Distributed immersed, async: cond = NamedTuple of IBS, active = NamedTuple of Array" begin
        underlying = RectilinearGrid(async_arch; size=(N, N, N), halo=(H, H, H), extent=(1, 1, 1), topology=(Bounded, Bounded, Bounded))
        ibg = ImmersedBoundaryGrid(underlying, GridFittedBottom(bottom_bump); active_cells_map=true)
        cmaps = generate_condition_maps(ibg, advection;
                                        condition_momentum_advection=true,
                                        condition_tracer_advection=true)
        @test ibg.interior_active_cells isa NamedTuple
        @test keys(ibg.interior_active_cells) == region_keys
        for key in (:momentum, :c)
            cm = cmaps[key]
            @test cm isa NamedTuple
            @test keys(cm) == region_keys
            @test cm.halo_independent_cells isa InteriorBoundarySet
            for region in region_keys
                isnothing(cm[region]) && continue
                check_split_consistency(cm[region], length(ibg.interior_active_cells[region]))
            end
        end
    end

    @testset "Distributed immersed, sync: cond = single IBS, active = Array (like serial)" begin
        underlying = RectilinearGrid(sync_arch; size=(N, N, N), halo=(H, H, H), extent=(1, 1, 1), topology=(Bounded, Bounded, Bounded))
        ibg = ImmersedBoundaryGrid(underlying, GridFittedBottom(bottom_bump); active_cells_map=true)
        cmaps = generate_condition_maps(ibg, advection;
                                        condition_momentum_advection=true,
                                        condition_tracer_advection=true)
        active = get_active_cells_map(ibg, Val(:core))
        @test active isa AbstractArray
        @test ibg.interior_active_cells isa AbstractArray
        for key in (:momentum, :c)
            @test cmaps[key] isa InteriorBoundarySet
            check_split_consistency(cmaps[key], length(active))
        end
    end
"""

@testset "Distributed condition maps (MPI)" begin
    script_filename = joinpath(@__DIR__, "distributed_condition_maps_script.jl")
    write(script_filename, distributed_condition_maps_script)
    try
        run(`$(mpiexec()) -n 2 $(Base.julia_cmd()) --project=$(Base.active_project()) -O0 $script_filename`)
    finally
        rm(script_filename, force=true)
    end
end
