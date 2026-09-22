using Random
using Oceananigans: initialize!
using Oceananigans.ImmersedBoundaries: PartialCellBottom
using Oceananigans.Grids: MutableVerticalDiscretization
using Oceananigans.Models: ZStarCoordinate, ZCoordinate
using Oceananigans.DistributedComputations: DistributedGrid, @root, @handshake

grid_type(::RectilinearGrid{F, X, Y}) where {F, X, Y} = "Rect{$X, $Y}"
grid_type(::LatitudeLongitudeGrid{F, X, Y}) where {F, X, Y} = "LatLon{$X, $Y}"

grid_type(g::ImmersedBoundaryGrid) = "Immersed" * grid_type(g.underlying_grid)

function test_zstar_coordinate(model, Ni, Δt)

    bᵢ = deepcopy(model.tracers.b)
    cᵢ = deepcopy(model.tracers.c)

    ∫bᵢ = Field(Integral(bᵢ))
    ∫cᵢ = Field(Integral(cᵢ))
    compute!(∫bᵢ)
    compute!(∫cᵢ)

    Bᵢ = interior(∫bᵢ, 1, 1, 1)
    Cᵢ = interior(∫cᵢ, 1, 1, 1)
    w   = model.velocities.w
    Nz  = model.grid.Nz

    for step in 1:Ni
        time_step!(model, Δt)

        ∫b = Field(Integral(model.tracers.b))
        ∫c = Field(Integral(model.tracers.c))
        compute!(∫b)
        compute!(∫c)

        Bₙ = interior(∫b, 1, 1, 1)
        condition = Bₙ ≈ Bᵢ
        if !condition
            @info "Stopping early: buoyancy not conserved at step $step, initial: $Bᵢ, final: $Bₙ"
        end
        @test condition

        Cₙ = interior(∫c, 1, 1, 1)
        condition = Cₙ ≈ Cᵢ
        if !condition
            @info "Stopping early: c tracer not conserved at step $step, initial: $Cᵢ, final: $Cₙ"
        end
        @test condition

        condition = maximum(abs, interior(w, :, :, Nz+1)) < eps(eltype(w))
        if !condition
            @info "Stopping early: nonzero vertical velocity at top at step $step with value: $(maximum(abs, interior(w, :, :, Nz+1)))"
        end
        @test condition

        @test maximum(model.tracers.constant) ≈ 1
        @test minimum(model.tracers.constant) ≈ 1
    end

    return nothing
end

function info_message(grid, free_surface, timestepper)
    msg1 = "$(summary(architecture(grid))) "
    msg2 = grid_type(grid)
    msg3 = " with $(timestepper)"
    msg4 = " using a " * string(getnamewrapper(free_surface))
    return msg1 * msg2 * msg3 * msg4
end

function zstar_test_architectures()
    archs = test_architectures()
    if length(archs) == 6 # Distributed with 6 archs, we only take the first 3
        archs = archs[1:3]
    end
    return archs
end

function zstar_test_topologies(arch)
    if arch isa Distributed
        # tests become too long because we test too many architectures,
        # given that `FullyConnected` acts as a `Periodic` we don't need
        # to test different topologies
        return [(Bounded, Bounded, Bounded)]
    else
        return [(Periodic, Periodic, Bounded),
                (Bounded, Bounded, Bounded)]
    end
end

function zstar_test_grids(arch, topology, z_stretched)
    Random.seed!(1234)

    # Distributed runs split the grid over up to 4 ranks and the split-explicit free surface extends
    # halos across ranks (to 7 with substeps=8), so each rank needs > 7 local cells; use a larger grid
    # there. Serial runs don't extend halos. Domains scale with the point count to keep spacing fixed.
    Nh = arch isa Distributed ? 32 : 8
    Lx = Nh * 2.5kilometers
    Ly = Nh * 0.5kilometers
    Lφ = Nh * 0.025

    rtgv  = RectilinearGrid(arch; size = (Nh, Nh, 5), x = (0, Lx), y = (-Ly/2, Ly/2), topology, z = z_stretched)
    irtgv = ImmersedBoundaryGrid(deepcopy(rtgv),  GridFittedBottom((x, y) -> rand() - 4))
    prtgv = ImmersedBoundaryGrid(deepcopy(rtgv), PartialCellBottom((x, y) -> rand() - 4))

    if topology[2] == Bounded
        llgv  = LatitudeLongitudeGrid(arch; size = (Nh, Nh, 5), latitude = (0, Lφ), longitude = (0, Lφ), topology, z = z_stretched)
        illgv = ImmersedBoundaryGrid(deepcopy(llgv),  GridFittedBottom((x, y) -> rand() - 4))
        pllgv = ImmersedBoundaryGrid(deepcopy(llgv), PartialCellBottom((x, y) -> rand() - 4))
        return [llgv, rtgv, illgv, irtgv, pllgv, prtgv]
    else
        return [rtgv, irtgv, prtgv]
    end
end
