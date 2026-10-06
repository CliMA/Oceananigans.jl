include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using Oceananigans.BoundaryConditions: PolarBoundaryCondition, PolarValueBoundaryCondition, update_pole_value!, fill_halo_regions!
import Oceananigans.BoundaryConditions: prepare_halo_fill!

# Every time index of a `FieldTimeSeries` shares the same boundary conditions, so all its slots share
# the single pole-value buffer of a `PolarValueBoundaryCondition`, which each halo fill first computes
# (the zonal mean of the last row) and then reads back. The slots must therefore be filled one after
# the other: if two fills interleave, a slot gets another slot's pole value. On the GPU they interleave
# whenever a task yields (e.g. while CUDA synchronizes streams); here a `yield()` after computing the
# pole value makes this happen on the CPU too.

@testset "Halos of a FieldTimeSeries with polar boundary conditions" begin
    @info "  Testing the polar halos of an in-memory FieldTimeSeries..."

    grid = LatitudeLongitudeGrid(CPU(); size = (16, 8), longitude = (0, 360), latitude = (-90, 90),
                                 halo = (3, 3), topology = (Periodic, Bounded, Flat))

    Nx, Ny = size(grid)[1:2]
    times = 0:29 # more slots than are filled together
    fts = FieldTimeSeries{Center, Center, Center}(grid, times)

    @test fts.boundary_conditions.north isa PolarValueBoundaryCondition

    for n in eachindex(times)
        interior(fts[n]) .= 100n .+ rand(Nx, Ny)
    end

    # Reference: each slot filled on its own
    north_halo(n) = Array(view(fts[n].data, 1:Nx, Ny+1, 1))
    reference = map(eachindex(times)) do n
        fill_halo_regions!(fts[n])
        north_halo(n)
    end

    # Yield between computing a pole value and using it
    @eval prepare_halo_fill!(bc::PolarBoundaryCondition, c, grid, loc) = (update_pole_value!(bc.condition, c, grid, loc); yield())

    try
        for n in eachindex(times)
            view(fts[n].data, :, Ny+1, :) .= NaN
        end

        fill_halo_regions!(fts)

        @test all(north_halo(n) == reference[n] for n in eachindex(times))
    finally
        @eval prepare_halo_fill!(bc::PolarBoundaryCondition, c, grid, loc) = update_pole_value!(bc.condition, c, grid, loc)
    end
end
