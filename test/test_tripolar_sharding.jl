using Test

include("reactant_test_utils.jl")
include("distributed_tripolar_tests_utils.jl")

# Here, we reuse the tests performed in `test_distributed_tripolar.jl`, to check that
# the sharding is performed correctly.

# We are running on 8 "fake" CPUs
ENV["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

@testset "Sharded tripolar grid and fields" begin
    child_arch = ReactantState()

    archs = [Distributed(child_arch, partition=Partition(1, 8)),
             Distributed(child_arch, partition=Partition(2, 4)),
             Distributed(child_arch, partition=Partition(4, 2))]

    for arch in archs
        @info "  Testing a tripolar grid on a $(arch.ranks) partition"
        local_grid  = TripolarGrid(arch; size = (40, 40, 1), z = (-1000, 0), halo = (2, 2, 2))
        global_grid = TripolarGrid(child_arch; size = (40, 40, 1), z = (-1000, 0), halo = (2, 2, 2))
        
        reconstruct_grid = reconstruct_global_grid(local_grid)

        @test reconstruct_grid == global_grid
        
        nx, ny, _ = size(local_grid)
        rx, ry, _ = arch.local_index .- 1

        jrange = 1 + ry * ny : (ry + 1) * ny
        irange = 1 + rx * nx : (rx + 1) * nx

        for var in [:Δxᶠᶠᵃ, :Δxᶜᶜᵃ, :Δxᶠᶜᵃ, :Δxᶜᶠᵃ,
                    :Δyᶠᶠᵃ, :Δyᶜᶜᵃ, :Δyᶠᶜᵃ, :Δyᶜᶠᵃ,
                    :Azᶠᶠᵃ, :Azᶜᶜᵃ, :Azᶠᶜᵃ, :Azᶜᶠᵃ]

            @test getproperty(local_grid, var)[1:nx, 1:ny] == getproperty(global_grid, var)[irange, jrange]
            @test getproperty(local_grid, var)[1:nx, 1:ny] == getproperty(global_grid, var)[irange, jrange]
            @test getproperty(local_grid, var)[1:nx, 1:ny] == getproperty(global_grid, var)[irange, jrange]
        end

        @info " Testing sharded fields on a $(arch.ranks) partition"

        u = [i + 10 * j for i in 1:40, j in 1:40]
        v = [i + 10 * j for i in 1:40, j in 1:40]
        c = [i + 10 * j for i in 1:40, j in 1:40]
    
        up = XFaceField(local_grid)
        vp = YFaceField(local_grid)
        cp = CenterField(local_grid)

        set!(up, u)
        set!(vp, v)
        set!(cp, c)
        
        us = XFaceField(global_grid)
        vs = YFaceField(global_grid)
        cs = CenterField(global_grid)

        set!(us, u)
        set!(vs, v)
        set!(cs, c)

        @test us == reconstruct_global_field(up)
        @test vs == reconstruct_global_field(vp)
        @test cs == reconstruct_global_field(cp)
    end
end
