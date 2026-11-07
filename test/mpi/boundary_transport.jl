using MPI

MPI.Init(threadlevel=:multiple)

include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using Oceananigans.DistributedComputations: Distributed, Partition, all_reduce
using Oceananigans.BoundaryConditions: BalancedTransport, NormalFlow
using Oceananigans.Models: enforce_net_zero_transport!

is_normal_flow(bc) = bc.classification isa NormalFlow

# Shift of each boundary this rank owns, from a face value and the imposed one.
function boundary_shifts(u, v, west, east, south, north)
    u_interior, v_interior = Array(interior(u)), Array(interior(v))
    u_bcs, v_bcs = u.boundary_conditions, v.boundary_conditions
    shifts = Float64[]
    is_normal_flow(u_bcs.west)  && west  != 0 && append!(shifts, west .- u_interior[1, :, :])
    is_normal_flow(u_bcs.east)  && east  != 0 && append!(shifts, u_interior[end, :, :] .- east)
    is_normal_flow(v_bcs.south) && south != 0 && append!(shifts, south .- v_interior[:, 1, :])
    is_normal_flow(v_bcs.north) && north != 0 && append!(shifts, v_interior[:, end, :] .- north)
    return shifts
end

balanced(value) = NormalFlowBoundaryCondition(value; scheme = BalancedTransport())
imposed(value) = value == 0 ? NormalFlowBoundaryCondition(value) : balanced(value)

@testset "Net-zero transport is global" begin
    Lx, Ly, Lz = 2.0, 1.0, 1.0

    # With zero south and north flow, the two middle ranks of Partition(x = 4) own no open face.
    configurations = [(1.0, 0.5, 0.3, 0.1), (1.0, 0.5, 0.0, 0.0)]

    for partition in (Partition(x = 4), Partition(y = 4), Partition(2, 2)), (west, east, south, north) in configurations
        arch = Distributed(child_arch; partition)
        grid = RectilinearGrid(arch; size = (16, 16, 4), x = (0, Lx), y = (0, Ly), z = (0, Lz),
                               topology = (Bounded, Bounded, Bounded))

        boundary_conditions = (u = FieldBoundaryConditions(west = balanced(west), east = balanced(east)),
                               v = FieldBoundaryConditions(south = imposed(south), north = imposed(north)))

        model = NonhydrostaticModel(grid; boundary_conditions, timestepper = :RungeKutta3)
        u, v, w = model.velocities

        net_inflow = (west - east) * Ly * Lz + (south - north) * Lx * Lz
        area = 2 * Ly * Lz + (south != 0 ? 2 * Lx * Lz : 0)

        fill_halo_regions!((u, v, w))
        enforce_net_zero_transport!(model.velocities, model.boundary_transport)

        @test model.boundary_transport.total_area_pool_boundaries ≈ area

        shifts = boundary_shifts(u, v, west, east, south, north)
        @test isempty(shifts) || all(s -> isapprox(s, net_inflow / area; atol = 1e-12), shifts)

        for _ in 1:3
            time_step!(model, 1e-3)
        end

        δ = Field(∂x(u) + ∂y(v) + ∂z(w))
        compute!(δ)
        @test maximum(abs, δ) < 1e-10
    end
end

@testset "Targeted transport is reached on the whole boundary" begin
    Lx, Ly, Lz = 2.0, 1.0, 1.0
    target = 0.3

    # Partition(x = 4) leaves two ranks with no open face; the others split the west and east boundaries.
    for partition in (Partition(x = 4), Partition(y = 4), Partition(2, 2))
        arch = Distributed(child_arch; partition)
        grid = RectilinearGrid(arch; size = (16, 16, 4), x = (0, Lx), y = (0, Ly), z = (0, Lz),
                               topology = (Bounded, Bounded, Bounded))

        targeted = NormalFlowBoundaryCondition(1.0; scheme = PerturbationAdvection(; target_transport = target))
        boundary_conditions = (; u = FieldBoundaryConditions(west = targeted, east = balanced(1.0)))

        model = NonhydrostaticModel(grid; boundary_conditions, timestepper = :RungeKutta3)
        u, v, w = model.velocities

        fill_halo_regions!((u, v, w))
        enforce_net_zero_transport!(model.velocities, model.boundary_transport)

        # This rank's part of the west transport, summed over every rank
        on_west = is_normal_flow(u.boundary_conditions.west)
        west_transport = Field(Integral(view(u, 1, :, :), dims = (2, 3), locally = true))
        compute!(west_transport)
        local_transport = on_west ? Array(interior(west_transport))[1, 1, 1] : 0.0
        @test all_reduce(+, local_transport, arch) ≈ target
    end
end
