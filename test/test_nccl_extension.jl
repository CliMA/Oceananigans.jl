using MPI
MPI.Init()

using Random
Random.seed!(1234)

include("dependencies_for_runtests.jl")

if child_arch isa CPU
    @info "NCCL extension tests require GPU — skipping on CPU."
else

include("dependencies_for_poisson_solvers.jl")

using NCCL
using Oceananigans.DistributedComputations: TransposableField,
                                            reconstruct_global_grid,
                                            DistributedGrid,
                                            DistributedFourierTridiagonalPoissonSolver,
                                            transpose_z_to_y!,
                                            transpose_y_to_z!,
                                            transpose_y_to_x!,
                                            transpose_x_to_y!,
                                            synchronize_communication!

using Oceananigans.BoundaryConditions: regularize_field_boundary_conditions
using Oceananigans.Models.NonhydrostaticModels: solve_for_pressure!

using OceananigansNCCLExt: NCCLDistributed

function random_divergent_source_term(grid::DistributedGrid)
    arch = architecture(grid)
    default_bcs = FieldBoundaryConditions()

    u_bcs = regularize_field_boundary_conditions(default_bcs, grid, :u)
    v_bcs = regularize_field_boundary_conditions(default_bcs, grid, :v)
    w_bcs = regularize_field_boundary_conditions(default_bcs, grid, :w)

    u_bcs = inject_halo_communication_boundary_conditions(u_bcs, (Face(), Center(), Center()), arch.local_rank, arch.connectivity, topology(grid))
    v_bcs = inject_halo_communication_boundary_conditions(v_bcs, (Center(), Face(), Center()), arch.local_rank, arch.connectivity, topology(grid))
    w_bcs = inject_halo_communication_boundary_conditions(w_bcs, (Center(), Center(), Face()), arch.local_rank, arch.connectivity, topology(grid))

    Ru = XFaceField(grid, boundary_conditions=u_bcs)
    Rv = YFaceField(grid, boundary_conditions=v_bcs)
    Rw = ZFaceField(grid, boundary_conditions=w_bcs)
    U = (u=Ru, v=Rv, w=Rw)

    Nx, Ny, Nz = size(grid)
    set!(Ru, rand(size(Ru)...))
    set!(Rv, rand(size(Rv)...))
    set!(Rw, rand(size(Rw)...))

    fill_halo_regions!(Ru)
    fill_halo_regions!(Rv)
    fill_halo_regions!(Rw)

    ArrayType = array_type(arch)
    R = zeros(Nx, Ny, Nz) |> ArrayType
    launch!(arch, grid, :xyz, divergence!, grid, U.u.data, U.v.data, U.w.data, R)

    return R, U
end

#####
##### Smoke test: NCCLDistributed construction + halo fill
#####

@testset "NCCL extension smoke tests" begin
    @info "  Testing NCCLDistributed construction..."
    arch = NCCLDistributed(GPU(); partition=Partition(4, 1, 1))
    @test arch isa Distributed

    grid = RectilinearGrid(arch;
                           topology = (Periodic, Periodic, Bounded),
                           size = (16, 16, 4),
                           extent = (1, 1, 1))

    @info "  Testing NCCL halo fill..."
    c = CenterField(grid)
    set!(c, (x, y, z) -> x + y)
    fill_halo_regions!(c)

    # After halo fill, interior should be unchanged
    @test all(isfinite, Array(interior(c)))
end

#####
##### Halo correctness: neighbor-value ping-pong
#####
##### Unlike the smoke test above (which only checks that the interior stays finite
##### after a halo fill), this verifies that fill_halo_regions! actually moves the
##### CORRECT neighbor data through NCCL into every halo region — edges in 1D
##### decompositions, edges *and* corners in 2D decompositions.
#####
##### Method: set each rank's interior to a rank-unique value 10*(rank+1). After the
##### fill, each halo region must hold 10*(neighbor_rank+1), where the neighbor for
##### each of the 8 directions is read from arch.connectivity. Directions with no
##### neighbor (`nothing`, e.g. a Bounded edge) are skipped. Indices are taken from
##### the ACTUAL parent array (the decomposed dimension's local extent differs from
##### the global size).

# Representative single cell for each of the 8 halo directions, given parent size
# (Tx, Ty), halo H, and an interior z index mz. (First cell of each halo block.)
function nccl_halo_probe_cells(Tx, Ty, H, mz)
    return (west      = (1,   H+1, mz), east      = (Tx, H+1, mz),
            south     = (H+1, 1,   mz), north      = (H+1, Ty,  mz),
            southwest = (1,   1,   mz), southeast = (Tx, 1,   mz),
            northwest = (1,   Ty,  mz), northeast = (Tx, Ty,  mz))
end

# Verify a single filled field: every halo region with a neighbor holds 10*(neighbor+1).
function nccl_halo_values_correct(c, conn, H)
    P = Array(parent(c))
    Tx, Ty, Tz = size(P)
    mz = H + 1
    cell = nccl_halo_probe_cells(Tx, Ty, H, mz)

    all_correct = true
    for dir in (:west, :east, :south, :north, :southwest, :southeast, :northwest, :northeast)
        nbr = getproperty(conn, dir)
        nbr === nothing && continue  # no neighbor (Bounded edge / 1D decomposition)
        i, j, k = getproperty(cell, dir)
        want = convert(eltype(P), 10 * (nbr + 1))
        all_correct &= P[i, j, k] == want
    end

    return all_correct
end

function test_nccl_halo_neighbor_values(grid_points, ranks, topo;
                                        halo=(3, 3, 3), field_type=CenterField,
                                        multi_field=false, async=false)
    arch = NCCLDistributed(GPU(); partition=Partition(ranks...))
    grid = RectilinearGrid(arch, topology=topo, size=grid_points, halo=halo, extent=(1, 1, 1))

    H = halo[1]
    rank = arch.local_rank
    conn = arch.connectivity

    fields = multi_field ? ntuple(_ -> field_type(grid), 3) : (field_type(grid),)
    for c in fields
        set!(c, convert(eltype(c), 10 * (rank + 1)))  # interior ← this rank's unique value
    end

    if async
        # Async overlap: issue all fills (deferred unpack), then complete the exchange.
        for c in fields
            fill_halo_regions!(c; async=true)
        end
        for c in fields
            synchronize_communication!(c)
        end
    else
        for c in fields
            fill_halo_regions!(c)  # the path under test
        end
    end

    CUDA.synchronize()
    return all(nccl_halo_values_correct(c, conn, H) for c in fields)
end

@testset "NCCL halo correctness (neighbor-value ping-pong)" begin
    # The Buildkite distributed GPU pipeline launches this job with 4 ranks
    # (slurm_ntasks: 4), so we cover the 1D strip decompositions (edges only) AND
    # the 2D 2×2 decomposition (edges + corners, exercising the NCCL corner path).
    for topo in ((Periodic, Periodic, Periodic),
                 (Periodic, Periodic, Bounded))
        @info "  Testing NCCL halo neighbor values with topology $topo, (4,1,1) ranks (edges)..."
        @test test_nccl_halo_neighbor_values((44, 44, 8), (4, 1, 1), topo)
        @info "  Testing NCCL halo neighbor values with topology $topo, (1,4,1) ranks (edges)..."
        @test test_nccl_halo_neighbor_values((44, 44, 8), (1, 4, 1), topo)
        @info "  Testing NCCL halo neighbor values with topology $topo, (2,2,1) ranks (edges + corners)..."
        @test test_nccl_halo_neighbor_values((44, 44, 8), (2, 2, 1), topo)
    end

    # Multi-field variant: repeated, independent fills must each be correct.
    @info "  Testing NCCL halo neighbor values with multiple fields, (2,2,1) ranks..."
    @test test_nccl_halo_neighbor_values((44, 44, 8), (2, 2, 1), (Periodic, Periodic, Bounded); multi_field=true)

    # Invariance to halo size: the correctness must not depend on H.
    for H in (2, 4)
        @info "  Testing NCCL halo neighbor values with halo=$H, (2,2,1) ranks..."
        @test test_nccl_halo_neighbor_values((44, 44, 8), (2, 2, 1), (Periodic, Periodic, Bounded); halo=(H, H, H))
    end

    # Face-located fields (the momentum fields ρu, ρv, ρw) — staggered halo geometry
    # differs from centers, so exchange them explicitly.
    for FT in (XFaceField, YFaceField, ZFaceField)
        @info "  Testing NCCL halo neighbor values for $FT, (2,2,1) ranks..."
        @test test_nccl_halo_neighbor_values((44, 44, 8), (2, 2, 1), (Periodic, Periodic, Bounded); field_type=FT)
    end

    # Async overlap path: deferred unpack completed by synchronize_communication!.
    for ranks in ((4, 1, 1), (1, 4, 1), (2, 2, 1))
        @info "  Testing NCCL async halo overlap, $ranks ranks..."
        @test test_nccl_halo_neighbor_values((44, 44, 8), ranks, (Periodic, Periodic, Bounded); async=true)
    end
end

#####
##### Transpose round-trip
#####

function test_nccl_transpose(grid_points, ranks, topo)
    arch = NCCLDistributed(GPU(); partition=Partition(ranks...))
    grid = RectilinearGrid(arch, topology=topo, size=grid_points, extent=(2π, 2π, 2π))

    ϕ = Field((Center(), Center(), Center()), grid, ComplexF64)
    Φ = TransposableField(ϕ)

    ϕ₀ = on_architecture(GPU(), rand(ComplexF64, size(ϕ)))
    set!(ϕ, ϕ₀)
    set!(Φ.zfield, ϕ)

    transpose_z_to_y!(Φ)
    transpose_y_to_x!(Φ)
    transpose_x_to_y!(Φ)
    transpose_y_to_z!(Φ)

    same_real = all(real.(Array(interior(ϕ))) .== real.(Array(interior(Φ.zfield))))
    same_imag = all(imag.(Array(interior(ϕ))) .== imag.(Array(interior(Φ.zfield))))

    return same_real & same_imag
end

@testset "NCCL transpose round-trip" begin
    for topo in ((Periodic, Periodic, Periodic),
                 (Periodic, Periodic, Bounded),
                 (Bounded, Bounded, Bounded))
        @info "  Testing NCCL transpose with topology $topo..."
        @test test_nccl_transpose((44, 44, 8), (4, 1, 1), topo)
        @test test_nccl_transpose((44, 44, 8), (1, 4, 1), topo)
    end
end

#####
##### Poisson solver
#####

function nccl_divergence_free_poisson_solution(grid_points, ranks, topo)
    arch = NCCLDistributed(GPU(); partition=Partition(ranks...))
    local_grid = RectilinearGrid(arch, topology=topo, size=grid_points, extent=(2π, 2π, 2π))

    ϕ   = CenterField(local_grid)
    ∇²ϕ = CenterField(local_grid)
    R, U = random_divergent_source_term(local_grid)

    global_grid = reconstruct_global_grid(local_grid)
    solver = DistributedFFTBasedPoissonSolver(global_grid, local_grid)

    solve_for_pressure!(ϕ, solver, nothing, U, 1)
    compute_∇²!(∇²ϕ, ϕ, arch, local_grid)

    return Array(interior(∇²ϕ)) ≈ Array(R)
end

function nccl_divergence_free_tridiagonal_solution(grid_points, ranks, stretched_direction)
    arch = NCCLDistributed(GPU(); partition=Partition(ranks...))

    if stretched_direction == :z
        z = collect(range(0, 2π, length=grid_points[3]+1))
        x = y = (0, 2π)
    end

    local_grid = RectilinearGrid(arch;
                                 topology = (Bounded, Bounded, Bounded),
                                 size = grid_points,
                                 halo = (2, 2, 2),
                                 x, y, z)

    ϕ   = CenterField(local_grid)
    ∇²ϕ = CenterField(local_grid)
    R, U = random_divergent_source_term(local_grid)

    global_grid = reconstruct_global_grid(local_grid)
    solver = DistributedFourierTridiagonalPoissonSolver(global_grid, local_grid)

    solve_for_pressure!(ϕ, solver, nothing, U, 1)
    compute_∇²!(∇²ϕ, ϕ, arch, local_grid)

    return Array(interior(∇²ϕ)) ≈ Array(R)
end

@testset "NCCL distributed Poisson solvers" begin
    for topo in ((Periodic, Periodic, Periodic),
                 (Periodic, Periodic, Bounded),
                 (Bounded, Bounded, Bounded))
        @info "  Testing NCCL FFT Poisson solver with topology $topo, (4,1,1) ranks..."
        @test nccl_divergence_free_poisson_solution((44, 44, 8), (4, 1, 1), topo)
        @info "  Testing NCCL FFT Poisson solver with topology $topo, (1,4,1) ranks..."
        @test nccl_divergence_free_poisson_solution((44, 44, 8), (1, 4, 1), topo)
    end

    @info "  Testing NCCL Fourier-Tridiagonal solver, z-stretched, (4,1,1) ranks..."
    @test nccl_divergence_free_tridiagonal_solution((44, 44, 8), (4, 1, 1), :z)
    @info "  Testing NCCL Fourier-Tridiagonal solver, z-stretched, (1,4,1) ranks..."
    @test nccl_divergence_free_tridiagonal_solution((44, 44, 8), (1, 4, 1), :z)
end

end # if child_arch isa CPU ... else
