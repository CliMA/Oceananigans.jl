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
                                            transpose_x_to_y!

using Oceananigans.BoundaryConditions: regularize_field_boundary_conditions
using Oceananigans.Models.NonhydrostaticModels: solve_for_pressure!

using OceananigansNCCLExt: NCCLDistributed

function random_divergent_source_term(grid::DistributedGrid)
    arch = architecture(grid)
    default_bcs = FieldBoundaryConditions()

    u_bcs = regularize_field_boundary_conditions(default_bcs, grid, :u)
    v_bcs = regularize_field_boundary_conditions(default_bcs, grid, :v)
    w_bcs = regularize_field_boundary_conditions(default_bcs, grid, :w)

    u_bcs = inject_halo_communication_boundary_conditions(u_bcs, arch.local_rank, arch.connectivity, topology(grid))
    v_bcs = inject_halo_communication_boundary_conditions(v_bcs, arch.local_rank, arch.connectivity, topology(grid))
    w_bcs = inject_halo_communication_boundary_conditions(w_bcs, arch.local_rank, arch.connectivity, topology(grid))

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
##### Transpose round-trip
#####

function test_nccl_transpose(grid_points, ranks, topo)
    arch = NCCLDistributed(GPU(); partition=Partition(ranks...))
    grid = RectilinearGrid(arch, topology=topo, size=grid_points, extent=(2π, 2π, 2π))

    ϕ = Field((Center, Center, Center), grid, ComplexF64)
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
