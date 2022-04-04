using Oceananigans.Solvers: solve!, HeptadiagonalIterativeSolver, sparse_approximate_inverse
using Oceananigans.Operators: volume, Δyᶠᶜᵃ, Δyᶜᶠᵃ, Δyᶜᶜᵃ, Δxᶠᶜᵃ, Δxᶜᶠᵃ, Δxᶜᶜᵃ, Δyᵃᶜᵃ, Δxᶜᵃᵃ, Δzᵃᵃᶠ, Δzᵃᵃᶜ, ∇²ᶜᶜᶜ
using Oceananigans.Architectures: arch_array
using KernelAbstractions: @kernel, @index
using Statistics, LinearAlgebra, SparseArrays

function calc_∇²!(∇²ϕ, ϕ, grid)
    arch = architecture(grid)
    fill_halo_regions!(ϕ)
    event = launch!(arch, grid, :xyz, ∇²!, ∇²ϕ, grid, ϕ)
    wait(event)
    fill_halo_regions!(∇²ϕ)
    return nothing
end

function identity_operator!(b, x)
    parent(b) .= parent(x)
    return nothing
end

function run_identity_operator_test(grid)
    arch = architecture(grid)

    if arch isa GPU
        mrg = MultiRegionGrid(grid, partition = XPartition(2), devices = (0, 0))
    else
        mrg = MultiRegionGrid(grid, partition = XPartition(2))
    end
    
    N = size(grid)
    M = prod(N)

    b = unified_array(arch, zeros(grid, M))
    A = zeros(grid, N...)
    D = zeros(grid, N...)
    C = zeros(grid, N...)
    fill!(C, 1)

    solver = UnifiedDiagonalIterativeSolver((A, A, A, C, D), grid = grid, mrg = mrg)

    fill!(b, rand())

    initial_guess = solution = CenterField(grid)
    set!(initial_guess, (x, y, z) -> rand())
    
    solve!(solver, b, 1.0)
    
    solution = reshape(solver.solution, grid.Nx, grid.Ny, grid.Nz)

    b = reshape(b, size(grid)...)

    @test norm(solution .- b) .< solver.tolerance
end

@kernel function _multiply_by_volume!(r, grid)
    i, j, k = @index(Global, NTuple)
    r[i, j, k] *= volume(i, j, k, grid, Center(), Center(), Center())
end

function compute_poisson_weights(grid)
    N = size(grid)
    Ax = zeros(N...)
    Ay = zeros(N...)
    Az = zeros(N...)
    C  = zeros(grid, N...)
    D  = arch_array(grid.architecture, ones(N...))
    for i = 1:grid.Nx, j = 1:grid.Ny, k = 1:grid.Nz
        Ax[i, j, k] = Δzᵃᵃᶜ(i, j, k, grid) * Δyᶠᶜᵃ(i, j, k, grid) / Δxᶠᶜᵃ(i, j, k, grid)
        Ay[i, j, k] = Δzᵃᵃᶜ(i, j, k, grid) * Δxᶜᶠᵃ(i, j, k, grid) / Δyᶜᶠᵃ(i, j, k, grid)
        Az[i, j, k] = Δxᶜᶜᵃ(i, j, k, grid) * Δyᶜᶜᵃ(i, j, k, grid) / Δzᵃᵃᶠ(i, j, k, grid)
    end
    return (Ax, Ay, Az, C, D)
end

function poisson_rhs!(r, grid)
    event = launch!(architecture(grid), grid, :xyz, _multiply_by_volume!, r, grid)
    wait(event)
    return nothing
end

using Oceananigans.MultiRegion
using Oceananigans.MultiRegion: UnifiedDiagonalIterativeSolver
using Oceananigans.Architectures: unified_array

function run_poisson_equation_test(grid)
    arch = architecture(grid)

    if arch isa GPU
        mrg = MultiRegionGrid(grid, partition = XPartition(2), devices = (0, 0))
    else
        mrg = MultiRegionGrid(grid, partition = XPartition(2))
    end
    
    # Solve ∇²ϕ = r
    ϕ_truth = CenterField(grid)

    # Initialize zero-mean "truth" solution with random numbers
    set!(ϕ_truth, (x, y, z) -> rand())
    parent(ϕ_truth) .-= mean(ϕ_truth)
    fill_halo_regions!(ϕ_truth)

    # Calculate Laplacian of "truth"
    ∇²ϕ = CenterField(grid)
    calc_∇²!(∇²ϕ, ϕ_truth, grid)
    
    rhs = deepcopy(∇²ϕ)
    poisson_rhs!(rhs, grid)
    rhs = copy(interior(rhs))
    rhs = reshape(rhs, length(rhs))
    rhs = unified_array(arch, rhs)
    weights = compute_poisson_weights(grid)
    solver  = UnifiedDiagonalIterativeSolver(weights, grid = grid, mrg = mrg)

    # Solve Poisson equation
    solve!(solver, rhs, 1.0)

    ϕ_temp = reshape(solver.solution, grid.Nx, grid.Ny, grid.Nz)
    ϕ_solution = CenterField(grid)
    set!(ϕ_solution, ϕ_temp)

    # Diagnose Laplacian of solution
    ∇²ϕ_solution = CenterField(grid)
    calc_∇²!(∇²ϕ_solution, ϕ_solution, grid)

    CUDA.@allowscalar begin
        @test all(interior(∇²ϕ_solution) .≈ interior(∇²ϕ))
        @test all(interior(ϕ_solution)   .≈ interior(ϕ_truth)) 
    end

    return nothing
end

@testset "HeptadiagonalIterativeSolver" begin
    topologies = [(Periodic, Periodic, Flat), (Bounded, Bounded, Flat), (Periodic, Bounded, Flat), (Bounded, Periodic, Flat)]

    for arch in archs, topo in topologies
        @info "Testing 2D HeptadiagonalIterativeSolver [$(typeof(arch)) $topo]..."
        
        grid = RectilinearGrid(arch, size=(4, 8), extent=(1, 3), topology = topo)
        run_identity_operator_test(grid)
        run_poisson_equation_multiregion_test(grid)
    end

    topologies = [(Periodic, Periodic, Periodic), (Bounded, Bounded, Periodic), (Periodic, Bounded, Periodic), (Bounded, Periodic, Bounded)]

    for arch in archs, topo in topologies
        @info "Testing 3D HeptadiagonalIterativeSolver [$(typeof(arch)) $topo]..."
        
        grid = RectilinearGrid(arch, size=(4, 8, 6), extent=(1, 3, 4), topology=topo)
        run_identity_operator_test(grid)
        run_poisson_equation_test(grid)
    end

    stretched_faces = [0, 1.5, 3, 7, 8.5, 10]
    topo = (Periodic, Periodic, Periodic)
    sz = (5, 5, 5)

    for arch in archs
        grids = [RectilinearGrid(arch, size = sz, x = stretched_faces, y = (0, 10), z = (0, 10), topology = topo), 
                 RectilinearGrid(arch, size = sz, x = (0, 10), y = stretched_faces, z = (0, 10), topology = topo), 
                 RectilinearGrid(arch, size = sz, x = (0, 10), y = (0, 10), z = stretched_faces, topology = topo)]

        for (grid, stretched_direction) in zip(grids, [:x, :y, :z])
            @info "  Testing HeptadiagonalIterativeSolver [stretched in $stretched_direction, $(typeof(arch))]..."
            run_poisson_equation_test(grid)
        end
    end

    @info "  Testing Sparse Approximate Inverse..."

    A   = sprand(100, 100, 0.1)
    A   = A + A' + 1I
    A⁻¹ = sparse(inv(Array(A)))
    M   = sparse_approximate_inverse(A, ε = 0.0, nzrel = size(A, 1))
    
    @test all(Array(M) .≈ A⁻¹)
    
end
