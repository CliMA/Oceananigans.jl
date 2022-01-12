using Oceananigans.Solvers: solve!, HeptadiagonalIterativeSolver, sparse_approximate_inverse
using Oceananigans.Fields: interior_copy
using Oceananigans.Operators: volume, Δyᶠᶜᵃ, Δyᶜᶠᵃ, Δxᶠᶜᵃ, Δxᶜᶠᵃ, Δyᵃᶜᵃ, Δxᶜᵃᵃ, Δzᵃᵃᶠ, Δzᵃᵃᶜ, ∇²ᶜᶜᶜ
using Oceananigans.Architectures: arch_array
using KernelAbstractions: @kernel, @index
using Statistics, LinearAlgebra, SparseArrays

function calc_∇²!(∇²ϕ, ϕ, arch, grid)
    fill_halo_regions!(ϕ, arch)
    event = launch!(arch, grid, :xyz, ∇²!, ∇²ϕ, grid, ϕ)
    wait(event)
    fill_halo_regions!(∇²ϕ, arch)
    return nothing
end


function identity_operator!(b, x)
    parent(b) .= parent(x)
    return nothing
end

function run_identity_operator_test(arch, grid)

    N = size(grid)
    M = prod(N)

    b = arch_array(arch, zeros(M))

    A = zeros(N...)
    C =  ones(N...)
    D = arch_array(arch, zeros(N...))

    solver = HeptadiagonalIterativeSolver((A, A, A, C, D), grid = grid)

    fill!(b, rand())

    initial_guess = solution = CenterField(arch, grid)
    set!(initial_guess, (x, y, z) -> rand())
    
    solve!(initial_guess, solver, b, 1.0)

    b = reshape(b, size(grid)...)
    @test norm(interior(solution) .- b) .< solver.tolerance
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
    C  = zeros(N...)
    D  = arch_array(grid.architecture, zeros(N...))
    for i =1:grid.Nx, j = 1:grid.Ny, k = 1:grid.Nz
        Ax[i, j, k] = Δzᵃᵃᶜ(i, j, k, grid) * Δyᶠᶜᵃ(i, j, k, grid) / Δxᶠᶜᵃ(i, j, k, grid)
        Ay[i, j, k] = Δzᵃᵃᶜ(i, j, k, grid) * Δxᶜᶠᵃ(i, j, k, grid) / Δyᶜᶠᵃ(i, j, k, grid)
        Az[i, j, k] = Δxᶜᵃᵃ(i, j, k, grid) * Δyᵃᶜᵃ(i, j, k, grid) / Δzᵃᵃᶠ(i, j, k, grid)
    end
    return (Ax, Ay, Az, C, D)
end


function poisson_rhs!(r, grid)
    event = launch!(grid.architecture, grid, :xyz, _multiply_by_volume!, r, grid)
    wait(event)
    return 
end

function run_poisson_equation_test(arch, grid)
    # Solve ∇²ϕ = r
    ϕ_truth = Field(Center, Center, Center, arch, grid)

    # Initialize zero-mean "truth" solution with random numbers
    set!(ϕ_truth, (x, y, z) -> rand())
    parent(ϕ_truth) .-= mean(ϕ_truth)
    fill_halo_regions!(ϕ_truth, arch)

    # Calculate Laplacian of "truth"
    ∇²ϕ = Field(Center, Center, Center, arch, grid)
    calc_∇²!(∇²ϕ, ϕ_truth, arch, grid)
    
    rhs = deepcopy(∇²ϕ)
    poisson_rhs!(rhs, grid)
    rhs = interior_copy(rhs)[:]
    weights = compute_poisson_weights(grid)
    solver  = HeptadiagonalIterativeSolver(weights, grid = grid, preconditioner_method = nothing)

    # Solve Poisson equation
    ϕ_solution = CenterField(arch, grid)

    solve!(ϕ_solution, solver, rhs, 1.0)

    # Diagnose Laplacian of solution
    ∇²ϕ_solution = CenterField(arch, grid)
    calc_∇²!(∇²ϕ_solution, ϕ_solution, arch, grid)

    parent(ϕ_solution) .-= mean(ϕ_solution)

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
        run_identity_operator_test(arch, grid)
        run_poisson_equation_test(arch, grid)
    end
    topologies = [(Periodic, Periodic, Periodic), (Bounded, Bounded, Periodic), (Periodic, Bounded, Periodic), (Bounded, Periodic, Bounded)]
    for arch in archs, topo in topologies
        @info "Testing 3D HeptadiagonalIterativeSolver [$(typeof(arch)) $topo]..."
        
        grid = RectilinearGrid(arch, size=(4, 8, 6), extent=(1, 3, 4), topology = topo)
        run_identity_operator_test(arch, grid)
        run_poisson_equation_test(arch, grid)
    end
    stretch_coord = [0, 1.5, 3, 7, 8.5, 10]
    for arch in archs
        grids = [RectilinearGrid(arch, size=(5, 5, 5), x = stretch_coord, y = (0, 10), z = (0, 10), topology = (Periodic, Periodic, Periodic)), 
                 RectilinearGrid(arch, size=(5, 5, 5), x = (0, 10), y = stretch_coord, z = (0, 10), topology = (Periodic, Periodic, Periodic)), 
                 RectilinearGrid(arch, size=(5, 5, 5), x = (0, 10), y = (0, 10), z = stretch_coord, topology = (Periodic, Periodic, Periodic))]

        for grid in grids
            @info "Testing stretched grid HeptadiagonalIterativeSolver [$(typeof(arch))]..."
            run_poisson_equation_test(arch, grid)
        end
    end

    @info "Testing Sparse Approximate Inverse Algorithm"

    A   = sprand(100, 100, 0.1)
    A   = A + A' + 1I
    A⁻¹ = sparse(inv(Array(A)))
    M   = sparse_approximate_inverse(A, ε = 0.0, nzrel = size(A, 1))

    @test all(Array(M) .≈ A⁻¹)

end