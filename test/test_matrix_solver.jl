using Oceananigans.Solvers: solve!
using Oceananigans.Operators: volume, Δyᶠᶜᵃ, Δyᶜᶠᵃ, Δxᶠᶜᵃ, Δxᶜᶠᵃ, Δyᵃᶜᵃ, Δxᶜᵃᵃ, Δzᵃᵃᶠ, Δzᵃᵃᶜ
using Oceananigans.Architectures: arch_array
using KernelAbstractions: @kernel, @index
using Statistics


@kernel function _∇²!(∇²f, grid, f)
    i, j, k = @index(Global, NTuple)
    @inbounds ∇²f[i, j, k] = ∇²ᶜᶜᶜ(i, j, k, grid, f)
end

function calc_∇²!(∇²ϕ, ϕ, arch, grid)
    fill_halo_regions!(ϕ, arch)
    event = launch!(arch, grid, :xyz, _∇²!, ∇²ϕ, grid, ϕ)
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

    solver = MatrixIterativeSolver((A, A, A, C, D), grid = grid)

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


function multiply_by_volume!(r, grid)
    event = launch!(grid.architecture, grid, :xyz, _multiply_by_volume!, r, grid)
    wait(event)
    return
end

function run_poisson_equation_test(arch, grid)
    # Solve ∇²ϕ = r
    ϕ_truth = Field(Center, Center, Center, arch, grid)
    N = size(grid)

    # Initialize zero-mean "truth" solution with random numbers
    set!(ϕ_truth, (x, y, z) -> rand())
    parent(ϕ_truth) .-= mean(ϕ_truth)
    fill_halo_regions!(ϕ_truth, arch)

    # Calculate Laplacian of "truth"
    ∇²ϕ = Field(Center, Center, Center, arch, grid)
    calc_∇²!(∇²ϕ, ϕ_truth, arch, grid)
    
    r = deepcopy(∇²ϕ)
    multiply_by_volume!(r, grid)
    
    r = interior_copy(r)[:]

    Ax = zeros(N...)
    Ay = zeros(N...)
    Az = zeros(N...)
    C  = zeros(N...)
    D  = arch_array(arch, zeros(N...))
    for i =1:grid.Nx, j = 1:grid.Ny, k = 1:grid.Nz
        Ax[i, j, k] = Δzᵃᵃᶜ(i, j, k, grid) * Δyᶠᶜᵃ(i, j, k, grid) / Δxᶠᶜᵃ(i, j, k, grid)
        Ay[i, j, k] = Δzᵃᵃᶜ(i, j, k, grid) * Δxᶜᶠᵃ(i, j, k, grid) / Δyᶜᶠᵃ(i, j, k, grid)
        Az[i, j, k] = Δxᶜᵃᵃ(i, j, k, grid) * Δyᵃᶜᵃ(i, j, k, grid) / Δzᵃᵃᶠ(i, j, k, grid)
    end

    solver = MatrixIterativeSolver((Ax, Ay, Az, C, D), grid = grid, tolerance = 1e-14)

    # Solve Poisson equation
    ϕ_solution = CenterField(arch, grid)

    solve!(ϕ_solution, solver, r, 1.0)

    # Diagnose Laplacian of solution
    ∇²ϕ_solution = CenterField(arch, grid)
    calc_∇²!(∇²ϕ_solution, ϕ_solution, arch, grid)

    parent(ϕ_solution) .-= mean(ϕ_solution)

    # Test
    extrema_tolerance = 1e-11
    std_tolerance = 1e-12

    CUDA.@allowscalar begin
        @test minimum(abs, interior(∇²ϕ_solution) .- interior(∇²ϕ)) < extrema_tolerance
        @test maximum(abs, interior(∇²ϕ_solution) .- interior(∇²ϕ)) < extrema_tolerance
        @test          std(interior(∇²ϕ_solution) .- interior(∇²ϕ)) < std_tolerance

        @test   minimum(abs, interior(ϕ_solution) .- interior(ϕ_truth)) < extrema_tolerance
        @test   maximum(abs, interior(ϕ_solution) .- interior(ϕ_truth)) < extrema_tolerance
        @test            std(interior(ϕ_solution) .- interior(ϕ_truth)) < std_tolerance
    end

    return nothing
end


@testset "MatrixIterativeSolver" begin

    topologies = [(Periodic, Periodic, Flat), (Bounded, Bounded, Flat), (Periodic, Bounded, Flat), (Bounded, Periodic, Flat)]
    for arch in archs, topo in topologies
        @info "Testing MatrixIterativeSolver [$(typeof(arch)) $topo]..."
        
        grid = RectilinearGrid(architecture = arch, size=(4, 8), extent=(1, 3), topology = topo)
        run_identity_operator_test(arch, grid)
        run_poisson_equation_test(arch, grid)
    end
    topologies = [(Periodic, Periodic, Periodic), (Bounded, Bounded, Periodic), (Periodic, Bounded, Periodic), (Bounded, Periodic, Bounded)]
    for arch in archs, topo in topologies
        @info "Testing MatrixIterativeSolver [$(typeof(arch)) $topo]..."
        
        grid = RectilinearGrid(architecture = arch, size=(4, 8, 6), extent=(1, 3, 4), topology = topo)
        run_identity_operator_test(arch, grid)
        run_poisson_equation_test(arch, grid)
    end
end