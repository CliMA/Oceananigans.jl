using Oceananigans.Operators: Δzᶜᶜᶜ, Δzᶜᶜᶠ
using Oceananigans.Architectures: device_event
import Oceananigans.Architectures: architecture

struct FourierTridiagonalPoissonSolver{G, B, R, S, β, T}
    grid :: G
    batched_tridiagonal_solver :: B
    source_term :: R
    storage :: S
    buffer :: β
    transforms :: T
end

architecture(solver::FourierTridiagonalPoissonSolver) = architecture(solver.grid)

@kernel function compute_main_diagonals!(D, grid, λx, λy)
    i, j = @index(Global, NTuple)
    Nz = grid.Nz

    @inbounds begin
        # Using a homogeneous Neumann (zero Gradient) boundary condition:
        D[i, j, 1] = -1 / Δzᶜᶜᶠ(i, j, 2, grid) - Δzᶜᶜᶜ(i, j, 1, grid) * (λx[i] + λy[j])

        @unroll for k in 2:Nz-1
            D[i, j, k] = - (1 / Δzᶜᶜᶠ(i, j, k+1, grid) + 1 / Δzᶜᶜᶠ(i, j, k, grid)) - Δzᶜᶜᶜ(i, j, k, grid) * (λx[i] + λy[j])
        end

        D[i, j, Nz] = -1 / Δzᶜᶜᶠ(i, j, Nz, grid) - Δzᶜᶜᶜ(i, j, Nz, grid) * (λx[i] + λy[j])
    end
end

function compute_batched_tridiagonals(grid, λx, λy)
    # Lower and upper diagonals are identical and independent of (i, j)
    Nx, Ny, Nz = size(grid)
    arch = architecture(grid)
    lower_diagonal = CUDA.@allowscalar [1 / Δzᶜᶜᶠ(1, 1, k, grid) for k in 2:Nz]
    lower_diagonal = arch_array(arch, lower_diagonal)
    upper_diagonal = lower_diagonal

    # Diagonal coefficients vary in horizontal due to variation of eigenvalues λx, λy
    diagonal = arch_array(arch, zeros(Nx, Ny, Nz))
    event = launch!(arch, grid, :xy, compute_main_diagonals!, diagonal, grid, λx, λy, dependencies=device_event(arch))
    wait(device(arch), event)

    return lower_diagonal, diagonal, upper_diagonal
end


"""
    FourierTridiagonalPoissonSolver(grid, planner_flag=FFTW.PATIENT)

Return a solver for the Poisson equation which uses Fourier transforms in the horizontal
and a tridiagonal solve in the vertical.
"""
function FourierTridiagonalPoissonSolver(grid, planner_flag=FFTW.PATIENT)
    TX, TY, TZ = topology(grid)
    TZ != Bounded && error("FourierTridiagonalPoissonSolver can only be used with a Bounded z topology.")

    Nx, Ny, Nz = size(grid)

    # Compute discrete Poisson eigenvalues
    λx = poisson_eigenvalues(grid.Nx, grid.Lx, 1, TX())
    λy = poisson_eigenvalues(grid.Ny, grid.Ly, 2, TY())

    arch = architecture(grid)
    λx = arch_array(arch, λx)
    λy = arch_array(arch, λy)

    # Plan required transforms for x and y
    sol_storage = arch_array(arch, zeros(complex(eltype(grid)), size(grid)...))
    transforms = plan_transforms(grid, sol_storage, planner_flag)

    lower_diagonal, diagonal, upper_diagonal = compute_batched_tridiagonals(grid, λx, λy)
    btsolver = BatchedTridiagonalSolver(grid; lower_diagonal, diagonal, upper_diagonal)
    
    # Need buffer for index permutations and transposes.
    buffer_needed = arch isa GPU && Bounded in (TX, TY)
    buffer = buffer_needed ? similar(sol_storage) : nothing

    # Storage space for right hand side of Poisson equation
    rhs = arch_array(arch, zeros(complex(eltype(grid)), size(grid)...))

    return FourierTridiagonalPoissonSolver(grid, btsolver, rhs, sol_storage, buffer, transforms)
end

function solve!(x, solver::FourierTridiagonalPoissonSolver, b=nothing)
    !isnothing(b) && set_source_term!(solver, b) # otherwise, assume source term is set correctly

    arch = architecture(solver)
    ϕ = solver.storage

    # Apply forward transforms in order
    for transform! in solver.transforms.forward
        transform!(solver.source_term, solver.buffer)
    end

    # Solve tridiagonal system of linear equations in z at every column.
    solve!(ϕ, solver.batched_tridiagonal_solver, solver.source_term)

    # Apply backward transforms in order
    for transform! in solver.transforms.backward
        transform!(ϕ, solver.buffer)
    end

    # TODO: is there a better way to set mean to 0?
    mean_ϕ = mean(real, ϕ)
    copy_event = launch!(arch, solver.grid, :xyz, copy_real_subtract_mean!, x, ϕ, mean_ϕ, indices(x), dependencies=device_event(arch))
    wait(device(arch), copy_event)

    return nothing
end

@kernel function copy_real_subtract_mean!(ϕ, ϕc, mean_ϕ, index_ranges)
    i, j, k = @index(Global, NTuple)

    i′ = offset_compute_index(index_ranges[1], i)
    j′ = offset_compute_index(index_ranges[2], j)
    k′ = offset_compute_index(index_ranges[3], k)

    @inbounds ϕ[i′, j′, k′] = real(ϕc[i, j, k] - mean_ϕ)
end

"""
    set_source_term!(solver, source_term)

Sets the source term in the discrete Poisson equation `solver` to `source_term` by
multiplying it by the vertical grid spacing at ``z`` cell centers.
"""
function set_source_term!(solver::FourierTridiagonalPoissonSolver, user_source_term)
    grid = solver.grid
    arch = architecture(solver)

    event = launch!(arch, grid, :xyz,
                    _set_tridiagonal_source_term!, solver.source_term, user_source_term, grid,
                    dependencies = device_event(arch))

    wait(device(arch), event)

    return nothing
end

@kernel function _set_tridiagonal_source_term!(solver_source_term, user_source_term, grid)
    i, j, k = @index(Global, NTuple)
    @inbounds solver_source_term[i, j, k] = user_source_term[i, j, k] * Δzᶜᶜᶜ(i, j, k, grid)
end

