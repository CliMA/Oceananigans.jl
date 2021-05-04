using Oceananigans.Operators: Δzᵃᵃᶜ, Δzᵃᵃᶠ

struct FourierTridiagonalPoissonSolver{A, G, B, S, β, T}
                  architecture :: A
                          grid :: G
    batched_tridiagonal_solver :: B
                       storage :: S
                        buffer :: β
                    transforms :: T
end

@kernel function compute_diagonals!(D, grid, λx, λy)
    i, j = @index(Global, NTuple)
    Nz = grid.Nz

    # Using a homogeneous Neumann (zero Gradient) boundary condition:
    D[i, j, 1] = -1 / Δzᵃᵃᶠ(i, j, 2, grid) - Δzᵃᵃᶜ(i, j, 1, grid) * (λx[i] + λy[j])

    @unroll for k in 2:Nz-1
        D[i, j, k] = - (1 / Δzᵃᵃᶠ(i, j, k+1, grid) + 1 / Δzᵃᵃᶠ(i, j, k, grid)) - Δzᵃᵃᶜ(i, j, k, grid) * (λx[i] + λy[j])
    end

    D[i, j, Nz] = -1 / Δzᵃᵃᶠ(i, j, Nz, grid) - Δzᵃᵃᶜ(i, j, Nz, grid) * (λx[i] + λy[j])
end

function FourierTridiagonalPoissonSolver(arch, grid, planner_flag=FFTW.PATIENT)
    TX, TY, TZ = topology(grid)
    TZ != Bounded && error("FourierTridiagonalPoissonSolver can only be used with a Bounded z topology.")

    if grid isa VerticallyStretchedRectilinearGrid && any([T() isa Flat for T in (TX, TY)])
        @warn "FourierTridiagonalPoissonSolver is probably wrong for topologies that contain " *
              "Flat dimensions."
    end

    Nx, Ny, Nz = size(grid)

    # Compute discrete Poisson eigenvalues
    λx = poisson_eigenvalues(grid.Nx, grid.Lx, 1, TX())
    λy = poisson_eigenvalues(grid.Ny, grid.Ly, 2, TY())

    λx = arch_array(arch, λx)
    λy = arch_array(arch, λy)

    # Plan required transforms for x and y
    sol_storage = arch_array(arch, zeros(complex(eltype(grid)), size(grid)...))
    transforms = plan_transforms(arch, grid, sol_storage, planner_flag)

    # Lower and upper diagonals are the same
    CUDA.allowscalar(true)
    lower_diagonal = CUDA.@allowscalar [1 / Δzᵃᵃᶠ(1, 1, k, grid) for k in 2:Nz]
    lower_diagonal = arch_array(arch, lower_diagonal)
    upper_diagonal = lower_diagonal
    CUDA.allowscalar(false)

    # Compute diagonal coefficients for each grid point
    diagonal = arch_array(arch, zeros(Nx, Ny, Nz))
    event = launch!(arch, grid, :xy, compute_diagonals!, diagonal, grid, λx, λy,
                    dependencies=Event(device(arch)))
    wait(device(arch), event)

    # Set up batched tridiagonal solver
    rhs_storage = arch_array(arch, zeros(complex(eltype(grid)), size(grid)...))

    btsolver = BatchedTridiagonalSolver(arch,
                                        grid = grid,
                                          dl = lower_diagonal,
                                           d = diagonal,
                                          du = upper_diagonal,
                                           f = rhs_storage)

    # Need buffer for index permutations and transposes.
    buffer_needed = arch isa GPU && Bounded in (TX, TY) ? true : false
    buffer = buffer_needed ? similar(sol_storage) : nothing

    return FourierTridiagonalPoissonSolver(arch, grid, btsolver, sol_storage, buffer, transforms)
end

function solve_poisson_equation!(solver::FourierTridiagonalPoissonSolver)
    ϕ = solver.storage
    RHS = solver.batched_tridiagonal_solver.f

    # Apply forward transforms in order
    [transform!(RHS, solver.buffer) for transform! in solver.transforms.forward]

    # Solve tridiagonal system of linear equations in z at every column.
    solve_batched_tridiagonal_system!(ϕ, solver.architecture, solver.batched_tridiagonal_solver)

    # Apply backward transforms in order
    [transform!(ϕ, solver.buffer) for transform! in solver.transforms.backward]

    ϕ .= real.(ϕ)

    # Set the volume mean of the solution to be zero.
    # Solutions to Poisson's equation are only unique up to a constant (the global mean
    # of the solution), so we need to pick a constant. We choose the constant to be zero
    # so that the solution has zero-mean.
    ϕ .= ϕ .- mean(ϕ)

    return nothing
end

@kernel function calculate_pressure_source_term_fourier_tridiagonal_solver!(RHS, grid, Δt, U★)
    i, j, k = @index(Global, NTuple)

    @inbounds RHS[i, j, k] = Δzᵃᵃᶜ(i, j, k, grid) * divᶜᶜᶜ(i, j, k, grid, U★.u, U★.v, U★.w) / Δt
end

"""
    set_source_term!(solver, source_term)

Sets the `source_term` ``R`` in the discrete Poisson equation `solver`.
The Poisson equation is

```math
\\nabla^2 \\phi = R
```

where ``\\nabla^2`` is the Laplacian, ``R`` is the Poisson `source_term`,
and ``\\phi`` is the solution to the Poisson equation.
"""
function set_source_term!(solver::FourierTridiagonalPoissonSolver, source_term)
    grid = solver.grid
    arch = solver.architecture

    solver_rhs = solver.batched_tridiagonal_solver.f
    solver_rhs .= source_term

    event = launch!(arch, grid, :xyz, multiply_by_Δzᵃᵃᶜ!, solver_rhs, grid, dependencies=Event(device(arch)))
    wait(device(arch), event)

    return nothing
end

@kernel function multiply_by_Δzᵃᵃᶜ!(a, grid)
    i, j, k = @index(Global, NTuple)
    a[i, j, k] *= Δzᵃᵃᶜ(i, j, k, grid)
end
