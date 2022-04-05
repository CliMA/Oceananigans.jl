using Oceananigans.Architectures
using Oceananigans.Architectures: architecture, arch_array
using Oceananigans.Grids: interior_parent_indices, topology
using Oceananigans.Utils: heuristic_workgroup
using KernelAbstractions: @kernel, @index
using IterativeSolvers, SparseArrays, LinearAlgebra
using CUDA, CUDA.CUSPARSE

mutable struct HeptadiagonalIterativeSolver{G, R, L, D, M, P, PM, PS, I, T, F}
                       grid :: G
               problem_size :: R
        matrix_constructors :: L
                   diagonal :: D
                     matrix :: M
             preconditioner :: P
      preconditioner_method :: PM
    preconditioner_settings :: PS
           iterative_solver :: I
                  tolerance :: T
                previous_Δt :: F
         maximum_iterations :: Int
end

"""
    HeptadiagonalIterativeSolver(coeffs;
                                 grid,
                                 iterative_solver = cg,
                                 maximum_iterations = prod(size(grid)),
                                 tolerance = 1e-13,
                                 reduced_dim = (false, false, false), 
                                 placeholder_timestep = -1.0, 
                                 preconditioner_method = :Default, 
                                 preconditioner_settings = nothing)

`HeptadiagonalIterativeSolver` is a framework to solve the problem `A * x = b`
(provided that `A` is a symmetric matrix).

The solver relies on sparse version of the matrix `A` which are defined by the
field matrix_constructors.

In particular, given coefficients `Ax`, `Ay`, `Az`, `C`, `D`, the solved problem will be

To have the equation solved on Center, Center, Center, the coefficients should be specified
as follows:

- `Ax` -> Face, Center, Center
- `Ay` -> Center, Face, Center
- `Az` -> Center, Center, Face
- `C`  -> Center, Center, Center
- `D`  -> Center, Center, Center

```julia
Axᵢ₊₁ ηᵢ₊₁ + Axᵢ ηᵢ₋₁ + Ayⱼ₊₁ ηⱼ₊₁ + Ayⱼ ηⱼ₋₁ + Azₖ₊₁ ηₖ₊₁ + Azₖ ηⱼ₋₁ 
- 2 ( Axᵢ₊₁ + Axᵢ + Ayⱼ₊₁ + Ayⱼ + Azₖ₊₁ + Azₖ ) ηᵢⱼₖ 
+   ( Cᵢⱼₖ + Dᵢⱼₖ/Δt^2 ) ηᵢⱼₖ = b
```

`solver.matrix` is precomputed with a value of `Δt = -1.0`

The sparse matrix `A` can be constructed with

- `SparseMatrixCSC(constructors...)` for CPU
- `CuSparseMatrixCSC(constructors...)` for GPU

The constructors are calculated based on the pentadiagonal coeffients passed as an input
(`matrix_from_coefficients`).

The diagonal term `- Az / (g * Δt²)` is added later on during the time stepping
to allow for variable time step. It is updated only when the previous time step 
is different (`Δt_previous != Δt`).

Preconditioning is done through the incomplete LU factorization. 

It works for GPU, but it relies on serial backward and forward substitution which are very
heavy and destroy all the computational advantage, therefore it is switched off until a
parallel backward/forward substitution is implemented. It is also updated based on the
matrix when `Δt != Δt_previous`
    
The iterative_solver used can is to be chosen from the IterativeSolvers.jl package. 
The default solver is a Conjugate Gradient (cg)

```julia
solver = HeptadiagonalIterativeSolver((Ax, Ay, Az, C, D), grid = grid)
```
"""
function HeptadiagonalIterativeSolver(coeffs;
                                      grid,
                                      iterative_solver = cg,
                                      maximum_iterations = prod(size(grid)),
                                      tolerance = 1e-13,
                                      reduced_dim = (false, false, false), 
                                      placeholder_timestep = -1.0, 
                                      preconditioner_method = :Default, 
                                      preconditioner_settings = nothing)

    arch = architecture(grid)

    matrix_constructors, diagonal, problem_size = matrix_from_coefficients(arch, grid, coeffs, reduced_dim)  

    # for the moment, placeholder preconditioner and matrix are calculated using a "placeholder" timestep of 1
    # They will be recalculated before the first time step of the simulation

    placeholder_constructors = deepcopy(matrix_constructors)
    M = prod(problem_size)
    update_diag!(placeholder_constructors, arch, M, M, diagonal, 1.0, 0)

    placeholder_matrix = arch_sparse_matrix(arch, placeholder_constructors)
    
    settings       = validate_settings(Val(preconditioner_method), arch, preconditioner_settings)
    reduced_matrix = arch_sparse_matrix(arch, speye(eltype(grid), 2))
    preconditioner = build_preconditioner(Val(preconditioner_method), reduced_matrix, settings)

    return HeptadiagonalIterativeSolver(grid,
                                 problem_size, 
                                 matrix_constructors,
                                 diagonal,
                                 placeholder_matrix,
                                 preconditioner,
                                 preconditioner_method,
                                 settings,
                                 iterative_solver, 
                                 tolerance,
                                 placeholder_timestep,
                                 maximum_iterations)
end

function matrix_from_coefficients(arch, grid, coeffs, reduced_dim)
    Ax, Ay, Az, C, D = coeffs

    Ax = arch_array(CPU(), Ax)
    Ay = arch_array(CPU(), Ay)
    Az = arch_array(CPU(), Az)
    C  = arch_array(CPU(), C)

    N = size(grid)

    topo = topology(grid)

    dims = validate_laplacian_direction.(N, topo, reduced_dim)
    Nx, Ny, Nz = N = validate_laplacian_size.(N, dims)
    M    = prod(N)
    diag = arch_array(arch, zeros(eltype(grid), M))

    # the following coefficients are the diagonals of the sparse matrix:
    #  - coeff_d is the main diagonal (coefficents of ηᵢⱼₖ)
    #  - coeff_x are the coefficients in the x-direction (coefficents of ηᵢ₋₁ⱼₖ and ηᵢ₊₁ⱼₖ)
    #  - coeff_y are the coefficients in the y-direction (coefficents of ηᵢⱼ₋₁ₖ and ηᵢⱼ₊₁ₖ)
    #  - coeff_z are the coefficients in the z-direction (coefficents of ηᵢⱼₖ₋₁ and ηᵢⱼₖ₊₁)
    #  - periodic boundaries are stored in coeff_bound_
    
    # position of diagonals for coefficients pos[1] and their boundary pos[2]
    posx = (1, Nx-1)
    posy = (1, Ny-1) .* Nx
    posz = (1, Nz-1) .* Nx .* Ny

    coeff_d       = zeros(eltype(grid), M)
    coeff_x       = zeros(eltype(grid), M - posx[1])
    coeff_y       = zeros(eltype(grid), M - posy[1])
    coeff_z       = zeros(eltype(grid), M - posz[1])
    coeff_bound_x = zeros(eltype(grid), M - posx[2])
    coeff_bound_y = zeros(eltype(grid), M - posy[2])
    coeff_bound_z = zeros(eltype(grid), M - posz[2])

    # initializing elements which vary during the simulation (as a function of Δt)
    loop! = _initialize_variable_diagonal!(Architectures.device(arch), heuristic_workgroup(N...), N)
    event = loop!(diag, D, N; dependencies=Event(Architectures.device(arch)))
    wait(event)

    # filling elements which stay constant in time
    fill_core_matrix!(coeff_d , coeff_x, coeff_y, coeff_z, Ax, Ay, Az, C, N, dims)
    if dims[1]  
        fill_boundaries_x!(coeff_d, coeff_bound_x, Ax, N, topo[1])
    end
    if dims[2]
        fill_boundaries_y!(coeff_d, coeff_bound_y, Ay, N, topo[2])
    end
    if dims[3]
        fill_boundaries_z!(coeff_d, coeff_bound_z, Az, N, topo[3])
    end

    sparse_matrix = spdiagm(0=>coeff_d,
                      posx[1]=>coeff_x,      -posx[1]=>coeff_x,
                      posx[2]=>coeff_bound_x,-posx[2]=>coeff_bound_x,
                      posy[1]=>coeff_y,      -posy[1]=>coeff_y,
                      posy[2]=>coeff_bound_y,-posy[2]=>coeff_bound_y,
                      posz[1]=>coeff_z,      -posz[1]=>coeff_z,
                      posz[2]=>coeff_bound_z,-posz[2]=>coeff_bound_z)

    ensure_diagonal_elements_are_present!(sparse_matrix)

    matrix_constructors = constructors(arch, sparse_matrix)

    return matrix_constructors, diag, N
end

@kernel function _initialize_variable_diagonal!(diag, D, N)  
    i, j, k = @index(Global, NTuple)
    t  = i + N[1] * (j - 1 + N[2] * (k - 1))
    diag[t] = D[i, j, k]
end

function fill_core_matrix!(coeff_d, coeff_x, coeff_y, coeff_z, Ax, Ay, Az, C, N, dims)
    Nx, Ny, Nz = N
    for k = 1:Nz, j = 1:Ny, i = 1:Nx
        t          = i +  Nx * (j - 1 + Ny * (k - 1))
        coeff_d[t] = C[i, j, k]
    end
    if dims[1]
        for k = 1:Nz, j = 1:Ny, i = 1:Nx-1
            t             = i +  Nx * (j - 1 + Ny * (k - 1))
            coeff_x[t]    = Ax[i+1, j, k] 
            coeff_d[t]   -= coeff_x[t]
            coeff_d[t+1] -= coeff_x[t]
        end
    end
    if dims[2]
        for k = 1:Nz, j = 1:Ny-1, i = 1:Nx
            t              = i +  Nx * (j - 1 + Ny * (k - 1))
            coeff_y[t]     = Ay[i, j+1, k] 
            coeff_d[t]    -= coeff_y[t] 
            coeff_d[t+Nx] -= coeff_y[t]
        end
    end
    if dims[3]
        for k = 1:Nz-1, j = 1:Ny, i = 1:Nx
            t                 = i +  Nx * (j - 1 + Ny * (k - 1))
            coeff_z[t]        = Az[i, j, k+1] 
            coeff_d[t]       -= coeff_z[t]
            coeff_d[t+Nx*Ny] -= coeff_z[t]
        end
    end
end

# No-flux boundary conditions are implied in the construction of the matrix, so only
# periodic boundary conditions have to be filled in
#
# In case of a periodic boundary condition we have to modify the diagonal
# as well as the off-diagonals. As an example, for x-periodic boundary conditions
# we have to modify the following diagonal elements
#
# row number (1  + Nx * (j - 1 + Ny * (k - 1))) => corresponding to i = 1 , j = j and k = k
# row number (Nx + Nx * (j - 1 + Ny * (k - 1))) => corresponding to i = Nx, j = j and k = k
#
# Since zero-flux BC were implied, we have to also have to add the coefficients corresponding to i-1 and i+1
# (respectively). Since the off-diagonal elements are symmetric we can fill it in only once

 @inline fill_boundaries_x!(coeff_d, coeff_bound_x, Ax, N, ::Type{Bounded}) = nothing
 @inline fill_boundaries_x!(coeff_d, coeff_bound_x, Ax, N, ::Type{Flat})    = nothing
function fill_boundaries_x!(coeff_d, coeff_bound_x, Ax, N, ::Type{Periodic})
    Nx, Ny, Nz = N
    for k = 1:Nz, j = 1:Ny
        tₘ = 1  + Nx * (j - 1 + Ny * (k - 1))
        tₚ = Nx + Nx * (j - 1 + Ny * (k - 1))
        coeff_bound_x[tₘ] = Ax[1, j, k]
        coeff_d[tₘ]      -= coeff_bound_x[tₘ]
        coeff_d[tₚ]      -= coeff_bound_x[tₘ]
    end
end

 @inline fill_boundaries_y!(coeff_d, coeff_bound_y, Ay, N, ::Type{Bounded}) = nothing
 @inline fill_boundaries_y!(coeff_d, coeff_bound_y, Ay, N, ::Type{Flat})    = nothing
function fill_boundaries_y!(coeff_d, coeff_bound_y, Ay, N, ::Type{Periodic})
    Nx, Ny, Nz = N

    for k = 1:Nz, i = 1:Nx
        tₘ = i + Nx * (1 - 1 + Ny * (k - 1))
        tₚ = i + Nx * (Ny- 1 + Ny * (k - 1))
        coeff_bound_y[tₘ] = Ay[i, 1, k]
        coeff_d[tₘ]      -= coeff_bound_y[tₘ]
        coeff_d[tₚ]      -= coeff_bound_y[tₘ]
    end
end
    
 @inline fill_boundaries_z!(coeff_d, coeff_bound_z, Az, N, ::Type{Bounded}) = nothing 
 @inline fill_boundaries_z!(coeff_d, coeff_bound_z, Az, N, ::Type{Flat})    = nothing
function fill_boundaries_z!(coeff_d, coeff_bound_z, Az, N, ::Type{Periodic})
    Nx, Ny, Nz = N
    for j = 1:Ny, i = 1:Nx
        tₘ = i + Nx * (j - 1 + Ny * (1 - 1))
        tₚ = i + Nx * (j - 1 + Ny * (Nz- 1))
        coeff_bound_z[tₘ] = Az[i, j, 1]
        coeff_d[tₘ]      -= coeff_bound_z[tₘ]
        coeff_d[tₚ]      -= coeff_bound_z[tₘ]
    end
end

function solve!(x, solver::HeptadiagonalIterativeSolver, b, Δt)

    arch = architecture(solver.matrix)
    
    # update matrix and preconditioner if time step changes
    if Δt != solver.previous_Δt
        constructors = deepcopy(solver.matrix_constructors)
        M = prod(solver.problem_size)
        update_diag!(constructors, arch, M, M, solver.diagonal, Δt, 0)
        solver.matrix = arch_sparse_matrix(arch, constructors) 
        solver.preconditioner = build_preconditioner(
                            Val(solver.preconditioner_method),
                            solver.matrix,
                            solver.preconditioner_settings)
        solver.previous_Δt = Δt
    end
    
    q = solver.iterative_solver(solver.matrix, b, maxiter=solver.maximum_iterations, reltol=solver.tolerance, Pl=solver.preconditioner)

    return q
end

function Base.show(io::IO, solver::HeptadiagonalIterativeSolver)
    print(io, "Matrix-based iterative solver with: \n")
    print(io, "├── Problem size = "  , solver.problem_size, '\n')
    print(io, "├── Grid = "  , solver.grid, '\n')
    print(io, "├── Solution method = ", solver.iterative_solver, '\n')
    print(io, "└── Preconditioner  = ", solver.preconditioner_method)
    return nothing
end
