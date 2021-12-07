using Oceananigans.Architectures
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, is_immersed
using Oceananigans.Architectures: architecture, arch_array
using Oceananigans.Grids: interior_parent_indices, topology
using Oceananigans.Fields: interior_copy
using Oceananigans.Utils: heuristic_workgroup
using KernelAbstractions: @kernel, @index
using LinearAlgebra, SparseArrays, IterativeSolvers
using CUDA, CUDA.CUSPARSE
using IncompleteLU

mutable struct MatrixIterativeSolver{A, G, R, L, D, M, P, I, T, F}
               architecture :: A
                       grid :: G
               problem_size :: R
        matrix_constructors :: L
                   diagonal :: D
                     matrix :: M
             preconditioner :: P
           iterative_solver :: I
                  tolerance :: T
                previous_Δt :: F
         maximum_iterations :: Int
end

"""
MatrixIterativeSolver is a framework to solve the problem A * X = b (provided that A is symmetric)

The solver relies on sparse version of the matrix A which are defined by the field
matrix_constructors.

In particular, given coefficients Ax, Ay, Az, C, D, the solved problem will be

Axᵢ₊₁ ηᵢ₊₁ + Axᵢ ηᵢ₋₁ + Ayⱼ₊₁ ηⱼ₊₁ + Ayⱼ ηⱼ₋₁ + Azₖ₊₁ ηₖ₊₁ + Azₖ ηⱼ₋₁ 
- 2 ( Axᵢ₊₁ + Axᵢ₊₁ + Ayⱼ₊₁ + Ayⱼ + Azₖ₊₁ + Azₖ ) ηᵢⱼₖ 
+   ( Cᵢⱼₖ + Dᵢⱼₖ/Δt^2 ) ηᵢⱼₖ = b

The sparse matrix A can be constructed with

-   SparseMatrixCSC(constructors...) for CPU
- CuSparseMatrixCSC(constructors...) for GPU

The constructors are calculated based on the pentadiagonal coeffients passed as an input (matrix_from_coefficients)

The diagonal term -Az / (g * Δt^2) is added later on during the time stepping
to allow for variable time step. It is updated only when Δt previous != Δt

Preconditioning is done through the incomplete LU factorization. 

It works for GPU, but it relies on serial backward and forward substitution which are very heavy and destroy all the
computational advantage, therefore it is switched off until a parallel backward/forward substitution is implemented
It is also updated based on the matrix when Δt != Δt_previous
    
The iterative_solver used can is to be chosen from the IterativeSolvers.jl package. 
The default solver is a Conjugate Gradient (cg)

"""

function MatrixIterativeSolver(coeffs;
                               grid,
                               iterative_solver = cg,
                               maximum_iterations = prod(size(grid)),
                               tolerance = 1e-13,
                               reduced_dim = (false, false, false), 
                               placeholder_timestep = 1.0, 
                               precondition = true)

    arch = grid.architecture

    if iterative_solver == (\) && arch isa GPU
        throw(ArgumentError("Cannot specify a Direct solve on a GPU, it would need scalar indexing!"))
    end

    matrix_constructors, diagonal, problem_size = matrix_from_coefficients(arch, grid, coeffs, reduced_dim)  

    # for the moment, a placeholder preconditioner is calculated using a "placeholder" timestep of 1
    placeholder_constructors = deepcopy(matrix_constructors)
    update_diag!(placeholder_constructors, arch, problem_size, diagonal, 1.0)

    placeholder_matrix = arch_sparse_matrix(arch, placeholder_constructors)

    if arch isa GPU || !precondition  #until we find a suitable backward substitution for GPU, the preconditioning takes a lot of time!!
        placeholder_preconditioner = Identity()
    else
        placeholder_preconditioner = ilu(placeholder_matrix, τ = 0.1)
    end
   
    return MatrixIterativeSolver(arch,
                                 grid,
                                 problem_size, 
                                 matrix_constructors,
                                 diagonal,
                                 placeholder_matrix,
                                 placeholder_preconditioner,
                                 iterative_solver, 
                                 tolerance,
                                 placeholder_timestep,
                                 maximum_iterations)
end

function matrix_from_coefficients(arch, grid, coeffs, reduced_dim)
    Ax, Ay, Az, C, D = coeffs
    N = size(grid)

    topo = topology(grid)

    dims = validate_laplacian_direction.(N, topo, reduced_dim)
    Nx, Ny, Nz = N = validate_laplacian_size.(N, dims)
    M    = prod(N)
    diag = arch_array(arch, zeros(eltype(grid), M))

    # the following coefficients are the diagonals of the sparse matrix:
    #  - coeff_d is the main diagonal (coefficents of xᵢⱼₖ)
    #  - coeff_x is the diagonal at +1  / -1 (coefficients of xᵢ₊₁ and xᵢ₋₁)
    #  - coeff_y is the diagonal at +Nx / -Nx (coefficients of xⱼ₊₁ and xⱼ₋₁)
    #  - coeff_z is the diagonal at +Nx / -Nx (coefficients of xₖ₊₁ and xₖ₋₁)
    #  - boundaries in x are filled in at Nx - 1 / 1 - Nx 
    #  - boundaries in y are filled in at N - Nx / Nx - N 
    
    # position of diagonals for coefficients d[1] and their boundary d[2]
    dx = (1,     Nx-1)
    dy = (Nx,    Nx*(Ny-1))
    dz = (Ny*Nx, Nx*Ny*(Nz-1))

    coeff_d       = zeros(eltype(grid), M)
    coeff_x       = zeros(eltype(grid), M - dx[1])
    coeff_y       = zeros(eltype(grid), M - dy[1])
    coeff_z       = zeros(eltype(grid), M - dz[1])
    coeff_bound_x = zeros(eltype(grid), M - dx[2])
    coeff_bound_y = zeros(eltype(grid), M - dy[2])
    coeff_bound_z = zeros(eltype(grid), M - dz[2])

    # initializing elements which vary during the simulation (as a function of Δt)
    loop! = _initialize_variable_diagonal!(Architectures.device(arch), heuristic_workgroup(N), N)
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
    adjust_for_immersed_boundaries!(coeff_d, coeff_x, coeff_y, coeff_z, coeff_bound_x, coeff_bound_y, coeff_bound_z, grid, N)

    sparse_matrix = spdiagm(0=>coeff_d,
                        dx[1]=>coeff_x,      -dx[1]=>coeff_x,
                        dx[2]=>coeff_bound_x,-dx[2]=>coeff_bound_x,
                        dy[1]=>coeff_y,      -dy[1]=>coeff_y,
                        dy[2]=>coeff_bound_y,-dy[2]=>coeff_bound_y,
                        dz[1]=>coeff_z,      -dz[1]=>coeff_z,
                        dz[2]=>coeff_bound_z,-dz[2]=>coeff_bound_z)

    dropzeros!(sparse_matrix)

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
    if dims[1]
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

@inline  adjust_for_immersed_boundaries!(cd, cx, cy, cz, cbx, cby, cbz, grid, N) = nothing
function adjust_for_immersed_boundaries!(cd, cx, cy, cz, cbx, cby, cbz, grid::ImmersedBoundaryGrid, N) 
    Nx, Ny, Nz = N
    for k = 1:Nz, j = 1:Ny, i = 1:Nx
        if !is_immersed(i, j, k, grid)
            t = i +  Nx * (j - 1 + Ny * (k - 1))
            cd[t] = 0
            cx[t] = 0
            cy[t] = 0
            cz[t] = 0
            cbx[t] = 0
            cby[t] = 0
            cbz[t] = 0
            cx[t+1] = 0
            cy[t+Nx] = 0
            cz[t+Nx*Ny] = 0
        end
    end 
    return
end

function solve!(x, solver::MatrixIterativeSolver, b, Δt)

    # update matrix and preconditioner if time step changes
    if Δt != solver.previous_Δt
        constructors = deepcopy(solver.matrix_constructors)
        update_diag!(constructors, solver.architecture, solver.problem_size, solver.diagonal, Δt)
        solver.matrix = arch_sparse_matrix(solver.architecture, constructors) 
        if solver.architecture isa CPU && solver.preconditioner != Identity()
            solver.preconditioner = ilu(solver.matrix, τ = 2.0)
        end   
        solver.previous_Δt = Δt
    end
        
    if solver.iterative_solver == (\)
        q = solver.iterative_solver(solver.matrix, b)
    else   
        q = solver.iterative_solver(solver.matrix, b; reltol=solver.tolerance, maxiter=solver.maximum_iterations, Pl=solver.preconditioner)
    end

    set!(x, reshape(q, solver.problem_size...))
    fill_halo_regions!(x, solver.architecture) 

    return
end

# We need to update the diagonal element each time the time step changes!!
function update_diag!(constr, arch, problem_size, diagonal, Δt)
    
    N = problem_size
    M = prod(N)
    col, row, val = unpack_constructors(arch, constr)
   
    loop! = _update_diag!(Architectures.device(arch), heuristic_workgroup(N), N)
    event = loop!(diagonal, col, row, val, Δt, N; dependencies=Event(Architectures.device(arch)))
    wait(event)

    constr = constructors(arch, M, (col, row, val))
end

@kernel function _update_diag!(diag, colptr, rowval, nzval, Δt, N)
    i, j, k = @index(Global, NTuple)
    t = i + N[1] * (j - 1 + N[2] * (k - 1)) 
    map = 1
    for idx in colptr[t]:colptr[t+1] - 1
        if rowval[idx] == t
            map = idx 
            break
        end
    end
    nzval[map] += diag[t] / Δt^2 
end

#unfortunately this cannot run on a GPU so we have to resort to that ugly loop in _update_diag!
@inline map_row_to_diag_element(i, rowval, colptr) =  colptr[i] - 1 + findfirst(rowval[colptr[i]:colptr[i+1]-1] .== i)

function Base.show(io::IO, solver::MatrixIterativeSolver)
    print(io, "matrix-based iterative solver with: \n")
    print(io, " Problem size = "  , size(solver.grid), '\n')
    print(io, " Grid = "  , solver.grid, '\n')
    print(io, " Solution method = ", solver.iterative_solver)
    return nothing
end

@inline function validate_laplacian_direction(N, topo, reduced_dim)
   
    dim = N > 1 && reduced_dim == false
    if N < 3 && topo == Bounded && dim == true
        throw(ArgumentError("cannot calculate laplacian in bounded domain with N < 3"))
    end

    return dim
end

@inline validate_laplacian_size(N, dim) = dim == true ? N : 1
  