using Oceananigans.Architectures: architecture, arch_array
using Oceananigans.Grids: interior_parent_indices, topology
using Oceananigans.Fields: interior_copy
using Oceananigans.Operators:  Δyᶜᶠᵃ,  Δxᶠᶜᵃ
using KernelAbstractions: @kernel, @index
using LinearAlgebra, SparseArrays, IterativeSolvers
using CUDA, CUDA.CUSPARSE
using IncompleteLU

mutable struct MatrixIterativeSolver{A, G, L, D, M, P, I, T, F}
               architecture :: A
                       grid :: G
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
                               maximum_iterations = (grid.Nx * grid.Ny),
                               tolerance = 1e-13,
                               precondition = true)

    arch = grid.architecture

    if iterative_solver == (\) && arch isa GPU
        throw(ArgumentError("Cannot specify a Direct solve on a GPU, it would need scalar indexing!"))
    end

    matrix_constructors, diagonal = matrix_from_coefficients(arch, grid, coeffs)  

    # for the moment, a placeholder preconditioner is calculated using a "placeholder" timestep of 1
    placeholder_constructors = deepcopy(matrix_constructors)
    update_diag!(placeholder_constructors, arch, grid, diagonal, 1.0)

    placeholder_matrix = arch_sparse_matrix(arch, placeholder_constructors)

    if arch isa GPU || !precondition  #until we find a suitable backward substitution for GPU, the preconditioning takes a lot of time!!
        placeholder_preconditioner = Identity()
    else
        placeholder_preconditioner = ilu(placeholder_matrix, τ = 0.1)
    end

    placeholder_timestep = eltype(grid)(-10000.0)
    
    return MatrixIterativeSolver(arch,
                                 grid,
                                 matrix_constructors,
                                 diagonal,
                                 placeholder_matrix,
                                 placeholder_preconditioner,
                                 iterative_solver, 
                                 tolerance,
                                 placeholder_timestep,
                                 maximum_iterations)
end

function matrix_from_coefficients(arch, grid, coeffs)
    Ax, Ay, C = coeffs
    Nx, Ny = (grid.Nx, grid.Ny)

    N = Nx * Ny
    topo = topology(grid)[1:2]

    validate_laplacian_size.((Nx, Ny), topo)

    diag  = arch_array(arch, zeros(eltype(grid), N))

    # the following coefficients are the diagonals of the sparse matrix:
    #  - coeff_d is the main diagonal (coefficents of xᵢⱼ)
    #  - coeff_x is the diagonal at +1  / -1 (coefficients of xᵢ₊₁ and xᵢ₋₁)
    #  - coeff_y is the diagonal at +Nx / -Nx (coefficients of xⱼ₊₁ and xⱼ₋₁)
    #  - boundaries in x are filled in at Nx - 1 / 1 - Nx 
    #  - boundaries in y are filled in at N - Nx / Nx - N 

    coeff_d       = zeros(eltype(grid), N)
    coeff_x       = zeros(eltype(grid), N - 1)
    coeff_y       = zeros(eltype(grid), Nx * (Ny-1))
    coeff_bound_x = zeros(eltype(grid), Nx * (Ny-1) + 1)
    coeff_bound_y = zeros(eltype(grid), Nx)

    # initializing elements which vary during the simulation (as a function of Δt)
    event_c = launch!(arch, grid, :xy, _initialize_variable_diagonal!, diag, C, Nx)
    wait(event_c)

    # filling elements which stay constant in time
    fill_core_matrix!(coeff_d, coeff_x, coeff_y, Ax, Ay, Nx, Ny)
    fill_boundaries_x!(coeff_d, coeff_bound_x, Ax, Nx, Ny, topo[1])
    fill_boundaries_y!(coeff_d, coeff_bound_y, Ay, Nx, Ny, topo[2])

    sparse_matrix = spdiagm(0=>Array(coeff_d),
                            1=>Array(coeff_x),         -1=>Array(coeff_x),
                         Nx-1=>Array(coeff_bound_x),-Nx+1=>Array(coeff_bound_x),
                           Nx=>Array(coeff_y),        -Nx=>Array(coeff_y), 
                         N-Nx=>Array(coeff_bound_y),-N+Nx=>Array(coeff_bound_y))

    dropzeros!(sparse_matrix)

    matrix_constructors = constructors(arch, sparse_matrix)

    return matrix_constructors, diag
end

@kernel function _initialize_variable_diagonal!(diag, C, Nx)  
    i, j = @index(Global, NTuple)
    t  = i + Nx * (j-1)
    diag[t] = C[i, j]
end

function fill_core_matrix!(coeff_d, coeff_x, coeff_y, Ax, Ay, Nx, Ny)
    for j = 1:Ny-1, i = 1:Nx
        t        = i + (j - 1) * Nx
        coeff_y[t]     = Ay[i,j+1] 
        coeff_d[t]    -= coeff_y[t]
        coeff_d[t+Nx] -= coeff_y[t]
    end
    for j = 1:Ny, i = 1:Nx-1
        t       = i + (j - 1) * Nx
        coeff_x[t]   = Ax[i+1,j] 
        coeff_d[t]   -= coeff_x[t]
        coeff_d[t+1] -= coeff_x[t]
    end
end

@inline fill_boundaries_x!(coeff_d, coeff_bound_x, Ax, Nx, Ny, ::Type{Bounded}) = nothing

function fill_boundaries_x!(coeff_d, coeff_bound_x, Ax, Nx, Ny, ::Type{Periodic})
    for j in 1:Ny
        tₘ = 1  + (j-1) * Nx
        tₚ = Nx + (j-1) * Nx
        coeff_bound_x[tₘ] = Ax[1,j]
        coeff_d[tₘ]      -= coeff_bound_x[tₘ]
        coeff_d[tₚ]      -= coeff_bound_x[tₘ]
    end
end


@inline fill_boundaries_y!(coeff_d, coeff_bound_y, Ay, Nx, Ny, ::Type{Bounded}) = nothing

function fill_boundaries_y!(coeff_d, coeff_bound_y, Ay, Nx, Ny, ::Type{Periodic})
    for i in 1:Nx
        tₘ = i + (1  -1) * Nx
        tₚ = i + (Ny -1) * Nx
        coeff_bound_y[tₘ] = Ay[i,1]
        coeff_d[tₘ]      -= coeff_bound_y[tₘ]
        coeff_d[tₚ]      -= coeff_bound_y[tₘ]
    end
end
    
# @inline fill_immersed_boundaries()

function matrix_from_linear_operation(grid, η, fun!, args...)
    arch = grid.architecture

    β = similar(η)

    matrix = sprand(grid.Nx * grid.Ny, grid.Nx * grid.Ny, 0.0)

    colptr, rowval, nzval = unpack_constructors(arch, constructors(arch, matrix))

    for j = 1:grid.Ny, i = 1:grid.Nx
        parent(η) .= 0.0
        η[i, j] = 1.0   
        t = i + (j-1) * Nx
        fun!(β, η, args...)
        sparse_vec = sparse(interior(β)[:])

        colptr[t+1] = colptr[t] + length(sparse_vec.nzval)
        rowval = [rowval..., sparse_vec.nzind...]
        nzval  = [nzval..., sparse_vec.nzval...]
    end

    return constructors(arch, grid.Nx * grid.Ny, (colptr, rowval, nzval))
end

function solve!(x, solver::MatrixIterativeSolver, b, Δt)

    # update matrix and preconditioner if time step changes
    if Δt != solver.previous_Δt
        constructors = deepcopy(solver.matrix_constructors)
        update_diag!(constructors, solver.architecture, solver.grid, solver.diagonal, Δt)
        solver.matrix    = arch_sparse_matrix(solver.architecture, constructors) 
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

    copy_into_x!(solver, x, q)
    fill_halo_regions!(x, solver.architecture) 

    return
end

function copy_into_x!(solver, x, q)
    event = launch!(solver.architecture, solver.grid, :xy, _copy_into_x!, x, q, solver.grid.Nx)
    wait(event)
end

@kernel function _copy_into_x!(x, q, Nx)
    i, j = @index(Global, NTuple)

    x[i, j] = q[i + Nx * (j - 1)]
end

# We need to update the diagonal element each time the time step changes!!
function update_diag!(constr, arch, grid, diagonal, Δt)
    
    col, row, val = unpack_constructors(arch, constr)

    event = launch!(arch, grid, (grid.Nx, grid.Ny), _update_diag!, diagonal, col, row, val, Δt, grid.Nx)
    wait(event)

    constr = constructors(arch, grid.Nx * grid.Ny, (col, row, val))
end

@kernel function _update_diag!(diag, colptr, rowval, nzval, Δt, Nx)
    i, j = @index(Global, NTuple)
    t = i + (j - 1) * Nx
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

@inline function validate_laplacian_size(N, topo)
    if N < 3 && topo == Bounded
        throw(Argumenterror("We cannot solve a Laplacian on a Bounded domain with N < 3"))
    end
end

