using Oceananigans.Architectures: architecture, arch_array
using Oceananigans.Grids: interior_parent_indices
using Oceananigans.Fields: interior_copy
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
                Δt_previous :: F
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
                               maximum_iterations = prod(size(template_field)),
                               tolerance = 1e-13,
                               precondition = true)

    arch = grid.architecture

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

    diag  = arch_array(arch, zeros(N))
    c     = arch_array(arch, zeros(N))
    ai    = arch_array(arch, zeros(N - 1))
    aj    = arch_array(arch, zeros(Nx * (Ny-1)))
    api   = arch_array(arch, zeros(Nx * (Ny-1) + 1))
    apj   = arch_array(arch, zeros(Nx))

    coeff_c_size = (Nx, Ny)
    coeff_x_size = (Nx-1, Ny)
    coeff_y_size = (Nx, Ny-1)

    event_c = launch!(arch, grid, coeff_c_size, _matrix_from_coeff_c!, diag, C, Nx)
    wait(event_c)
    event_x = launch!(arch, grid, coeff_x_size, _matrix_from_coeff_x!, c, ai, Ax, Nx)
    wait(event_x)
    event_x = launch!(arch, grid, coeff_x_size, _matrix_from_coeff_x_plus!, c, Ax, Nx)
    wait(event_x)
    event_y = launch!(arch, grid, coeff_y_size, _matrix_from_coeff_y!, c, aj, Ay, Nx)
    wait(event_y)
    event_y = launch!(arch, grid, coeff_y_size, _matrix_from_coeff_y_plus!, c, Ay, Nx)
    wait(event_y)
    
    if topology(grid)[1] == Periodic
        coeff_period_x_size = (1, Ny)
        event_period_x = launch!(arch, grid, coeff_period_x_size, _matrix_from_period_x!, c, api, Ax, N, Nx)
        wait(event_period_x)
    end

    if topology(grid)[2] == Periodic
        coeff_period_y_size = (Nx, 1)
        event_period_y = launch!(arch, grid, coeff_period_y_size, _matrix_from_period_y!, c, apj, Ay, N, Nx)
        wait(event_period_y)
    end

    sparse_matrix = spdiagm(0=>Array(c),
                            1=>Array(ai),    -1=>Array(ai),
                         Nx-1=>Array(api),-Nx+1=>Array(api),
                           Nx=>Array(aj),   -Nx=>Array(aj), 
                         N-Nx=>Array(apj),-N+Nx=>Array(apj))

    dropzeros!(sparse_matrix)

    matrix_constructors = constructors(arch, sparse_matrix)

    return matrix_constructors, diag
end

@kernel function _matrix_from_coeff_c!(diag, C, Nx)  
    i, j = @index(Global, NTuple)
    t  = i + Nx * (j-1)

    diag[t] = C[i, j]
end

#We have to split in two the kernels for synchronization reasons
@kernel function _matrix_from_coeff_x!(c, ai, Ax, Nx)
    i, j = @index(Global, NTuple)

    t  = i + Nx * (j-1)

    c[t]   -= Ax[i+1, j]
    ai[t]   = Ax[i+1, j]        
end

@kernel function _matrix_from_coeff_y!(c, aj, Ay, Nx)
    i, j = @index(Global, NTuple)

    t  = i + Nx * (j-1)
        
    c[t]    -= Ay[i, j+1]
    aj[t]    = Ay[i, j+1]       
end

@kernel function _matrix_from_coeff_x_plus!(c, Ax, Nx)
    i, j = @index(Global, NTuple)

    t  = i + Nx * (j-1)

    c[t+1] -= Ax[i+1, j]        
end

@kernel function _matrix_from_coeff_y_plus!(c, Ay, Nx)
    i, j = @index(Global, NTuple)

    t  = i + Nx * (j-1)
   
    c[t+Nx] -= Ay[i, j+1] 
end

# here we do not since every point
@kernel function _matrix_from_period_x!(c, api, Ax, N, Nx)
    i, j = @index(Global, NTuple)

    t = 1 + Nx * (j - 1)
    c[t]      -= Ax[i, j]
    c[t+Nx-1] -= Ax[i, j] 
    api[t]     = Ax[i, j]       
end

@kernel function _matrix_from_period_y!(c, apj, Ay, N, Nx)
    i, j = @index(Global, NTuple)

    c[i]      -= Ay[i, j]
    c[i+N-Nx] -= Ay[i, j] 
    apj[i]     = Ay[i, j]       
end

function solve!(x, solver::MatrixIterativeSolver, b, Δt)

    # update matrix and preconditioner if time step changes
    if Δt != solver.Δt_previous
        constructors = deepcopy(solver.matrix_constructors)
        update_diag!(constructors, solver.architecture, solver.grid, solver.diagonal, Δt)
        solver.matrix    = arch_sparse_matrix(solver.architecture, constructors) 
        if solver.architecture isa CPU && solver.preconditioner != Identity()
            solver.preconditioner = ilu(solver.matrix, τ = 0.1)
        end   
        solver.Δt_previous = Δt
    end
        
    q = solver.iterative_solver(solver.matrix, b; reltol=solver.tolerance, maxiter=solver.maximum_iterations, Pl=solver.preconditioner)

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

    x.data[i, j] = q[i + Nx * (j - 1)]
end

# We need to update the diagonal element each time the time step changes!!
function update_diag!(constr, arch, grid, diagonal, Δt)
    
    col, row, val = unpack_constructors(arch, constr)

    event = launch!(arch, grid, :xy, _update_diag!, diagonal, col, row, val, Δt, grid.Nx)
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
