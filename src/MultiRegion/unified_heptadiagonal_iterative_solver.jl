using Oceananigans.Architectures
using Oceananigans.Architectures: architecture, arch_array, unified_array, device
using KernelAbstractions: @kernel, @index
using IterativeSolvers, SparseArrays, LinearAlgebra
using Oceananigans.Solvers: 
                Unified,
                matrix_from_coefficients,
                update_diag!, 
                arch_sparse_matrix, 
                constructors,
                validate_settings,
                iterating,
                update_diag!

import Oceananigans.Solvers: solve!, iterate!

##########
##########
##########  Have to fix the Diag element!!!!
##########
##########

mutable struct UnifiedDiagonalIterativeSolver{G, R, L, D, M, P, F, T, B}
                       grid :: G
               problem_size :: R
        matrix_constructors :: L
                   diagonal :: D
                     matrix :: M
                       ρⁱ⁻¹ :: P
                   solution :: F
             matrix_product :: F
           search_direction :: F
                   residual :: F
                  tolerance :: T
                previous_Δt :: B
         maximum_iterations :: Int
                  iteration :: Int
                          n :: Int
end

function UnifiedDiagonalIterativeSolver(coeffs;
                                             grid,
                                             mrg,
                                             maximum_iterations = prod(size(grid)),
                                             tolerance = 1e-13,
                                             reduced_dim = (false, false, false), 
                                             placeholder_timestep = -1.0)

    arch = architecture(mrg) isa GPU ? Unified() : CPU()
    temp_constructors, temp_diagonal, problem_size = matrix_from_coefficients(CPU(), grid, coeffs, reduced_dim)  
    temp_matrix = arch_sparse_matrix(CPU(), temp_constructors)
    N = prod(problem_size)
    M = length(mrg)
    n = div(N, M)

    solution       = similar(unified_array(architecture(mrg), temp_diagonal))
    matrix_product = similar(unified_array(architecture(mrg), temp_diagonal))
    residual       = similar(unified_array(architecture(mrg), temp_diagonal))
    search         = similar(unified_array(architecture(mrg), temp_diagonal))

    placeholder_matrix  = []
    matrix_constructors = []
    diagonal            = []
    
    for (r, dev) in enumerate(mrg.devices)
        switch_device!(dev)
        push!(placeholder_matrix, temp_matrix[n*(r-1)+1:n*r, :])
        push!(matrix_constructors, constructors(arch, placeholder_matrix[r]))
        push!(diagonal, unified_array(architecture(mrg), temp_diagonal[n*(r-1)+1:n*r]))
        placeholder_matrix[r] = arch_sparse_matrix(architecture(mrg), placeholder_matrix[r])
    end

    return UnifiedDiagonalIterativeSolver(mrg,
                                          problem_size, 
                                          matrix_constructors,
                                          diagonal,
                                          placeholder_matrix,
                                          zero(eltype(grid)),
                                          solution,
                                          matrix_product, 
                                          search,
                                          residual,
                                          tolerance,
                                          placeholder_timestep,
                                          maximum_iterations,
                                          0,
                                          n)
end

@inline function update_solver!(solver, Δt)
    arch = architecture(solver.grid)
    constructors = deepcopy(solver.matrix_constructors)
    for r in 1:length(solver.grid)
        update_diag!(constructors[r], arch, solver.n, solver.diagonal[r], Δt, solver.n * (r - 1))
        solver.matrix[r] = arch_sparse_matrix(arch, constructors[r])
    end
    sync_device!(solver.grid.devices[1])
    solver.previous_Δt = Δt
end

function solve!(x, solver::UnifiedDiagonalIterativeSolver, b, Δt, args...)
    
    prefetch_solver!(solver, architecture(solver.grid))
    
    # @apply_regionally redistribute!(solver.solution, x, solver.grid, solver.n, Iterate(1:length(solver.grid)), _linearize_multi_field!)
    # update matrix and preconditioner if time step changes
    # if Δt != solver.previous_Δt
    #     update_solver!(solver, Δt)
    # end    

    # Initialize
    solver.iteration = 0

    # q = A*x
    q = solver.matrix_product
    unified_mul!(q, solver, solver.solution)

    # r = b - A*x
    solver.residual .= b .- q

    while iterating(solver)
        iterate!(solver.solution, solver, args...)
    end

    # @apply_regionally redistribute!(solver.solution, x, solver.grid, solver.n, Iterate(1:length(solver.grid)), _redistribute_linear_solution!)
    # fill_halo_regions!(x) # blocking

    return nothing
end

function iterate!(x, solver::UnifiedDiagonalIterativeSolver, args...)
    q = solver.matrix_product
    r = solver.residual
    p = solver.search_direction

    z = r
    ρ = dot(z, r)

    if solver.iteration == 0
        p .= z
    else
        β = ρ / solver.ρⁱ⁻¹
        p .= z .+ β .* p
    end

    # q = A*x
    unified_mul!(q, solver, p)

    α = ρ / dot(p, q)

    x .+= α .* p
    r .-= α .* q

    solver.iteration += 1
    solver.ρⁱ⁻¹ = ρ
    
    return
end

@inline function unified_mul!(q, solver, x)
    for (idx, dev) in enumerate(solver.grid.devices)
            # switch_device!(dev)
            @views q[solver.n*(idx-1)+1:solver.n*idx] .= solver.matrix[idx] * x
    end
    # for (idx, dev) in enumerate(solver.grid.devices)
    #     sync_device!(dev)
    # end
end 

@inline prefetch_solver!(solver, ::CPU) = nothing

@inline function prefetch_solver!(solver, ::GPU)
    bytes = sizeof(eltype(solver.grid)) * prod(solver.problem_size)
    for (idx, dev) in enumerate(solver.grid.devices)
        switch_device!(dev)
        Mem.prefetch(solver.solution.storage.buffer,         bytes; device = dev)
        Mem.prefetch(solver.matrix_product.storage.buffer,   bytes; device = dev)
        Mem.prefetch(solver.residual.storage.buffer,         bytes; device = dev)
        Mem.prefetch(solver.search_direction.storage.buffer, bytes; device = dev)
    end
    for (idx, dev) in enumerate(solver.grid.devices)
        sync_device!(dev)
    end
end

function Base.show(io::IO, solver::UnifiedDiagonalIterativeSolver)
    print(io, "Oceananigans-compatible preconditioned conjugate gradient solver.\n")
    print(io, " Problem size = "  , solver.problem_size, '\n')
    print(io, " Grid = "  , solver.grid, "\n")
    print(io, "Divided length = ", solver.n)
    return nothing
end
