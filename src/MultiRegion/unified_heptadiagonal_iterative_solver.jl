using Oceananigans.Architectures
using Oceananigans.Architectures: architecture, arch_array, unified_array, device
using KernelAbstractions: @kernel, @index
using IterativeSolvers, SparseArrays, LinearAlgebra
using Oceananigans.Solvers:
                matrix_from_coefficients,
                update_diag!, 
                arch_sparse_matrix, 
                constructors,
                validate_settings,
                iterating,
                update_diag!

import Oceananigans.Grids: architecture
import Oceananigans.Solvers: solve!, iterate!

mutable struct UnifiedDiagonalIterativeSolver{G, Dev, R, L, D, M, P, F, T, B}
                       grid :: G
                    devices :: Dev
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

    arch = architecture(mrg) 
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
    
    devices = mrg.devices 

    for (idx, dev) in enumerate(devices)
        switch_device!(dev)
        push!(placeholder_matrix, temp_matrix[n*(idx-1)+1:n*idx, :])
        push!(matrix_constructors, constructors(arch, placeholder_matrix[idx]))
        push!(diagonal, arch_array(arch, temp_diagonal[n*(idx-1)+1:n*idx]))
        placeholder_matrix[idx] = arch_sparse_matrix(arch, placeholder_matrix[idx])
    end
    sync_all_devices!(devices)

    return UnifiedDiagonalIterativeSolver(mrg,
                                          devices,
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

architecture(solver::UnifiedDiagonalIterativeSolver) = architecture(solver.grid)

@inline function update_solver!(solver, Δt)
    arch = architecture(solver.grid)
    for (idx, dev) in enumerate(solver.devices)
        switch_device!(dev)
        constr = deepcopy(solver.matrix_constructors[idx])
        update_diag!(constr, arch, solver.n, prod(solver.problem_size), solver.diagonal[idx], Δt, solver.n * (idx - 1))
        solver.matrix[idx] = arch_sparse_matrix(arch, constr)
    end
    sync_all_devices!(solver.devices)
    solver.previous_Δt = Δt
end

function solve!(x, solver::UnifiedDiagonalIterativeSolver, b, Δt, args...)
    
    # arch = architecture(solver.grid)
    # prefetch_solver!(solver, arch)
    
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

    return solver.solution
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
    @show solver.iteration, solver.ρⁱ⁻¹
    return
end

@inline function unified_mul!(q, solver, x)
    for (idx, dev) in enumerate(solver.devices)
        switch_device!(dev)
        @views q[solver.n*(idx-1)+1:solver.n*idx] .= solver.matrix[idx] * x
    end
    sync_all_devices!(solver.devices)
end 

@inline function unified_mul_and_dot!(q, solver, x, ρ)
    α = []
    for (idx, dev) in enumerate(solver.devices)
        switch_device!(dev)
        @views q[solver.n*(idx-1)+1:solver.n*idx] .= solver.matrix[idx] * x
        push!(α, dot(q[solver.n*(idx-1)+1:solver.n*idx], p[solver.n*(idx-1)+1:solver.n*idx])) 
    end
    sync_all_devices!(solver.devices)
    return ρ / sum(α)
end 

@inline prefetch!(array, bytes, dev::CuDevice) = Mem.prefetch(array.storage.buffer, bytes; device = dev)
@inline prefetch!(array, bytes, dev::CPU)      = nothing

@inline prefetch_solver!(solver, ::CPU) = nothing
@inline function prefetch_solver!(solver, ::GPU)
    bytes = sizeof(eltype(solver.grid)) * prod(solver.problem_size)
    for (idx, dev) in enumerate(solver.devices)
        switch_device!(dev)
        Mem.prefetch(solver.solution.storage.buffer,         bytes; device = dev)
        Mem.prefetch(solver.matrix_product.storage.buffer,   bytes; device = dev)
        Mem.prefetch(solver.residual.storage.buffer,         bytes; device = dev)
        Mem.prefetch(solver.search_direction.storage.buffer, bytes; device = dev)
    end
    for (idx, dev) in enumerate(solver.devices)
        sync_device!(dev)
    end
end

function Base.show(io::IO, solver::UnifiedDiagonalIterativeSolver)
    print(io, "Oceananigans-compatible preconditioned conjugate gradient solver.\n")
    print(io, " Problem size = "  , solver.problem_size, '\n')
    print(io, " Grid = "  , solver.grid, "\n")
    print(io, " Devices = ", solver.devices, "\n")
    print(io, " Divided length = ", solver.n)
    return nothing
end
