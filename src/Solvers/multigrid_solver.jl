using Oceananigans.Architectures: architecture
using LinearAlgebra
using AlgebraicMultigrid: _solve!, init, RugeStubenAMG
using CUDA
using AMGX

import Oceananigans.Architectures: architecture

abstract type MultigridSolver{A, G, L, T, F} end

mutable struct MultigridCPUSolver{A, G, L, T, F, M} <: MultigridSolver{A, G, L, T, F}
                  architecture :: A
                          grid :: G
                        matrix :: L
                        abstol :: T
                        reltol :: T
                       maxiter :: Int
                       x_array :: F
                       b_array :: F
                 amg_algorithm :: M
end

mutable struct MultigridGPUSolver{A, G, L, T, F, S} <: MultigridSolver{A, G, L, T, F}
                  architecture :: A
                          grid :: G
                        matrix :: L
                        abstol :: T
                        reltol :: T
                       maxiter :: Int
                       x_array :: F
                       b_array :: F
                   amgx_solver :: S
end

mutable struct AMGXMultigridSolver{C, R, S, M, V, A}
                         config :: C
                      resources :: R
                         solver :: S
                  device_matrix :: M
                       device_x :: V
                       device_b :: V
                     csr_matrix :: A
end

architecture(solver::MultigridSolver) = solver.architecture
    
"""
    MultigridSolver(linear_operation!::Function,
                    args...;
                    template_field::AbstractField,
                    maxiter = prod(size(template_field)),
                    reltol = sqrt(eps(eltype(template_field.grid))),
                    abstol = 0,
                    amg_algorithm = RugeStubenAMG()
                    )

Returns a MultigridSolver that solves the linear equation
``A x = b`` using a multigrid method, where `A * x` is
determined by `linear_operation!`

`linear_operation!` is a function with signature `linear_operation!(Ax, x, args...)` 
that calculates `A * x` for given `x` and stores the result in `Ax`.

The solver is used by calling

```
solve!(x, solver::MultigridSolver, b; kwargs...)
```

for `solver`, right-hand side `b`, solution `x`, and optional keyword arguments `kwargs...`.

Arguments
=========

* `template_field`: Dummy field that is the same type and size as `x` and `b`, which
                    is used to infer the `architecture`, `grid`, and to create work arrays
                    that are used internally by the solver.

* `maxiter`: Maximum number of iterations the solver may perform before exiting.

* `reltol, abstol`: Relative and absolute tolerance for convergence of the algorithm.
                    The iteration stops when `norm(A * x - b) < max(reltol * norm(b), abstol)`.

* `amg_algorithm`: Algebraic Multigrid algorithm defining mapping between different grid spacings.
                   Note: This keyword is relevant only on the CPU.
"""
function MultigridSolver(linear_operation!::Function,
                         args...;
                         template_field::AbstractField,
                         maxiter = prod(size(template_field)),
                         reltol = 0.1*sqrt(eps(eltype(template_field.grid))),
                         abstol = 0,
                         amg_algorithm = RugeStubenAMG(),
                         matrix = nothing,
                         )

    arch = architecture(template_field)
    
    if matrix === nothing
        matrix = initialize_matrix(arch, template_field, linear_operation!, args...)
    end

    Nx, Ny, Nz = size(template_field)

    FT = eltype(template_field.grid)

    b_array = arch_array(arch, zeros(FT, Nx * Ny * Nz))
    x_array = arch_array(arch, zeros(FT, Nx * Ny * Nz))

    arch == CPU() && return MultigridCPUSolver(arch,
                                               template_field.grid,
                                               matrix,
                                               FT(abstol),
                                               FT(reltol),
                                               maxiter,
                                               x_array,
                                               b_array,
                                               amg_algorithm
                                               )

    arch == GPU() && begin
        try
            AMGX.initialize()
            AMGX.initialize_plugins()
        catch e
            AMGX.finalize_plugins()
            AMGX.finalize()        
            AMGX.initialize()
            AMGX.initialize_plugins()
        end

        if reltol == 0
            @info "Multigrid solver will use abstol = ($abstol)"
            config = AMGX.Config(Dict("monitor_residual" => 1, "max_iters" => maxiter, "store_res_history" => 1, "tolerance" => abstol))
        else
            @info "Multigrid solver will use reltol = ($reltol)"
            config = AMGX.Config(Dict("monitor_residual" => 1, "max_iters" => maxiter, "store_res_history" => 1, "tolerance" => reltol, "convergence" => "RELATIVE_INI_CORE"))
        end
        resources = AMGX.Resources(config)
        solver = AMGX.Solver(resources, AMGX.dDDI, config)
        device_matrix = AMGX.AMGXMatrix(resources, AMGX.dDDI)
        device_b = AMGX.AMGXVector(resources, AMGX.dDDI)
        device_x = AMGX.AMGXVector(resources, AMGX.dDDI)
        csr_matrix = CuSparseMatrixCSR(transpose(matrix))
        
        # # FIXME do on GPU
        # loop! = zero_index!(Architectures.device(arch), 16, length(csr_matrix.rowPtr))
        # event = loop!(csr_matrix.rowPtr; dependencies=device_event(arch))
        @inline subtract_one(x) = convert(Int32, x-1)
        
        AMGX.upload!(device_matrix, 
                     map(subtract_one, csr_matrix.rowPtr), # annoyingly arrays need to be 0-indexed rather than 1-indexed
                     map(subtract_one, csr_matrix.colVal),
                     csr_matrix.nzVal
                     )
        
        AMGX.setup!(solver, device_matrix)

        amgx_solver = AMGXMultigridSolver(config,
                                          resources,
                                          solver,
                                          device_matrix,
                                          device_x,
                                          device_b,
                                          csr_matrix
                                          )

        return MultigridGPUSolver(arch,
                                  template_field.grid,
                                  matrix,
                                  FT(abstol),
                                  FT(reltol),
                                  maxiter,
                                  x_array,
                                  b_array,
                                  amgx_solver
                                  )
    end
end

# @kernel function zero_index!(cuarray :: CuArray)
#     i = @index(Global, Linear)
#     cuarray[i] = convert(Int32, cuarray[i] - 1)
# end

function initialize_matrix(::CPU, template_field, linear_operator!, args...)
    Nx, Ny, Nz = size(template_field)
    A = spzeros(eltype(template_field.grid), Nx*Ny*Nz, Nx*Ny*Nz)
    make_column(f) = reshape(interior(f), Nx*Ny*Nz)

    eᵢⱼₖ = similar(template_field)
    ∇²eᵢⱼₖ = similar(template_field)
    
    for k = 1:Nz, j in 1:Ny, i in 1:Nx
        parent(eᵢⱼₖ) .= 0
        parent(∇²eᵢⱼₖ) .= 0
        eᵢⱼₖ[i, j, k] = 1
        fill_halo_regions!(eᵢⱼₖ)
        linear_operator!(∇²eᵢⱼₖ, eᵢⱼₖ, args...)

        A[:, Ny*Nx*(k-1) + Nx*(j-1) + i] .= make_column(∇²eᵢⱼₖ)
    end
    
    return A
end

function initialize_matrix(::GPU, template_field, linear_operator!, args...)
    Nx, Ny, Nz = size(template_field)
    FT = eltype(template_field.grid)

    make_column(f) = reshape(interior(f), Nx*Ny*Nz)

    eᵢⱼₖ = similar(template_field)
    ∇²eᵢⱼₖ = similar(template_field)

    colptr = CuArray{Int}(undef, Nx*Ny*Nz+1)  # Can we infer the type of int?
    rowval = CuArray{Int}(undef, 0)
    nzval  = CuArray{FT}(undef, 0)

    CUDA.@allowscalar colptr[1] = 1

    for k = 1:Nz, j in 1:Ny, i in 1:Nx
        parent(eᵢⱼₖ) .= 0
        parent(∇²eᵢⱼₖ) .= 0
        CUDA.@allowscalar eᵢⱼₖ[i, j, k] = 1
        fill_halo_regions!(eᵢⱼₖ)
        linear_operator!(∇²eᵢⱼₖ, eᵢⱼₖ, args...)
        count = 0
        for n = 1:Nz, m in 1:Ny, l in 1:Nx
            CUDA.@allowscalar if ∇²eᵢⱼₖ[l, m, n] != 0
                append!(rowval, Ny*Nx*(n-1) + Nx*(m-1) + l)
                CUDA.@allowscalar append!(nzval, ∇²eᵢⱼₖ[l, m, n])
                count += 1
            end
        end
        CUDA.@allowscalar colptr[Ny*Nx*(k-1) + Nx*(j-1) + i + 1] = colptr[Ny*Nx*(k-1) + Nx*(j-1) + i] + count
    end

    return CuSparseMatrixCSC(colptr, rowval, nzval, (Nx*Ny*Nz, Nx*Ny*Nz))
end

"""
    solve!(x, solver::MultigridCPUSolver, b; kwargs...)

Solve `A * x = b` using a multigrid method on CPU, where `A * x` is determined
by `solver.matrix`.
"""
function solve!(x, solver::MultigridCPUSolver, b; kwargs...)
    Nx, Ny, Nz = size(b)

    solver.b_array .= reshape(interior(b), Nx * Ny * Nz)
    solver.x_array .= reshape(interior(x), Nx * Ny * Nz)

    solt = init(solver.amg_algorithm, solver.matrix, solver.b_array)

    _solve!(solver.x_array, solt.ml, solt.b,
            maxiter = solver.maxiter,
            abstol = solver.abstol,
            reltol = solver.reltol,
            kwargs...)

    interior(x) .= reshape(solver.x_array, Nx, Ny, Nz)
end

"""
    solve!(x, solver::MultigridGPUSolver, b; kwargs...)

Solve `A * x = b` using a multigrid method on GPU, where `A * x` is determined
by `solver.matrix`.
"""
function solve!(x, solver::MultigridGPUSolver, b; kwargs...)
    Nx, Ny, Nz = size(b)

    solver.b_array .= reshape(interior(b), Nx * Ny * Nz)
    solver.x_array .= reshape(interior(x), Nx * Ny * Nz)

    s = solver.amgx_solver
    AMGX.upload!(s.device_b, solver.b_array)
    AMGX.upload!(s.device_x, solver.x_array)
    AMGX.setup!(s.solver, s.device_matrix)
    AMGX.solve!(s.device_x, s.solver, s.device_b)
    AMGX.copy!(solver.x_array, s.device_x)

    interior(x) .= reshape(solver.x_array, Nx, Ny, Nz)
end

function finalize_solver!(solver::MultigridGPUSolver)
    s = solver.amgx_solver
    close(s.device_matrix)
    close(s.device_x)
    close(s.device_b)
    close(s.solver)
    close(s.resources)
    close(s.config) 
    AMGX.finalize_plugins()
    AMGX.finalize()
end

finalize_solver!(::MultigridCPUSolver) = nothing

Base.show(io::IO, solver::MultigridSolver) = 
print(io, "MultigridSolver on ", string(typeof(architecture(solver))), ": \n",
              "├── grid: ", summary(solver.grid), "\n",
              "├── matrix: ", prettysummary(solver.matrix), "\n",
              "├── reltol: ", prettysummary(solver.reltol), "\n",
              "├── abstol: ", prettysummary(solver.abstol), "\n",
              "└── maxiter: ", solver.maxiter, "\n",)
