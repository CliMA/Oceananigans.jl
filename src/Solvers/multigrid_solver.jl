using Oceananigans.Architectures: architecture
using LinearAlgebra
using AlgebraicMultigrid: _solve!, RugeStubenAMG, ruge_stuben, SmoothedAggregationAMG, smoothed_aggregation
using CUDA
using AMGX

import Oceananigans.Architectures: architecture

"Boolean denoting whether AMGX.jl can be loaded on machine."
const hasamgx   = @static (Sys.islinux() && Sys.ARCH == :x86_64) ? true : false

abstract type MultigridSolver{A, G, L, T, F} end

mutable struct MultigridCPUSolver{A, G, L, T, F, R, M} <: MultigridSolver{A, G, L, T, F}
                  architecture :: A
                          grid :: G
                        matrix :: L
                        abstol :: T
                        reltol :: T
                       maxiter :: Int
                       x_array :: F
                       b_array :: F
                 amg_algorithm :: R
                            ml :: M
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
                   Options are `RugeStubenAMG()` (default) or `SmoothedAggregationAMG()`.
                   Note: This keyword is relevant *only* on the CPU.
"""
function MultigridSolver(linear_operation!::Function,
                         args...;
                         template_field::AbstractField,
                         maxiter = prod(size(template_field)),
                         reltol = sqrt(eps(eltype(template_field.grid))),
                         abstol = 0,
                         amg_algorithm = RugeStubenAMG(),
                         )

    arch = architecture(template_field)

    (arch == GPU() && !hasamgx) && error("Multigrid on the GPU requires a linux operating system due to AMGX.jl")

    matrix = compute_matrix_for_linear_operation(arch, template_field, linear_operation!, args...)

    return  MultigridSolver(matrix; template_field, maxiter, reltol, abstol, amg_algorithm)
end

function MultigridSolver(matrix;
                         template_field::AbstractField,
                         maxiter = prod(size(template_field)),
                         reltol = sqrt(eps(eltype(template_field.grid))),
                         abstol = 0,
                         amg_algorithm = RugeStubenAMG(),
                         )

    arch = architecture(template_field)

    (arch == GPU() && !hasamgx) && error("Multigrid on the GPU requires a linux operating system due to AMGX.jl")

    Nx, Ny, Nz = size(template_field)

    FT = eltype(template_field.grid)

    b_array = arch_array(arch, zeros(FT, Nx * Ny * Nz))
    x_array = arch_array(arch, zeros(FT, Nx * Ny * Nz))

    return MultigridSolver_on_architecture(arch; template_field, maxiter, reltol=FT(reltol), abstol=FT(abstol),
                                           amg_algorithm, matrix, x_array, b_array)

end

function MultigridSolver_on_architecture(::CPU;
                                         template_field::AbstractField,
                                         maxiter,
                                         reltol,
                                         abstol,
                                         amg_algorithm,
                                         matrix,
                                         x_array,
                                         b_array
                                         )
    ml = create_multilevel(amg_algorithm, matrix)

    return MultigridCPUSolver(CPU(),
                              template_field.grid,
                              matrix,
                              abstol,
                              reltol,
                              maxiter,
                              x_array,
                              b_array,
                              amg_algorithm,
                              ml
                              )
end

function MultigridSolver_on_architecture(::GPU;
                                         template_field::AbstractField,
                                         maxiter,
                                         reltol,
                                         abstol,
                                         amg_algorithm,
                                         matrix,
                                         x_array,
                                         b_array
                                         )

    amgx_solver = AMGXMultigridSolver(matrix, maxiter, reltol, abstol)

    return MultigridGPUSolver(GPU(),
                              template_field.grid,
                              matrix,
                              abstol,
                              reltol,
                              maxiter,
                              x_array,
                              b_array,
                              amgx_solver
                              )
end

function AMGXMultigridSolver(matrix::CuSparseMatrixCSC, maxiter = 1, reltol = sqrt(eps(eltype(matrix))), abstol = 0)
    tolerance, convergence = reltol == 0 ? (abstol, "ABSOLUTE") : (reltol, "RELATIVE_INI_CORE")
    try
        global config = AMGX.Config(Dict("monitor_residual" => 1, "max_iters" => maxiter, "store_res_history" => 1, "tolerance" => tolerance, "convergence" => convergence))
    catch e 
        @info "It appears you are using the multigrid solver on GPU. Have you called `initialize_AMGX()`?"
        AMGX.initialize()
        AMGX.initialize_plugins()
        global config = AMGX.Config(Dict("monitor_residual" => 1, "max_iters" => maxiter, "store_res_history" => 1, "tolerance" => tolerance, "convergence" => convergence))
    end
    resources = AMGX.Resources(config)
    solver = AMGX.Solver(resources, AMGX.dDDI, config)
    device_matrix = AMGX.AMGXMatrix(resources, AMGX.dDDI)
    device_b = AMGX.AMGXVector(resources, AMGX.dDDI)
    device_x = AMGX.AMGXVector(resources, AMGX.dDDI)
    csr_matrix = CuSparseMatrixCSR(transpose(matrix))
    
    @inline subtract_one(x) = x - oneunit(x)
    
    AMGX.upload!(device_matrix, 
                 map(subtract_one, csr_matrix.rowPtr), # annoyingly arrays need to be 0-indexed rather than 1-indexed
                 map(subtract_one, csr_matrix.colVal),
                 csr_matrix.nzVal
                 )
    
    AMGX.setup!(solver, device_matrix)

    return AMGXMultigridSolver(config,
                               resources,
                               solver,
                               device_matrix,
                               device_x,
                               device_b,
                               csr_matrix
                               )
end

@inline create_multilevel(::RugeStubenAMG, A) = ruge_stuben(A)
@inline create_multilevel(::SmoothedAggregationAMG, A) = smoothed_aggregation(A)

"""
    solve!(x, solver::MultigridSolver, b; kwargs...)

Solve `A * x = b` using a multigrid method, where `A` is `solver.matrix`.
"""
function solve!(x, solver::MultigridCPUSolver, b; kwargs...)
    Nx, Ny, Nz = size(b)

    solver.b_array .= reshape(interior(b), Nx * Ny * Nz)
    solver.x_array .= reshape(interior(x), Nx * Ny * Nz)

    _solve!(solver.x_array, solver.ml, solver.b_array;
            maxiter = solver.maxiter,
            abstol = solver.abstol,
            reltol = solver.reltol,
            kwargs...)

    interior(x) .= reshape(solver.x_array, Nx, Ny, Nz)

    return nothing
end

function solve!(x, solver::MultigridGPUSolver, b; kwargs...)
    Nx, Ny, Nz = size(b)

    solver.b_array .= reshape(interior(b), Nx * Ny * Nz)
    solver.x_array .= reshape(interior(x), Nx * Ny * Nz)

    s = solver.amgx_solver
    AMGX.upload!(s.device_b, solver.b_array)
    AMGX.upload!(s.device_x, solver.x_array)
    AMGX.solve!(s.device_x, s.solver, s.device_b)
    AMGX.copy!(solver.x_array, s.device_x)

    interior(x) .= reshape(solver.x_array, Nx, Ny, Nz)

    return nothing
end

"""
    initialize_AMGX(architecture)
    
Initialize the AMGX package required to use the multigrid solver on `architecture`. 
This function needs to be called before creating a multigrid solver on GPU.
"""
function initialize_AMGX(::GPU)
    try
        AMGX.initialize(); AMGX.initialize_plugins()
    catch e
        @info "It appears AMGX was not finalized. Have you called `finalize_AMGX`?"
        AMGX.finalize_plugins()
        AMGX.finalize()
        AMGX.initialize()
        AMGX.initialize_plugins()
    end
end

initialize_AMGX(::CPU) = nothing

"""
    finalize_AMGX(architecture)

Finalize the AMGX package required to use the multigrid solver on `architecture`. 
This should be called after `finalize_solver!`.
"""
function finalize_AMGX(::GPU)
    AMGX.finalize_plugins(); AMGX.finalize()
end

finalize_AMGX(::CPU) = nothing


function finalize_solver!(s::AMGXMultigridSolver)
    @info "Finalizing the AMGX Multigrid solver on GPU"
    close(s.device_matrix)
    close(s.device_x)
    close(s.device_b)
    close(s.solver)
    close(s.resources)
    close(s.config)

    return nothing
end

finalize_solver!(solver::MultigridGPUSolver) = finalize_solver!(solver.amgx_solver)

finalize_solver!(::MultigridCPUSolver) = nothing

Base.show(io::IO, solver::MultigridSolver) = 
print(io, "MultigridSolver on ", string(typeof(architecture(solver))), ": \n",
              "├── grid: ", summary(solver.grid), "\n",
              "├── matrix: ", prettysummary(solver.matrix), "\n",
              "├── reltol: ", prettysummary(solver.reltol), "\n",
              "├── abstol: ", prettysummary(solver.abstol), "\n",
              "└── maxiter: ", solver.maxiter, "\n",)
