using Oceananigans.Architectures: architecture
using Oceananigans.Grids: interior_parent_indices
using Statistics: norm, dot
using LinearAlgebra
using AlgebraicMultigrid: _solve!, init, RugeStubenAMG
using Oceananigans.Operators: volume, Δyᶠᶜᵃ, Δyᶜᶠᵃ, Δyᶜᶜᵃ, Δxᶠᶜᵃ, Δxᶜᶠᵃ, Δxᶜᶜᵃ, Δyᵃᶜᵃ, Δxᶜᵃᵃ, Δzᵃᵃᶠ, Δzᵃᵃᶜ, ∇²ᶜᶜᶜ
using CUDA
using AMGX

import Oceananigans.Architectures: architecture

mutable struct MultigridSolver{A, G, L, T, M, F, S}
               architecture :: A
                       grid :: G
                     matrix :: L
                     abstol :: T
                     reltol :: T
                    maxiter :: Int
              amg_algorithm :: M
                    x_array :: F
                    b_array :: F
         amgx_solver_struct :: S
end

mutable struct AMGXMultigridSolver{C, R, S, M, V}
                         config :: C
                      resources :: R
                    amgx_solver :: S
                  device_matrix :: M
                       device_b :: V
                       device_x :: V
end

architecture(solver::MultigridSolver) = solver.architecture
    
"""
    MultigridSolver(linear_operation!::Function,
                    args...;
                    template_field::AbstractField,
                    maxiter = prod(size(template_field)),
                    reltol = sqrt(eps(eltype(template_field.grid))),
                    abstol = 0reltol,
                    amg_algorithm = RugeStubenAMG(),
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

* `amg_algorithm`: Algebraic Multigrid algorithm defining mapping between different grid spacings

!!! compat "Multigrid solver on GPUs"
    Currently Multigrid solver is only supported on CPUs.
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

    # arch == GPU() && error("Multigrid solver is only supported on CPUs.")

    matrix = initialize_matrix(template_field, linear_operation!, args...)

    Nx, Ny, Nz = size(template_field)

    FT = eltype(template_field.grid)

    b_array = arch_array(arch, zeros(FT, Nx * Ny * Nz))
    x_array = arch_array(arch, zeros(FT, Nx * Ny * Nz))

    if arch == GPU()
        AMGX.initialize()
        AMGX.initialize_plugins()
        # FIXME! Also pass tolerance
        config = AMGX.Config(Dict("monitor_residual" => 1, "max_iters" => maxiter, "store_res_history" => 1));
        resources = AMGX.Resources(config)
        amgx_solver = AMGX.Solver(resources, AMGX.dDDI, config)
        device_matrix = AMGX.AMGXMatrix(resources, AMGX.dDDI)
        device_b = AMGX.AMGXVector(resources, AMGX.dDDI)
        device_x = AMGX.AMGXVector(resources, AMGX.dDDI)
        csr_matrix = CuSparseMatrixCSR(transpose(matrix))
        @inline sub_one(x) = convert(Int32, x-1)
        AMGX.upload!(device_matrix, 
            map(sub_one, csr_matrix.rowPtr), # annoyingly arrays need to be 0 indexed rather than 1 indexed
            map(sub_one, csr_matrix.colVal),
            csr_matrix.nzVal
            )
        AMGX.setup!(amgx_solver, device_matrix)
        amgx_solver_struct = AMGXMultigridSolver(config, 
                                                resources, 
                                                amgx_solver, 
                                                device_matrix, 
                                                device_b, 
                                                device_x
                                                ) 
    else 
        amgx_solver_struct = nothing
    end

    return MultigridSolver(arch,
                           template_field.grid,
                           matrix,
                           FT(abstol),
                           FT(reltol),
                           maxiter,
                           amg_algorithm,
                           x_array,
                           b_array,
                           amgx_solver_struct
                           )
end


function initialize_matrix(template_field, linear_operator!, args...)
    constructors = fill_matrix_elements2!(template_field, linear_operator!, args...)
    return arch_sparse_matrix(architecture(template_field), constructors)
end

function fill_matrix_elements!(A, template_field, linear_operator!, args...)
    Nx, Ny, Nz = size(template_field)
    make_column(f) = reshape(interior(f), Nx*Ny*Nz)

    eᵢⱼₖ = similar(template_field)
    ∇²eᵢⱼₖ = similar(template_field)
    
    for k = 1:Nz, j in 1:Ny, i in 1:Nx
        eᵢⱼₖ .= 0
        ∇²eᵢⱼₖ .= 0
        eᵢⱼₖ[i, j, k] = 1
        fill_halo_regions!(eᵢⱼₖ)
        linear_operator!(∇²eᵢⱼₖ, eᵢⱼₖ, args...)

        A[:, Ny*Nx*(k-1) + Nx*(j-1) + i] .= make_column(∇²eᵢⱼₖ)
    end

    return nothing
end

function fill_matrix_elements2!(template_field, linear_operator!, args...)
    Nx, Ny, Nz = size(template_field)
    make_column(f) = reshape(interior(f), Nx*Ny*Nz)

    eᵢⱼₖ = similar(template_field)
    ∇²eᵢⱼₖ = similar(template_field)

    arch = architecture(template_field)
    colptr = array_type(arch){Int64}(undef, Nx*Ny*Nz+1)
    rowval = array_type(arch){Int64}(undef, 0)
    nzval  = array_type(arch){Float64}(undef, 0)

    CUDA.@allowscalar colptr[1] = 1

    for k = 1:Nz, j in 1:Ny, i in 1:Nx
        eᵢⱼₖ .= 0
        ∇²eᵢⱼₖ .= 0
        CUDA.@allowscalar eᵢⱼₖ[i, j, k] = 1
        fill_halo_regions!(eᵢⱼₖ)
        linear_operator!(∇²eᵢⱼₖ, eᵢⱼₖ, args...)
        count = 0
        for n = 1:Nz, m in 1:Ny, l in 1:Nx
            # CUDA.@allowscalar value = ∇²eᵢⱼₖ[l, m, n]
            CUDA.@allowscalar if ∇²eᵢⱼₖ[l, m, n] != 0
                append!(rowval, Ny*Nx*(n-1) + Nx*(m-1) + l)
                CUDA.@allowscalar append!(nzval, ∇²eᵢⱼₖ[l, m, n])
                count += 1
            end
        end
        CUDA.@allowscalar colptr[Ny*Nx*(k-1) + Nx*(j-1) + i + 1] = colptr[Ny*Nx*(k-1) + Nx*(j-1) + i] + count
    end
    if arch == CPU()
        return Nx*Ny*Nz, Nx*Ny*Nz, colptr, rowval, nzval
    end
    return colptr, rowval, nzval, (Nx*Ny*Nz, Nx*Ny*Nz)
end

"""
    solve!(x, solver::MultigridSolver, b; kwargs...)

Solve `A * x = b` using a multigrid method, where `A * x` is determined
by `solver.matrix`.
"""
function solve!(x, solver::MultigridSolver, b; kwargs...)
    Nx, Ny, Nz = size(b)

    solver.b_array .= reshape(interior(b), Nx * Ny * Nz)
    solver.x_array .= reshape(interior(x), Nx * Ny * Nz)

    if architecture(solver) == CPU()
        solt = init(solver.amg_algorithm, solver.matrix, solver.b_array)

        _solve!(solver.x_array, solt.ml, solt.b, maxiter=solver.maxiter, abstol = solver.abstol, reltol=solver.reltol, kwargs...)
    else 
        s = solver.amgx_solver_struct
        AMGX.upload!(s.device_b, solver.b_array)
        AMGX.upload!(s.device_x, solver.x_array)
        AMGX.setup!(s.amgx_solver, s.device_matrix)
        AMGX.solve!(s.device_x, s.amgx_solver, s.device_b)
        solver.x_array = CuArray(Vector(s.device_x)) #FIXME
    end

    interior(x) .= reshape(solver.x_array, Nx, Ny, Nz)
end

function finalize_solver(solver::MultigridSolver)
    s = solver.amgx_solver_struct
    if s != nothing
        close(s.device_matrix); close(s.device_x); close(s.device_b); close(s.amgx_solver); close(s.resources); close(s.config); 
        AMGX.finalize_plugins(); AMGX.finalize()
    end
end


Base.show(io::IO, solver::MultigridSolver) = 
print(io, "MultigridSolver on ", string(typeof(architecture(solver))), ": \n",
              "├── grid: ", summary(solver.grid), '\n',
              "├── matrix: ", prettysummary(solver.matrix), '\n',
              "├── reltol: ", prettysummary(solver.reltol), '\n',
              "├── abstol: ", prettysummary(solver.abstol), '\n',
              "├── maxiter: ", solver.maxiter, '\n',
              "└── amg_algorithm: ", typeof(solver.amg_algorithm))
