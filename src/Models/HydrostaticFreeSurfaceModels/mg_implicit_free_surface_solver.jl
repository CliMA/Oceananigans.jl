using Oceananigans.Solvers
using Oceananigans.Operators
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBottom
using Oceananigans.Architectures
using Oceananigans.Grids: with_halo, isrectilinear
using Oceananigans.Fields: Field, ZReducedField
using Oceananigans.Architectures: device, unsafe_free!
using Oceananigans.Models.HydrostaticFreeSurfaceModels: Az_∇h²ᶜᶜᶜ
using Oceananigans.Solvers: constructors, arch_sparse_matrix, update_diag!, unpack_constructors, matrix_from_coefficients
using Oceananigans.Utils: prettysummary
using SparseArrays: _insert!
using CUDA.CUSPARSE: CuSparseMatrixCSR
using AMGX

import Oceananigans.Solvers: solve!, precondition!, finalize_solver!
import Oceananigans.Architectures: architecture

"""
    mutable struct MGImplicitFreeSurfaceSolver{S, V, F, R, C, D}

The multigrid implicit free-surface solver.

$(TYPEDFIELDS)
"""
mutable struct MGImplicitFreeSurfaceSolver{A, S, V, F, R, C, D}
    "Architecture"
    architecture :: A
    "The multigrid solver"
    multigrid_solver :: S
    "The vertically-integrated lateral areas"
    vertically_integrated_lateral_areas :: V
    "The previous time step"
    previous_Δt :: F
    "The right hand side of the free surface evolution equation"
    right_hand_side :: R
    "The matrix constructors of the linear operator without the `Az / (g Δt²)` term"
    matrix_constructors :: C
    "The `Az / g` term"
    diagonal :: D
end

architecture(solver::MGImplicitFreeSurfaceSolver) =
    architecture(solver.multigrid_solver)

"""
    MGImplicitFreeSurfaceSolver(grid::AbstractGrid, 
                                settings = nothing, 
                                gravitational_acceleration = nothing, 
                                placeholder_timestep = -1.0)

Return a solver based on a multigrid method for the elliptic equation
    
```math
[∇ ⋅ H ∇ - 1 / (g Δt²)] ηⁿ⁺¹ = (∇ʰ ⋅ Q★ - ηⁿ / Δt) / (g Δt)
```

representing an implicit time discretization of the linear free surface evolution equation
for a fluid with variable depth `H`, horizontal areas `Az`, barotropic volume flux `Q★`, time
step `Δt`, gravitational acceleration `g`, and free surface at the `n`-th time-step `ηⁿ`.
"""
function MGImplicitFreeSurfaceSolver(grid::AbstractGrid, 
                                     settings = nothing,
                                     gravitational_acceleration = g_Earth,
                                     reduced_dim = (false, false, false),
                                     placeholder_timestep = -1.0)
    arch = architecture(grid)

    right_hand_side = Field{Center, Center, Nothing}(grid)

    # Initialize vertically integrated lateral face areas
    ∫ᶻ_Axᶠᶜᶜ = Field{Face, Center, Nothing}(with_halo((3, 3, 1), grid))
    ∫ᶻ_Ayᶜᶠᶜ = Field{Center, Face, Nothing}(with_halo((3, 3, 1), grid))

    vertically_integrated_lateral_areas = (xᶠᶜᶜ = ∫ᶻ_Axᶠᶜᶜ, yᶜᶠᶜ = ∫ᶻ_Ayᶜᶠᶜ)

    compute_vertically_integrated_lateral_areas!(vertically_integrated_lateral_areas)
    coeffs = compute_matrix_coefficients(vertically_integrated_lateral_areas, grid, gravitational_acceleration)
    matrix_constructors, diagonal, problem_size = matrix_from_coefficients(arch, right_hand_side, coeffs, reduced_dim)  

    # Placeholder preconditioner and matrix are calculated using a "placeholder" timestep of -1.0
    # They are then recalculated before the first time step of the simulation.

    placeholder_constructors = deepcopy(matrix_constructors)
    M = prod(problem_size)
    update_diag!(placeholder_constructors, arch, M, M, diagonal, 1.0, 0)

    matrix = arch_sparse_matrix(arch, placeholder_constructors)
    fill_halo_regions!(vertically_integrated_lateral_areas)

    # set some defaults
    if settings !== nothing
        settings = Dict{Symbol, Any}(settings)
    else
        settings = Dict{Symbol, Any}()
    end
    settings[:maxiter] = get(settings, :maxiter, grid.Nx * grid.Ny)
    settings[:reltol] = get(settings, :reltol, min(1e-7, 10 * sqrt(eps(eltype(grid)))))

    # initialize solver with Δt = nothing so that linear matrix is not computed;
    # see `initialize_matrix` methods
    solver = MultigridSolver(Az_∇h²ᶜᶜᶜ_linear_operation!, ∫ᶻ_Axᶠᶜᶜ, ∫ᶻ_Ayᶜᶠᶜ;
                             template_field = right_hand_side,
                             matrix,
                             settings...)

    return MGImplicitFreeSurfaceSolver(arch, solver, vertically_integrated_lateral_areas, placeholder_timestep, right_hand_side, matrix_constructors, diagonal)
end

finalize_solver!(solver::MGImplicitFreeSurfaceSolver) = finalize_solver!(solver.multigrid_solver)

"""
Returns `L(ηⁿ)`, where `ηⁿ` is the free surface displacement at time step `n`
and `L` is the linear operator that arises
in an implicit time step for the free surface displacement `η` 
without the final term that is dependent on Δt

(See the docs section on implicit time stepping.)
"""
function Az_∇h²ᶜᶜᶜ_linear_operation!(L_ηⁿ⁺¹, ηⁿ⁺¹, ∫ᶻ_Axᶠᶜᶜ, ∫ᶻ_Ayᶜᶠᶜ)
    grid = L_ηⁿ⁺¹.grid
    arch = architecture(L_ηⁿ⁺¹)
    fill_halo_regions!(ηⁿ⁺¹)

    event = launch!(arch, grid, :xy, _Az_∇h²ᶜᶜᶜ_linear_operation!,
                    L_ηⁿ⁺¹, grid,  ηⁿ⁺¹, ∫ᶻ_Axᶠᶜᶜ, ∫ᶻ_Ayᶜᶠᶜ,
                    dependencies = device_event(arch))

    wait(device(arch), event)

    return nothing
end

@kernel function _Az_∇h²ᶜᶜᶜ_linear_operation!(L_ηⁿ⁺¹, grid, ηⁿ⁺¹, ∫ᶻ_Axᶠᶜᶜ, ∫ᶻ_Ayᶜᶠᶜ)
    i, j = @index(Global, NTuple)
    @inbounds L_ηⁿ⁺¹[i, j, 1] = Az_∇h²ᶜᶜᶜ(i, j, 1, grid, ∫ᶻ_Axᶠᶜᶜ, ∫ᶻ_Ayᶜᶠᶜ, ηⁿ⁺¹)
end

build_implicit_step_solver(::Val{:Multigrid}, grid, settings, gravitational_acceleration) =
    MGImplicitFreeSurfaceSolver(grid, settings, gravitational_acceleration)

#####
##### Solve...
#####

function solve!(η, implicit_free_surface_solver::MGImplicitFreeSurfaceSolver{CPU}, rhs, g, Δt)
    solver = implicit_free_surface_solver.multigrid_solver

    # if `Δt` changed then re-compute the matrix elements
    if Δt != implicit_free_surface_solver.previous_Δt
        arch = architecture(solver.matrix)
        constructors = deepcopy(implicit_free_surface_solver.matrix_constructors)
        M = prod(size(η))
        update_diag!(constructors, arch, M, M, implicit_free_surface_solver.diagonal, Δt, 0)
        solver.matrix = arch_sparse_matrix(arch, constructors) 

        unsafe_free!(constructors)

        implicit_free_surface_solver.previous_Δt = Δt
    end
    solve!(η, solver, rhs)

    return nothing
end

function solve!(η, implicit_free_surface_solver::MGImplicitFreeSurfaceSolver{GPU}, rhs, g, Δt)
    solver = implicit_free_surface_solver.multigrid_solver

    # if `Δt` changed then re-compute the matrix elements
    if Δt != implicit_free_surface_solver.previous_Δt
        arch = architecture(solver.matrix)
        constructors = deepcopy(implicit_free_surface_solver.matrix_constructors)
        M = prod(size(η))
        update_diag!(constructors, arch, M, M, implicit_free_surface_solver.diagonal, Δt, 0)
        solver.matrix = arch_sparse_matrix(arch, constructors) 

        unsafe_free!(constructors)

        s = solver.amgx_solver
        solver.amgx_solver.csr_matrix = CuSparseMatrixCSR(transpose(solver.matrix))
        @inline subtract_one(x) = convert(Int32, x-1)
        AMGX.upload!(s.device_matrix, 
                        map(subtract_one, solver.amgx_solver.csr_matrix.rowPtr),
                        map(subtract_one, solver.amgx_solver.csr_matrix.colVal),
                        solver.amgx_solver.csr_matrix.nzVal
                        )
        AMGX.setup!(s.solver, s.device_matrix)

        implicit_free_surface_solver.previous_Δt = Δt
    end
    solve!(η, solver, rhs)

    return nothing
end

function compute_implicit_free_surface_right_hand_side!(rhs, implicit_solver::MGImplicitFreeSurfaceSolver,
                                                        g, Δt, ∫ᶻQ, η)
    solver = implicit_solver.multigrid_solver
    arch = architecture(solver)
    grid = solver.grid

    event = launch!(arch, grid, :xy,
                    implicit_free_surface_right_hand_side!,
                    rhs, grid, g, Δt, ∫ᶻQ, η,
                    dependencies = device_event(arch))
    
    wait(device(arch), event)
    return nothing
end
