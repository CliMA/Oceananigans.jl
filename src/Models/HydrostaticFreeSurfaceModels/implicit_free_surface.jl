using Oceananigans.Grids: AbstractGrid
using Oceananigans.Architectures: device
using Oceananigans.Operators: ∂xᶠᶜᶜ, ∂yᶜᶠᶜ, Δzᶜᶜᶠ, Δzᶜᶜᶜ
using Oceananigans.BoundaryConditions: regularize_field_boundary_conditions
using Oceananigans.Solvers: solve!
using Oceananigans.Utils: prettysummary
using Oceananigans.Fields
using Oceananigans.Utils: prettytime

using Adapt

struct ImplicitFreeSurface{E, G, B, I, M, S} <: AbstractFreeSurface{E, G}
    η :: E
    gravitational_acceleration :: G
    barotropic_volume_flux :: B
    implicit_step_solver :: I
    solver_method :: M
    solver_settings :: S
end

Base.show(io::IO, fs::ImplicitFreeSurface) =
    isnothing(fs.η) ?
    print(io, "ImplicitFreeSurface with ", fs.solver_method, "\n",
              "├─ gravitational_acceleration: ", prettysummary(fs.gravitational_acceleration), "\n",
              "├─ solver_method: ", fs.solver_method, "\n", # TODO: implement summary for solvers
              "└─ settings: ", isempty(fs.solver_settings) ? "Default" : fs.solver_settings) :
    print(io, "ImplicitFreeSurface with ", fs.solver_method, "\n",
              "├─ grid: ", summary(fs.η.grid), "\n",
              "├─ η: ", summary(fs.η), "\n",
              "├─ gravitational_acceleration: ", prettysummary(fs.gravitational_acceleration), "\n",
              "├─ implicit_step_solver: ", nameof(typeof(fs.implicit_step_solver)), "\n", # TODO: implement summary for solvers
              "└─ settings: ", fs.solver_settings)

"""
    ImplicitFreeSurface(; solver_method=:Default, gravitational_acceleration=g_Earth, solver_settings...)

Return an implicit free-surface solver. The implicit free-surface equation is

```math
\\left [ 𝛁_h ⋅ (H 𝛁_h) - \\frac{1}{g Δt^2} \\right ] η^{n+1} = \\frac{𝛁_h ⋅ 𝐐_⋆}{g Δt} - \\frac{η^{n}}{g Δt^2} ,
```

where ``η^n`` is the free-surface elevation at the ``n``-th time step, ``H`` is depth, ``g`` is
the gravitational acceleration, ``Δt`` is the time step, ``𝛁_h`` is the horizontal gradient operator,
and ``𝐐_⋆`` is the barotropic volume flux associated with the predictor velocity field ``𝐮_⋆``, i.e., 

```math
𝐐_⋆ = \\int_{-H}^0 𝐮_⋆ \\, 𝖽 z ,
```

where 

```math
𝐮_⋆ = 𝐮^n + \\int_{t_n}^{t_{n+1}} 𝐆ᵤ \\, 𝖽t .
```

This equation can be solved, in general, using the [`PreconditionedConjugateGradientSolver`](@ref) but 
other solvers can be invoked in special cases.

If ``H`` is constant, we divide through out to obtain

```math
\\left ( ∇^2_h - \\frac{1}{g H Δt^2} \\right ) η^{n+1}  = \\frac{1}{g H Δt} \\left ( 𝛁_h ⋅ 𝐐_⋆ - \\frac{η^{n}}{Δt} \\right ) .
```

Thus, for constant ``H`` and on grids with regular spacing in ``x`` and ``y`` directions, the free
surface can be obtained using the [`FFTBasedPoissonSolver`](@ref).

`solver_method` can be either of:
* `:FastFourierTransform` for [`FFTBasedPoissonSolver`](@ref)
* `:HeptadiagonalIterativeSolver`  for [`HeptadiagonalIterativeSolver`](@ref)
* `:PreconditionedConjugateGradient` for [`PreconditionedConjugateGradientSolver`](@ref)

By default, if the grid has regular spacing in the horizontal directions then the `:FastFourierTransform` is chosen,
otherwise the `:HeptadiagonalIterativeSolver`.
"""
ImplicitFreeSurface(; solver_method=:Default, gravitational_acceleration=g_Earth, solver_settings...) =
    ImplicitFreeSurface(nothing, gravitational_acceleration, nothing, nothing, solver_method, solver_settings)

Adapt.adapt_structure(to, free_surface::ImplicitFreeSurface) =
    ImplicitFreeSurface(Adapt.adapt(to, free_surface.η), free_surface.gravitational_acceleration,
                        nothing, nothing, nothing, nothing)

# Internal function for HydrostaticFreeSurfaceModel
function FreeSurface(free_surface::ImplicitFreeSurface{Nothing}, velocities, grid)
    η = FreeSurfaceDisplacementField(velocities, free_surface, grid)
    gravitational_acceleration = convert(eltype(grid), free_surface.gravitational_acceleration)

    # Initialize barotropic volume fluxes
    barotropic_x_volume_flux = Field((Face, Center, Nothing), grid)
    barotropic_y_volume_flux = Field((Center, Face, Nothing), grid)
    barotropic_volume_flux = (u=barotropic_x_volume_flux, v=barotropic_y_volume_flux)

    user_solver_method = free_surface.solver_method # could be = :Default
    solver = build_implicit_step_solver(Val(user_solver_method), grid, free_surface.solver_settings, gravitational_acceleration)
    solver_method = nameof(typeof(solver))

    return ImplicitFreeSurface(η,
                               gravitational_acceleration,
                               barotropic_volume_flux,
                               solver,
                               solver_method,
                               free_surface.solver_settings)
end

is_horizontally_regular(grid) = false
is_horizontally_regular(::RectilinearGrid{<:Any, <:Any, <:Any, <:Any, <:Number, <:Number}) = true

function build_implicit_step_solver(::Val{:Default}, grid, settings, gravitational_acceleration)
    default_method = is_horizontally_regular(grid) ? :FastFourierTransform : :HeptadiagonalIterativeSolver
    return build_implicit_step_solver(Val(default_method), grid, settings, gravitational_acceleration)
end

@inline explicit_barotropic_pressure_x_gradient(i, j, k, grid, ::ImplicitFreeSurface) = 0
@inline explicit_barotropic_pressure_y_gradient(i, j, k, grid, ::ImplicitFreeSurface) = 0

"""
Implicitly step forward η.
"""
ab2_step_free_surface!(free_surface::ImplicitFreeSurface, model, Δt, χ) =
    implicit_free_surface_step!(free_surface::ImplicitFreeSurface, model, Δt, χ)

function implicit_free_surface_step!(free_surface::ImplicitFreeSurface, model, Δt, χ)
    η      = free_surface.η
    g      = free_surface.gravitational_acceleration
    rhs    = free_surface.implicit_step_solver.right_hand_side
    ∫ᶻQ    = free_surface.barotropic_volume_flux
    solver = free_surface.implicit_step_solver
    arch   = model.architecture

    fill_halo_regions!(model.velocities)

    # Compute right hand side of implicit free surface equation
    @apply_regionally local_compute_integrated_volume_flux!(∫ᶻQ, model.velocities, arch)
    
    u = ∫ᶻQ.u
    v = ∫ᶻQ.v

    for _ in 1:3
        fill_halo_regions!(∫ᶻQ)
        @apply_regionally replace_horizontal_vector_halos!((; u, v, w=nothing), model.grid)
    end
    
    Nx, Ny, Nz = size(model.grid)
    Hx, Hy, Hz = halo_size(model.grid)
    
    operation_corner_points = "average" # Choose operation_corner_points to be "average", "CCW", "CW".
    
    for region in [1, 3, 5]

        region_south = mod(region + 4, 6) + 1
        region_east = region + 1
        region_north = mod(region + 2, 6)
        region_west = mod(region + 4, 6)

        # Northwest corner region
        for k in -Hz+1:Nz+Hz
            # Local y direction
            # (a) Proceed from [1, Ny+1] to [1, Ny+Hy].
            # (b) Shift left by one index in the first dimension to proceed from [0, Ny+1] to [0, Ny+Hy].
            u[region][0, Ny+1:Ny+Hy, k] .= reverse(-u[region_west][2, Ny-Hy+1:Ny, k]')
            v[region][0, Ny+1, k] = -u[region][1, Ny, k]
            v[region][0, Ny+2:Ny+Hy, k] .= reverse(-v[region_west][1, Ny-Hy+2:Ny, k]')
            # Local x direction
            # (a) Proceed from [1-Hx, Ny] to [0, Ny].
            # (b) Shift up by one index in the second dimension to proceed from [1-Hx, Ny+1] to [0, Ny+1].
            u[region][1-Hx:0, Ny+1, k] .= reverse(-u[region_north][2:Hx+1, Ny, k])
            v[region][1-Hx:0, Ny+1, k] .= -u[region_west][1, Ny-Hx+1:Ny, k]
            # Corner point operation
            u_CCW = -u[region_west][2, Ny, k]
            u_CW = -u[region_north][2, Ny, k]
            u[region][0, Ny+1, k] = operation_corner_points == "average" ? 0.5 * (u_CCW + u_CW) :
                                    operation_corner_points == "CCW" ? u_CCW :
                                    operation_corner_points == "CW" ? u_CW : nothing
            v_CCW = -u[region][1, Ny, k] 
            v_CW = -u[region_west][1, Ny, k]
            v[region][0, Ny+1, k] = operation_corner_points == "average" ? 0.5 * (v_CCW + v_CW) :
                                    operation_corner_points == "CCW" ? v_CCW :
                                    operation_corner_points == "CW" ? v_CW : nothing
        end

        # Northeast corner region
        for k in -Hz+1:Nz+Hz
            # Local y direction
            # (a) Proceed from [Nx, Ny+1] to [Nx, Ny+Hy].
            # (b) Shift right by one index in the first dimension to proceed from [Nx+1, Ny+1] to [Nx+1, Ny+Hy].
            u[region][Nx+1, Ny+1:Ny+Hy, k] .= -v[region_north][1:Hy, 1, k]'
            v[region][Nx+1, Ny+1:Ny+Hy, k] .= u[region_east][1:Hy, Ny, k]'
            # Local x direction
            # (a) Proceed from [Nx+1, Ny] to [Nx+Hx, Ny].
            # (b) Shift up by one index in the second dimension to proceed from [Nx+1, Ny+1] to [Nx+Hx, Ny+1].
            u[region][Nx+1:Nx+Hx, Ny+1, k] .= u[region_north][1:Hx, 1, k]
            v[region][Nx+1:Nx+Hx, Ny+1, k] .= v[region_north][1:Hx, 1, k]
            # Corner point operation
            u_CCW = u[region_north][1, 1, k]
            u_CW = -v[region_north][1, 1, k]
            u[region][Nx+1, Ny+1, k] = operation_corner_points == "average" ? 0.5 * (u_CCW + u_CW) :
                                       operation_corner_points == "CCW" ? u_CCW :
                                       operation_corner_points == "CW" ? u_CW : nothing
            v_CCW = v[region_north][1, 1, k]
            v_CW = u[region_east][1, Ny, k]
            v[region][Nx+1, Ny+1, k] = operation_corner_points == "average" ? 0.5 * (v_CCW + v_CW) :
                                       operation_corner_points == "CCW" ? v_CCW :
                                       operation_corner_points == "CW" ? v_CW : nothing
        end

        # Southwest corner region
        for k in -Hz+1:Nz+Hz
            # Local y direction
            # (a) Proceed from [1, 1-Hy] to [1, 0].
            # (b) Shift left by one index in the first dimension to proceed from [0, 1-Hy] to [0, 0].
            u[region][0, 1-Hy:0, k] .= u[region_west][Nx, Ny-Hy+1:Ny, k]'
            v[region][0, 1-Hy:0, k] .= v[region_west][Nx, Ny-Hy+1:Ny, k]'
            # Local x direction
            # (a) Proceed from [1-Hx, 1] to [0, 1].
            # (b) Shift down by one index in the second dimension to proceed from [1-Hx, 0] to [0, 0].
            u[region][1-Hx:0, 0, k] .= v[region_south][1, Ny-Hx+1:Ny, k]
            v[region][1-Hx:0, 0, k] .= -u[region_south][2, Ny-Hx+1:Ny, k]
            # Corner point operation
            u_CCW = v[region_south][1, Ny, k]
            u_CW = u[region_west][Nx, Ny, k]
            u[region][0, 0, k] = operation_corner_points == "average" ? 0.5 * (u_CCW + u_CW) :
                                 operation_corner_points == "CCW" ? u_CCW :
                                 operation_corner_points == "CW" ? u_CW : nothing
            v_CCW = -u[region_south][2, Ny, k]
            v_CW = v[region_west][Nx, Ny, k]
            v[region][0, 0, k] = operation_corner_points == "average" ? 0.5 * (v_CCW + v_CW) :
                                 operation_corner_points == "CCW" ? v_CCW :
                                 operation_corner_points == "CW" ? v_CW : nothing
        end

        # Southeast corner region
        for k in -Hz+1:Nz+Hz
            # Local y direction
            # (a) Proceed from [Nx, 1-Hy] to [Nx, 0].
            # (b) Shift right by one index in the first dimension to proceed from [Nx+1, 1-Hy] to [Nx+1, 0].
            u[region][Nx+1, 1-Hy:0, k] .= reverse(v[region_east][1:Hy, 1, k]')
            v[region][Nx+1, 1-Hy:0, k] .= reverse(-u[region_east][2:Hy+1, 1, k]')
            # Local x direction
            # (a) Proceed from [Nx+1, 1] to [Nx+Hx, 1].
            # (b) Shift down by one index in the second dimension to proceed from [Nx+1, 0] to [Nx+Hx, 0].
            u[region][Nx+1, 0, k] = -v[region][Nx, 1, k]
            u[region][Nx+2:Nx+Hx, 0, k] .= reverse(-v[region_south][Nx, Ny-Hx+2:Ny, k])
            v[region][Nx+1:Nx+Hx, 0, k] .= u[region_south][Nx, Ny-Hx+1:Ny, k]
            # Corner point operation
            u_CCW = v[region_east][1, 1, k]
            u_CW = -v[region][Nx, 1, k]
            u[region][Nx+1, 0, k] = operation_corner_points == "average" ? 0.5 * (u_CCW + u_CW) :
                                    operation_corner_points == "CCW" ? u_CCW :
                                    operation_corner_points == "CW" ? u_CW : nothing
            v_CCW = -u[region_east][2, 1, k]
            v_CW = u[region_south][Nx, Ny, k]
            v[region][Nx+1, 0, k] = operation_corner_points == "average" ? 0.5 * (v_CCW + v_CW) :
                                    operation_corner_points == "CCW" ? v_CCW :
                                    operation_corner_points == "CW" ? v_CW : nothing
        end
    end
    
    for region in [2, 4, 6]
        region_south = mod(region + 3, 6) + 1
        region_east = mod(region, 6) + 2
        region_north = mod(region, 6) + 1
        region_west = region - 1

        # Northwest corner region
        for k in -Hz+1:Nz+Hz
            # Local y direction
            # (a) Proceed from [1, Ny+1] to [1, Ny+Hy].
            # (b) Shift left by one index in the first dimension to proceed from [0, Ny+1] to [0, Ny+Hy].
            u[region][0, Ny+1:Ny+Hy, k] .= reverse(v[region_west][Nx-Hy+1:Nx, Ny, k]')
            v[region][0, Ny+1, k] = -u[region][1, Ny, k]
            v[region][0, Ny+2:Ny+Hy, k] .= reverse(-u[region_west][Nx-Hy+2:Nx, Ny, k]')
            # Local x direction
            # (a) Proceed from [1-Hx, Ny] to [0, Ny].
            # (b) Shift up by one index in the second dimension to proceed from [1-Hx, Ny+1] to [0, Ny+1].
            u[region][1-Hx:0, Ny+1, k] .= reverse(-v[region_north][1, 2:Hx+1, k])
            v[region][1-Hx:0, Ny+1, k] .= reverse(u[region_north][1, 1:Hx, k])
            # Corner point operation
            u_CCW = v[region_west][Nx, Ny, k]
            u_CW = -v[region_north][1, 2, k]
            u[region][0, Ny+1, k] = operation_corner_points == "average" ? 0.5 * (u_CCW + u_CW) :
                                    operation_corner_points == "CCW" ? u_CCW :
                                    operation_corner_points == "CW" ? u_CW : nothing
            v_CCW = -u[region][1, Ny, k]
            v_CW = u[region_north][1, 1, k]
            v[region][0, Ny+1, k] = operation_corner_points == "average" ? 0.5 * (v_CCW + v_CW) :
                                    operation_corner_points == "CCW" ? v_CCW :
                                    operation_corner_points == "CW" ? v_CW : nothing    
        end

        # Northeast corner region
        for k in -Hz+1:Nz+Hz
            # Local y direction
            # (a) Proceed from [Nx, Ny+1] to [Nx, Ny+Hy].
            # (b) Shift right by one index in the first dimension to proceed from [Nx+1, Ny+1] to [Nx+1, Ny+Hy].
            u[region][Nx+1, Ny+1:Ny+Hy, k] .= u[region_east][1, 1:Hy, k]'
            v[region][Nx+1, Ny+1:Ny+Hy, k] .= v[region_east][1, 1:Hy, k]'
            # Local x direction
            # (a) Proceed from [Nx+1, Ny] to [Nx+Hx, Ny].
            # (b) Shift up by one index in the second dimension to proceed from [Nx+1, Ny+1] to [Nx+Hx, Ny+1].
            u[region][Nx+1:Nx+Hx, Ny+1, k] .= v[region_north][Nx, 1:Hx, k]
            v[region][Nx+1:Nx+Hx, Ny+1, k] .= -u[region_east][1, 1:Hx, k]
            # Corner point operation
            u_CCW = v[region_north][Nx, 1, k]
            u_CW = u[region_east][1, 1, k]
            u[region][Nx+1, Ny+1, k] = operation_corner_points == "average" ? 0.5 * (u_CCW + u_CW) :
                                       operation_corner_points == "CCW" ? u_CCW :
                                       operation_corner_points == "CW" ? u_CW : nothing
            v_CCW = -u[region_east][1, 1, k]
            v_CW = v[region_east][1, 1, k]
            v[region][Nx+1, Ny+1, k] = operation_corner_points == "average" ? 0.5 * (v_CCW + v_CW) :
                                       operation_corner_points == "CCW" ? v_CCW :
                                       operation_corner_points == "CW" ? v_CW : nothing
        end
        
        # Southwest corner region
        for k in -Hz+1:Nz+Hz
            # Local y direction
            # (a) Proceed from [1, 1-Hy] to [1, 0].
            # (b) Shift left by one index in the first dimension to proceed from [0, 1-Hy] to [0, 0].
            u[region][0, 1-Hy:0, k] .= -v[region_west][Nx-Hy+1:Nx, 2, k]'
            v[region][0, 1-Hy:0, k] .= u[region_west][Nx-Hy+1:Nx, 1, k]'
            # Local x direction
            # (a) Proceed from [1-Hx, 1] to [0, 1].
            # (b) Shift down by one index in the second dimension to proceed from [1-Hx, 0] to [0, 0].
            u[region][1-Hx:0, 0, k] .= u[region_south][Nx-Hx+1:Nx, Ny, k]
            v[region][1-Hx:0, 0, k] .= v[region_south][Nx-Hx+1:Nx, Ny, k]
            # Corner point operation
            u_CCW = u[region_south][Nx, Ny, k]
            u_CW = -v[region_west][Nx, 2, k]
            u[region][0, 0, k] = operation_corner_points == "average" ? 0.5 * (u_CCW + u_CW) :
                                 operation_corner_points == "CCW" ? u_CCW :
                                 operation_corner_points == "CW" ? u_CW : nothing
            v_CCW = v[region_south][Nx, Ny, k]
            v_CW = u[region_west][Nx, 1, k]
            v[region][0, 0, k] = operation_corner_points == "average" ? 0.5 * (v_CCW + v_CW) :
                                 operation_corner_points == "CCW" ? v_CCW :
                                 operation_corner_points == "CW" ? v_CW : nothing
        end
        
        # Southeast corner region
        for k in -Hz+1:Nz+Hz
            # Local y direction
            # (a) Proceed from [Nx, 1-Hy] to [Nx, 0].
            # (b) Shift right by one index in the first dimension to proceed from [Nx+1, 1-Hy] to [Nx+1, 0].
            u[region][Nx+1, 1-Hy:0, k] .= -v[region_south][Nx-Hy+1:Nx, 1, k]'
            v[region][Nx+1, 1-Hy:0, k] .= reverse(-v[region_east][Nx, 2:Hy+1, k]')
            # Local x direction
            # (a) Proceed from [Nx+1, 1] to [Nx+Hx, 1].
            # (b) Shift down by one index in the second dimension to proceed from [Nx+1, 0] to [Nx+Hx, 0].
            u[region][Nx+1, 0, k] = -v[region][Nx, 1, k]
            u[region][Nx+2:Nx+Hx, 0, k] .= reverse(-u[region_south][Nx-Hx+2:Nx, 1, k])
            v[region][Nx+1:Nx+Hx, 0, k] .= reverse(-v[region_south][Nx-Hx+1:Nx, 2, k])
            # Corner point operation
            u_CCW = -v[region_south][Nx, 1, k]
            u_CW = -v[region][Nx, 1, k]
            u[region][Nx+1, 0, k] = operation_corner_points == "average" ? 0.5 * (u_CCW + u_CW) :
                                    operation_corner_points == "CCW" ? u_CCW :
                                    operation_corner_points == "CW" ? u_CW : nothing
            v_CCW = -v[region_east][Nx, 2, k]
            v_CW = -v[region_south][Nx, 2, k]
            v[region][Nx+1, 0, k] = operation_corner_points == "average" ? 0.5 * (v_CCW + v_CW) :
                                    operation_corner_points == "CCW" ? v_CCW :
                                    operation_corner_points == "CW" ? v_CW : nothing
        end        
    end
    
    compute_implicit_free_surface_right_hand_side!(rhs, solver, g, Δt, ∫ᶻQ, η)

    # Solve for the free surface at tⁿ⁺¹
    start_time = time_ns()

    solve!(η, solver, rhs, g, Δt)

    @debug "Implicit step solve took $(prettytime((time_ns() - start_time) * 1e-9))."

    fill_halo_regions!(η)

    return nothing
end

function local_compute_integrated_volume_flux!(∫ᶻQ, velocities, arch)
    
    foreach(mask_immersed_field!, velocities)

    # Compute barotropic volume flux. Blocking.
    compute_vertically_integrated_volume_flux!(∫ᶻQ, velocities)

    return nothing
end
