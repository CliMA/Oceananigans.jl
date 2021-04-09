using Oceananigans.Solvers
using Oceananigans.Operators

struct ImplicitFreeSurfaceSolver{S}
    solver   :: S
end

"""
    implicit_free_surface_linear_operator!(result, x, arch, grid, bcs; args...)

Returns `L(ηⁿ)`, where `ηⁿ` is the free surface displacement at time step `n`
and `L` is the linear operator that arises
in an implicit time step for the free surface displacement `η`.

(See the docs section on implicit time stepping.)
"""
function implicit_free_surface_linear_operator!(L_ηⁿ, ηⁿ, arch, Δt, g)
    grid = L_ηⁿ.grid

    event = launch!(arch, grid, :xy, implicit_η!, ∇²_baro, Δt, g, grid, ηⁿ, L_ηⁿ, dependencies=Event(device(arch)))
    wait(device(arch), event)

    fill_halo_regions!(result, arch)

    return nothing
end

function ImplicitFreeSurfaceSolver(arch, template_field, 
                                   vertically_integrated_lateral_face_areas;
                                   Amatrix_operator=nothing, 
                                   maxit=nothing, 
                                   tol=nothing)
       
    ∇²_baro = ∇²_baro_operator(vertically_integrated_lateral_face_areas.Ax, vertically_integrated_lateral_face_areas.Ay)

    if isnothing(Amatrix_operator)
        function Amatrix_operator!(L_ηⁿ, ηⁿ, arch, Δt, g)
            grid = L_ηⁿ.grid

            event = launch!(arch, grid, :xy, implicit_η!, L_ηⁿ, ∇²_baro, Δt, g, grid, ηⁿ, dependencies=Event(device(arch)))
            wait(device(arch), event)

            fill_halo_regions!(result, arch)

            return nothing
        end
    end

    if isnothing(maxit)
        maxit = template_field.grid.Nx * template_field.grid.Ny
    end

    if isnothing(tol)
        tol = 1.e-13
    end

    pcg_params = (
        PCmatrix_function = nothing,
        Amatrix_function = Amatrix_function!,
        Template_field = template_field,
        maxit = maxit,
        tol = tol,
    )

    S = PreconditionedConjugateGradientSolver(arch = arch, parameters = pcg_params)

    return ImplicitFreeSurfaceSolver(S)
end
