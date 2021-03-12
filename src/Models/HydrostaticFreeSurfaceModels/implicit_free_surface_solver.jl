using Oceananigans.Solvers
using Oceananigans.Operators

struct ImplicitFreeSurfaceSolver{S}
    solver   :: S
end

function ImplicitFreeSurfaceSolver(arch, template_field, 
                                   vertically_integrated_lateral_face_areas;
                                   Amatrix_operator=nothing, 
                                   maxit=nothing, 
                                   tol=nothing)
       
    ∇²_baro=∇²_baro_operator( vertically_integrated_lateral_face_areas.Ax, vertically_integrated_lateral_face_areas.Ay)

    if isnothing( Amatrix_operator )
        function Amatrix_function!(result, x, arch, grid, bcs; args...)
            Δt=args.data.Δt
             g=args.data.g
            event = launch!(arch, grid, :xy, implicit_η!, ∇²_baro, Δt, g, grid, x, result, dependencies=Event(device(arch)))
            wait(device(arch), event)
            fill_halo_regions!(result, bcs, arch, grid)
            return nothing
        end
    end

    if isnothing( maxit )
        maxit = template_field.grid.Nx * template_field.grid.Ny
    end

    if isnothing( tol )
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
