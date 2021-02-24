using Oceananigans.Solvers

struct ImplicitFreeSurfaceSolver{S}
    solver :: S
end

function ImplicitFreeSurfaceSolver(arch, template_field; Amatrix_operator=nothing)

    @kernel function implicit_η!(grid, f, implicit_η_f, g, Δt)
        ### Not sure what to call this
        ### it is for left hand side operator in
        ### (-g∇ₕ² + 1/Δt )ϕⁿ⁺¹=ϕⁿ/Δt + ∇ₕHUˢᵗᵃʳ

        i, j = @index(Global, NTuple)
        @inbounds implicit_η_f[i, j] = -g * ∇²(i, j, 1, grid, f) + f[i,j]/Δt

    end


    if Amatrix_operator == nothing
        function Amatrix_function!(result, x, arch, grid, bcs)
            event = launch!(arch, grid, :xyz, implicit_η!, grid, x, result, dependencies=Event(device(arch)))
            wait(device(arch), event)
            fill_halo_regions!(result, bcs, arch, grid)
            return nothing
        end
    end

     pcg_params = (
        PCmatrix_function = nothing,
        Amatrix_function = Amatrix_function!,
        Template_field = template_field,
        maxit = 1000, # grid.Nx * grid.Ny,
        tol = 1.e-13
    )


    S = PreconditionedConjugateGradientSolver(arch = arch, parameters = pcg_params)
    return ImplicitFreeSurfaceSolver(S)
end
