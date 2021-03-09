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
       
    ## Some of this should probably end up in some operator and grid generic auxilliaries
    Ax_baro = vertically_integrated_lateral_face_areas.Ax
    Ay_baro = vertically_integrated_lateral_face_areas.Ay
    @inline Ax_∂xᶠᵃᵃ_baro(i, j, k, grid, c) = Ax_baro[i, j, 1] * ∂xᶠᵃᵃ(i, j, 1, grid, c)
    @inline Ay_∂yᵃᶠᵃ_baro(i, j, k, grid, c) = Ay_baro[i, j, 1] * ∂yᵃᶠᵃ(i, j, 1, grid, c)
    @inline function ∇²_baro(i, j, k, grid, c)
       return  δxᶜᵃᵃ(i, j, 1, grid, Ax_∂xᶠᵃᵃ_baro, c) +
               δyᵃᶜᵃ(i, j, 1, grid, Ay_∂yᵃᶠᵃ_baro, c)
    end

    @kernel function implicit_η!(Δt, g, grid, f, implicit_η_f)
        i, j = @index(Global, NTuple)
        ### Not sure what to call this
        ### it is for left hand side operator in
        ### ( ∇ʰ⋅H∇ʰ - 1/gΔt² ) ηⁿ⁺¹ = 1/(gΔt) ∇ʰH U̅ˢᵗᵃʳ - 1/(gΔt²) ηⁿ
        ### written in a discrete finite volume form in which the equation
        ### is arranged to ensure a symmtric form
        ### e.g.
        ### 
        ### δⁱÂʷ∂ˣηⁿ⁺¹ + δʲÂˢ∂ʸηⁿ⁺¹ - 1/gΔt² Aᶻηⁿ⁺¹ =
        ###  1/(gΔt)(δⁱÂʷu̅ˢᵗᵃʳ + δʲÂˢv̅ˢᵗᵃʳ) - 1/gΔt² Aᶻηⁿ
        ###
        ### where  ̂ indicates a vertical integral, and
        ###        ̅ indicates a vertical average
        ###
        ### g  = Main.model.free_surface.gravitational_acceleration ### AGHHHH - need to sort this out later.....
        ### Δt = Main.simulation.Δt                                 ### AGHHHH - need to sort this out later.....
        i, j = @index(Global, NTuple)
        @inbounds implicit_η_f[i, j, 1] =  ∇²_baro(i, j, 1, grid, f) - Azᵃᵃᵃ(i, j, 1, grid)*f[i,j, 1]/(g*Δt^2)
    end

    if isnothing( Amatrix_operator )
        function Amatrix_function!(result, x, arch, grid, bcs; args...)
            Δt=args.data.Δt
             g=args.data.g
            event = launch!(arch, grid, :xy, implicit_η!, Δt, g, grid, x, result, dependencies=Event(device(arch)))
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
