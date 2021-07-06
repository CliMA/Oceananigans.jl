using Oceananigans.Grids: xnode, ynode, znode, Center, AbstractGrid

import Oceananigans.TimeSteppers: correct_immersed_tendencies!

"""
    correct_immersed_tendencies!(model, Δt, γⁿ, ζⁿ)
    
Correct the tendency terms to implement no-slip boundary conditions on an immersed boundary
 without the contribution from the non-hydrostatic pressure. 
Makes velocity vanish within the immersed surface.
"""

correct_immersed_tendencies!(model::IncompressibleModel, Δt, γⁿ, ζⁿ) =
    correct_immersed_tendencies!(model, model.immersed_boundary, Δt, γⁿ, ζⁿ)

# if no immersed boundary, do nothing (no cost)
correct_immersed_tendencies!(model, ::Nothing, Δt, γⁿ, ζⁿ) = nothing

# otherwise, unpack the model
function correct_immersed_tendencies!(model, immersed_boundary, Δt, γⁿ, ζⁿ)

    workgroup, worksize = work_layout(model.grid, :xyz)

    barrier = Event(device(model.architecture))

    correct_immersed_tendencies_kernel! = _correct_immersed_tendencies!(device(model.architecture), workgroup, worksize)
    
    # event we want to occur, evaluate using kernel function
    correct_tendencies_event =
        correct_immersed_tendencies_kernel!(model.timestepper.Gⁿ,
                                            model.grid,
                                            immersed_boundary,
                                            model.timestepper.G⁻,
                                            model.velocities,
                                            Δt, γⁿ, ζⁿ,
                                            dependencies=barrier)

    # wait for these things to happen before continuing in calculations
    wait(device(model.architecture), correct_tendencies_event)

    return nothing
end

@kernel function _correct_immersed_tendencies!(Gⁿ, grid::AbstractGrid{FT}, immersed, G⁻, velocities, Δt, γⁿ, ζⁿ) where FT
    i, j, k = @index(Global, NTuple)
    
    # Evaluate x, y, z at cell centers to determine if node is immersed
    x = xnode(Center(), i, grid)
    y = ynode(Center(), j, grid)
    z = znode(Center(), k, grid)

    @inbounds begin
        # correcting velocity tendency terms: if immersd boundary gives true then correct tendency, otherwise don't (it's a fluid node)
        Gⁿ.u[i, j, k] = ifelse(immersed(x, y, z),
                               - (velocities.u[i, j, k] + ζⁿ * Δt * G⁻.u[i, j, k]) / (γⁿ * Δt),
                               Gⁿ.u[i, j, k])

        Gⁿ.v[i, j, k] = ifelse(immersed(x, y, z),
                               - (velocities.v[i, j, k] + ζⁿ * Δt * G⁻.v[i, j, k]) / (γⁿ * Δt),
                               Gⁿ.v[i, j, k])

        Gⁿ.w[i, j, k] = ifelse(immersed(x, y, z),
                               - (velocities.w[i, j, k] + ζⁿ * Δt * G⁻.w[i, j, k]) / (γⁿ * Δt),
                               Gⁿ.w[i, j, k])
    end
end
