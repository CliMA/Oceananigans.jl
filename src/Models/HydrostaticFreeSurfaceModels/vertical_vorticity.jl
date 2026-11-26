using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Grids: Center, Face
using Oceananigans.Operators: ζ₃ᶠᶠᶜ

"""
$(TYPEDSIGNATURES)

Return a `KernelFunctionOperation` that represents vertical vorticity for the
`HydrostaticFreeSurfaceModel`. The kernel function is `Oceananigans.Operators.ζ₃ᶠᶠᶜ`,
and thus computed consistently with the `VectorInvariant` momentum advection scheme
for curvilinear grids.
"""
function vertical_vorticity(model::HydrostaticFreeSurfaceModel)
    u, v, w = model.velocities
    return KernelFunctionOperation{Face, Face, Center}(ζ₃ᶠᶠᶜ, model.grid, u, v)
end
