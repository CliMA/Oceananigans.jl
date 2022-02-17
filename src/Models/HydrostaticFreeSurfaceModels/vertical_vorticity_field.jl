using Oceananigans.Grids: Face, Face, Center
using Oceananigans.Operators: ζ₃ᶠᶠᶜ
using Oceananigans.AbstractOperations: KernelFunctionOperation

"""
    VerticalVorticityField(model)

Returns a Field that `compute!`s vertical vorticity in a
manner consistent with the `VectorInvariant` momentum advection scheme
for curvilinear grids.

In particular, `VerticalVorticityField` uses `ζ₃ᶠᶠᶜ`, which in turn computes
the vertical vorticity by first integrating the velocity field around the borders
of the vorticity cell to find the vertical circulation, and then dividing by the area of
the vorticity cell to compute vertical vorticity.
"""
function VerticalVorticityField(model; kw...)
    grid = model.grid
    u, v, w = model.velocities
    vorticity_operation = KernelFunctionOperation{Face, Face, Center}(ζ₃ᶠᶠᶜ, grid, computed_dependencies=(u, v))
    return Field(vorticity_operation; kw...)
end

