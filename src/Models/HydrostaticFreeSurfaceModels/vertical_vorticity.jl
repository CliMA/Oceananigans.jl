using Oceananigans.Grids: Center, Face, SphericalShellGrid
using Oceananigans.Operators: covariant_to_contravariant_velocity_uᶠᶜᶜ,
                              covariant_to_contravariant_velocity_vᶜᶠᶜ,
                              covariant_kinetic_energyᶜᶜᶜ,
                              covariant_vertical_vorticity_componentᶠᶠᶜ,
                              covariant_vertical_vorticityᶠᶠᶜ,
                              ζ₃ᶠᶠᶜ
using Oceananigans.AbstractOperations: KernelFunctionOperation

@inline vertical_vorticity_kernel_function(grid) = ζ₃ᶠᶠᶜ
@inline vertical_vorticity_kernel_function(grid::SphericalShellGrid) = covariant_vertical_vorticityᶠᶠᶜ

"""
    vertical_vorticity(model::HydrostaticFreeSurfaceModel)

Return a `KernelFunctionOperation` that represents vertical vorticity for the
`HydrostaticFreeSurfaceModel`. The kernel function is `Oceananigans.Operators.ζ₃ᶠᶠᶜ`,
and thus computed consistently with the `VectorInvariant` momentum advection scheme
for curvilinear grids. For `SphericalShellGrid`, the kernel function is
`Oceananigans.Operators.covariant_vertical_vorticityᶠᶠᶜ`.
"""
function vertical_vorticity(model::HydrostaticFreeSurfaceModel)
    u = model.velocities.u
    v = model.velocities.v
    kernel_function = vertical_vorticity_kernel_function(model.grid)
    return KernelFunctionOperation{Face, Face, Center}(kernel_function, model.grid, u, v)
end

"""
    contravariant_velocities(model::HydrostaticFreeSurfaceModel)

Return a `NamedTuple` with
`u = J u^1`-type contravariant velocity on `fcc` and
`v = J u^2`-type contravariant velocity on `cfc`,
represented as `KernelFunctionOperation`s.

This currently supports `SphericalShellGrid` only.
"""
function contravariant_velocities(model::HydrostaticFreeSurfaceModel)
    return _contravariant_velocities(model, model.grid)
end

@inline function _contravariant_velocities(model::HydrostaticFreeSurfaceModel, grid::SphericalShellGrid)
    u = model.velocities.u
    v = model.velocities.v

    contravariant_u = KernelFunctionOperation{Face, Center, Center}(covariant_to_contravariant_velocity_uᶠᶜᶜ, grid, u, v)
    contravariant_v = KernelFunctionOperation{Center, Face, Center}(covariant_to_contravariant_velocity_vᶜᶠᶜ, grid, u, v)

    return (u = contravariant_u, v = contravariant_v)
end

function _contravariant_velocities(model::HydrostaticFreeSurfaceModel, grid)
    throw(ArgumentError("contravariant_velocities is currently implemented only for `SphericalShellGrid`, got $(typeof(grid))."))
end

"""
    kinetic_energy(model::HydrostaticFreeSurfaceModel)

Return a `KernelFunctionOperation` for the covariant pointwise kinetic energy density
\$\\frac12 g_{ij} u^i u^j\$. This currently supports `SphericalShellGrid` only.
"""
function kinetic_energy(model::HydrostaticFreeSurfaceModel)
    return _kinetic_energy(model, model.grid)
end

@inline function _kinetic_energy(model::HydrostaticFreeSurfaceModel, grid::SphericalShellGrid)
    u = model.velocities.u
    v = model.velocities.v
    return KernelFunctionOperation{Center, Center, Center}(covariant_kinetic_energyᶜᶜᶜ, grid, u, v)
end

function _kinetic_energy(model::HydrostaticFreeSurfaceModel, grid)
    throw(ArgumentError("kinetic_energy is currently implemented only for `SphericalShellGrid`, got $(typeof(grid))."))
end

"""
    relative_vorticity(model::HydrostaticFreeSurfaceModel)

Return a `KernelFunctionOperation` for the covariant vertical vorticity component
\$\\zeta_{12}\$ on `ffc` for a `HydrostaticFreeSurfaceModel`.
This currently supports `SphericalShellGrid` only.
"""
function relative_vorticity(model::HydrostaticFreeSurfaceModel)
    return _relative_vorticity(model, model.grid)
end

@inline function _relative_vorticity(model::HydrostaticFreeSurfaceModel, grid::SphericalShellGrid)
    u = model.velocities.u
    v = model.velocities.v
    return KernelFunctionOperation{Face, Face, Center}(covariant_vertical_vorticity_componentᶠᶠᶜ, grid, u, v)
end

function _relative_vorticity(model::HydrostaticFreeSurfaceModel, grid)
    throw(ArgumentError("relative_vorticity is currently implemented only for `SphericalShellGrid`, got $(typeof(grid))."))
end
