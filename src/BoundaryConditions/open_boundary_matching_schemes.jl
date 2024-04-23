using Oceananigans.Architectures: architecture, on_architecture
using Oceananigans.Operators: Axᶜᶜᶠ, Ayᶜᶠᶜ, Azᶜᶜᶠ

"""
    MeanOutflow

Advect out of the domain with the boundary mean flow, and relax inflows to external state.
"""
struct MeanOutflow{BF, IT}
                boundary_flux :: BF
        mean_outflow_velocity :: AbstractArray
  inflow_relaxation_timescale :: IT
end

(ms::MeanOutflow)() = ms

Adapt.adapt_structure(to, mo::MeanOutflow) = 
    MeanOutflow(nothing, adapt(to, mo.mean_outflow_velocity), adapt(to, mo.inflow_relaxation_timescale))

function MeanOutflowOpenBoundaryCondition(grid, side, val; inflow_relaxation_timescale = 1, kwargs...)
    boundary_flux = boundary_flux_field(grid, Val(side))

    classifcation = Open(MeanOutflow(boundary_flux, [0.], inflow_relaxation_timescale))
    
    return BoundaryCondition(classifcation, val; kwargs...)
end

# fields is not yet defiled so have to use an array
boundary_flux_field(grid, ::Union{Val{:west}, Val{:east}}) = on_architecture(architecture(grid), zeros(size(grid, 2), size(grid, 3)))
boundary_flux_field(grid, ::Union{Val{:south}, Val{:north}}) = on_architecture(architecture(grid), zeros(size(grid, 1), size(grid, 3)))
boundary_flux_field(grid, ::Union{Val{:bottom}, Val{:top}}) = on_architecture(architecture(grid), zeros(size(grid, 2), size(grid, 3)))

const MOOBC = BoundaryCondition{<:Open{<:MeanOutflow}}

boundary_normal_velocity(velocities, ::Union{Val{:west}, Val{:east}}) = velocities.u
boundary_normal_velocity(velocities, ::Union{Val{:south}, Val{:north}}) = velocities.v
boundary_normal_velocity(velocities, ::Union{Val{:bottom}, Val{:top}}) = velocities.w

function update_boundary_condition!(bc::MOOBC, field, model, side)
    ms = bc.classification.matching_scheme

    u = boundary_normal_velocity(model.velocities, side)
    F = ms.boundary_flux

    arch = architecture(model)
    grid = model.grid

    launch!(arch, grid, :yz, _update_boundary_flux!, F, grid, u, side)

    ms.mean_outflow_velocity[1] = sum(F) / (grid.Ly * grid.Lz)
end

@kernel function _update_boundary_flux!(F, grid, u, ::Val{:west})
    j, k = @index(Global, NTuple)

    @inbounds F[j, k] = -u[1, j, k] * Axᶜᶜᶠ(1, j, k, grid)
end

@kernel function _update_boundary_flux!(F, grid, u, ::Val{:east})
    j, k = @index(Global, NTuple)

    i = grid.Nx

    @inbounds F[j, k] = u[i, j, k] * Axᶜᶜᶠ(i, j, k, grid)
end

@kernel function _update_boundary_flux!(F, grid, u, ::Val{:south})
    i, k = @index(Global, NTuple)

    @inbounds F[i, k] = -u[i, 1, k] * Ayᶜᶠᶜ(i, 1, k, grid)
end

@kernel function _update_boundary_flux!(F, grid, u, ::Val{:north})
    i, k = @index(Global, NTuple)

    j = grid.Ny

    @inbounds F[i, k] = u[i, j, k] * Ayᶜᶠᶜ(i, j, k, grid)
end

@kernel function _update_boundary_flux!(F, grid, u, ::Val{:bottom})
    i, j = @index(Global, NTuple)

    @inbounds F[i, j] = -u[i, j, 1] * Azᶜᶜᶠ(i, j, 1, grid)
end

@kernel function _update_boundary_flux!(F, grid, u, ::Val{:top})
    i, j = @index(Global, NTuple)

    k = grid.Nz

    @inbounds F[i, j] = u[i, j, k] * Azᶜᶜᶠ(i, j, k, grid)
end
