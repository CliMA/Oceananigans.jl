using Oceananigans.Operators: intrinsic_vector, extrinsic_vector
using Oceananigans.Utils: @apply_regionally, KernelParameters
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Grids: OrthogonalSphericalShellGrid, SphericalShellGrid
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid

import Oceananigans.Fields: set!

"""
    set!(model::HydrostaticFreeSurfaceModel; kwargs...)

Set velocity and tracer fields of `model`. The keyword arguments `kwargs...`
take the form `name = data`, where `name` refers to one of the fields of either:
(i) `model.velocities`, (ii) `model.tracers`, or (iii) `model.free_surface.displacement`,
and the `data` may be an array, a function with arguments `(x, y, z)`, or any data type
for which a `set!(ϕ::AbstractField, data)` function exists.

Example
=======

```jldoctest
using Oceananigans
grid = RectilinearGrid(size=(16, 16, 16), extent=(1, 1, 1))
model = HydrostaticFreeSurfaceModel(grid; tracers=:T)

# Set u to a parabolic function of z, v to random numbers damped
# at top and bottom, and T to some silly array of half zeros,
# half random numbers.

u₀(x, y, z) = z / model.grid.Lz * (1 + z / model.grid.Lz)
v₀(x, y, z) = 1e-3 * rand() * u₀(x, y, z)

T₀ = rand(size(model.grid)...)
T₀[T₀ .< 0.5] .= 0

set!(model, u=u₀, v=v₀, T=T₀)

model.velocities.u

# output

16×16×16 Field{Face, Center, Center} on RectilinearGrid on CPU
├── grid: 16×16×16 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── boundary conditions: FieldBoundaryConditions
│   └── west: Periodic, east: Periodic, south: Periodic, north: Periodic, bottom: ZeroFlux, top: ZeroFlux, immersed: Nothing
└── data: 22×22×22 OffsetArray(::Array{Float64, 3}, -2:19, -2:19, -2:19) with eltype Float64 with indices -2:19×-2:19×-2:19
    └── max=-0.0302734, min=-0.249023, mean=-0.166992
```
"""
@inline function set!(model::HydrostaticFreeSurfaceModel;
                      u=nothing, v=nothing, intrinsic_velocities=false,
                      reconcile_state=true,
                      kwargs...)
    u_is_omitted = isnothing(u)
    v_is_omitted = isnothing(v)

    u = u_is_omitted ? ZeroField() : u
    v = v_is_omitted ? ZeroField() : v

    velocity_fields_are_set = !(u isa ZeroField && v isa ZeroField)
    free_surface_fields_are_set = false

    set_velocities!(model, u, v;
                    intrinsic_velocities,
                    set_u = !u_is_omitted,
                    set_v = !v_is_omitted)

    for (fldname, value) in kwargs
        if fldname ∈ propertynames(model.velocities)
            ϕ = getproperty(model.velocities, fldname)
            velocity_fields_are_set = true
        elseif fldname ∈ propertynames(model.tracers)
            ϕ = getproperty(model.tracers, fldname)
        elseif fldname ∈ propertynames(model.free_surface)
            ϕ = getproperty(model.free_surface, fldname)
            free_surface_fields_are_set = true
        elseif fldname === :η
            # The free surface displacement is accessed via `model.free_surface.displacement`
            # but the public interface uses `η` as the canonical name.
            ϕ = model.free_surface.displacement
            free_surface_fields_are_set = true
        else
            throw(ArgumentError("name $fldname not found in model.velocities, model.tracers, or model.free_surface"))
        end

        @apply_regionally set!(ϕ, value)
    end

    if velocity_fields_are_set || free_surface_fields_are_set
        compute_auxiliary_fields!(model.auxiliary_fields)
        fill_halo_regions!(model.velocities, model.clock, fields(model))
        invoke(compute_transport_velocities!,
               Tuple{HydrostaticFreeSurfaceModel, Any},
               model,
               model.free_surface)
    end

    if velocity_fields_are_set
        Oceananigans.TurbulenceClosures.refresh_velocity_dependent_closure_fields!(model.closure_fields, model.closure, model)
    end

    reconcile_state && reconcile_state!(model)
    update_state!(model)
    velocity_fields_are_set && initialize_closure_fields!(model.closure_fields, model.closure, model)

    return nothing
end


const IntrinsicCoordinateGrid = Union{
    OrthogonalSphericalShellGrid,
    SphericalShellGrid,
    ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, OrthogonalSphericalShellGrid},
    ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, SphericalShellGrid},
}

"""
    set_velocities!(model, u, v; intrinsic_velocities=false)

Set the velocities of `model` from `u` and `v`.

If `intrinsic_velocities` is true, then `(u, v)` is assumed to be a horizontal vector
in the intrinsic coordinate system of the grid. Otherwise, `(u, v)` is assumed to represent
an extrinsic vector, and is rotated into the intrinsic coordinate system.

This abstraction is necessary for spherical curvilinear grids, including
`OrthogonalSphericalShellGrid`, `SphericalShellGrid`, and their immersed-boundary
derivatives, where the extrinsic and intrinsic coordinate systems differ.
"""
function set_velocities!(model, u, v; intrinsic_velocities=false, set_u=true, set_v=true)
    if intrinsic_velocities || !(model.grid isa IntrinsicCoordinateGrid)
        set_intrinsic_velocities!(model.velocities, u, v; set_u, set_v)
    else
        set_from_extrinsic_velocities!(model.velocities, model.grid, u, v; set_u, set_v)
    end
    return nothing
end

function set_intrinsic_velocities!(velocities, u, v; set_u=true, set_v=true)
    if Oceananigans.Fields.requires_single_component_quadfolded_vector_field_set(velocities.u, velocities.v, u, v)
        Oceananigans.Fields.set_single_component_quadfolded_vector_fields!(velocities.u, velocities.v, u, v)
    elseif Oceananigans.Fields.requires_paired_quadfolded_vector_field_set(velocities.u, velocities.v, u, v)
        Oceananigans.Fields.set_paired_quadfolded_vector_fields!(velocities.u, velocities.v, u, v)
    else
        set_u && set!(velocities.u, u)
        set_v && set!(velocities.v, v)
    end

    return nothing
end

function set_from_extrinsic_velocities!(velocities, grid, u, v; set_u=true, set_v=true)
    !set_u && !set_v && return nothing

    grid = grid
    arch = grid.architecture

    uᶠᶜᶜ = Oceananigans.Fields.XFaceField(grid)
    vᶠᶜᶜ = Oceananigans.Fields.XFaceField(grid)
    uᶜᶠᶜ = Oceananigans.Fields.YFaceField(grid)
    vᶜᶠᶜ = Oceananigans.Fields.YFaceField(grid)

    xface_parameters = KernelParameters(Oceananigans.Grids.interior_indices(velocities.u))

    yface_parameters = KernelParameters(Oceananigans.Grids.interior_indices(velocities.v))

    launch!(arch, grid, xface_parameters, _compute_current_extrinsic_xface_velocities!, uᶠᶜᶜ, vᶠᶜᶜ, grid, velocities.u, velocities.v)
    launch!(arch, grid, yface_parameters, _compute_current_extrinsic_yface_velocities!, uᶜᶠᶜ, vᶜᶠᶜ, grid, velocities.u, velocities.v)

    set_u && begin
        set!(uᶠᶜᶜ, u)
        set!(uᶜᶠᶜ, u)
    end

    set_v && begin
        set!(vᶠᶜᶜ, v)
        set!(vᶜᶠᶜ, v)
    end

    launch!(arch, grid, xface_parameters, _rotate_xface_velocities!, velocities.u, grid, uᶠᶜᶜ, vᶠᶜᶜ)
    launch!(arch, grid, yface_parameters, _rotate_yface_velocities!, velocities.v, grid, uᶜᶠᶜ, vᶜᶠᶜ)
    fill_halo_regions!((velocities.u, velocities.v))
    return nothing
end

@kernel function _compute_current_extrinsic_xface_velocities!(uₑ, vₑ, grid, uᵢ, vᵢ)
    i, j, k = @index(Global, NTuple)
    ue, ve = extrinsic_vector(i, j, k, grid,
                              Oceananigans.Grids.Face(),
                              Oceananigans.Grids.Center(),
                              Oceananigans.Grids.Center(),
                              uᵢ, vᵢ)
    @inbounds begin
        uₑ[i, j, k] = ue
        vₑ[i, j, k] = ve
    end
end

@kernel function _compute_current_extrinsic_yface_velocities!(uₑ, vₑ, grid, uᵢ, vᵢ)
    i, j, k = @index(Global, NTuple)
    ue, ve = extrinsic_vector(i, j, k, grid,
                              Oceananigans.Grids.Center(),
                              Oceananigans.Grids.Face(),
                              Oceananigans.Grids.Center(),
                              uᵢ, vᵢ)
    @inbounds begin
        uₑ[i, j, k] = ue
        vₑ[i, j, k] = ve
    end
end

@kernel function _rotate_xface_velocities!(u, grid, uₑ, vₑ)
    i, j, k = @index(Global, NTuple)
    ur, _ = intrinsic_vector(i, j, k, grid,
                             Oceananigans.Grids.Face(),
                             Oceananigans.Grids.Center(),
                             Oceananigans.Grids.Center(),
                             uₑ, vₑ)
    @inbounds u[i, j, k] = ur
end

@kernel function _rotate_yface_velocities!(v, grid, uₑ, vₑ)
    i, j, k = @index(Global, NTuple)
    _, vr = intrinsic_vector(i, j, k, grid,
                             Oceananigans.Grids.Center(),
                             Oceananigans.Grids.Face(),
                             Oceananigans.Grids.Center(),
                             uₑ, vₑ)
    @inbounds v[i, j, k] = vr
end
