using Oceananigans.TimeSteppers: update_state!
using Oceananigans.Operators: intrinsic_vector, ℑxyᶠᶜᵃ, ℑxyᶜᶠᵃ
using Oceananigans.Utils: @apply_regionally, apply_regionally!

import Oceananigans.Fields: set!

"""
    set!(model::HydrostaticFreeSurfaceModel; kwargs...)

Set velocity and tracer fields of `model`. The keyword arguments `kwargs...`
take the form `name = data`, where `name` refers to one of the fields of either:
(i) `model.velocities`, (ii) `model.tracers`, or (iii) `model.free_surface.η`,
and the `data` may be an array, a function with arguments `(x, y, z)`, or any data type
for which a `set!(ϕ::AbstractField, data)` function exists.

Example
=======

```jldoctest
using Oceananigans
grid = RectilinearGrid(size=(16, 16, 16), extent=(1, 1, 1))
model = HydrostaticFreeSurfaceModel(; grid, tracers=:T)

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
                      u=ZeroField(), v=ZeroField(), intrinsic_velocities=false,
                      kwargs...)

    set_velocities!(model, u, v; intrinsic_velocities)

    for (fldname, value) in kwargs
        if fldname ∈ propertynames(model.velocities)
            ϕ = getproperty(model.velocities, fldname)
        elseif fldname ∈ propertynames(model.tracers)
            ϕ = getproperty(model.tracers, fldname)
        elseif fldname ∈ propertynames(model.free_surface)
            ϕ = getproperty(model.free_surface, fldname)
        else
            throw(ArgumentError("name $fldname not found in model.velocities, model.tracers, or model.free_surface"))
        end

        @apply_regionally set!(ϕ, value)
    end

    # initialize!(model)
    initialization_update_state!(model; compute_tendencies=false)

    return nothing
end


const IntrinsicCoordinateGrid = Union{
    OrthogonalSphericalShellGrid,
    ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, OrthogonalSphericalShellGrid},
}

"""
    set_velocities!(model, u, v; intrinsic_velocities=false)

Set the velocities of `model` from `u` and `v`.

If `intrinsic_velocities` is true, then `(u, v)` is assumed to be a horizontal vector
in the intrinsic coordinate system of the grid. Otherwise, `(u, v)` is assumed to represent
an extrinsic vector, and is rotated into the intrinsic coordinate system.

This abstraction is necessary for `OrthogonalSphericalShellGrid` and derivatives thereof,
where the extrinsic and intrinsic coordinate systems differ.
"""
function set_velocities!(model, u, v; intrinsic_velocities=false)
    if intrinsic_velocities || !(model.grid isa IntrinsicCoordinateGrid)
        u isa ZeroField || set!(model.velocities.u, u)
        v isa ZeroField || set!(model.velocities.v, v)
    else
        set_from_extrinsic_velocities!(model.velocities, model.grid, u, v)
    end
    return nothing
end

function set_from_extrinsic_velocities!(velocities, grid, u, v)
    grid = grid
    arch = grid.architecture
    uᶜᶜᶜ = CenterField(grid) 
    vᶜᶜᶜ = CenterField(grid) 
    u isa ZeroField || set!(uᶜᶜᶜ, u)
    v isa ZeroField || set!(vᶜᶜᶜ, v)
    launch!(arch, grid, :xyz, _rotate_velocities!, uᶜᶜᶜ, vᶜᶜᶜ, grid)
    fill_halo_regions!(uᶜᶜᶜ)
    fill_halo_regions!(vᶜᶜᶜ)
    launch!(arch, grid, :xyz, _interpolate_velocities!,
            velocities.u, velocities.v, grid, uᶜᶜᶜ, vᶜᶜᶜ)
    return nothing
end

@kernel function _rotate_velocities!(u, v, grid)
    i, j, k = @index(Global, NTuple)
    # Rotate u, v from extrinsic to intrinsic coordinate system
    ur, vr = intrinsic_vector(i, j, k, grid, u, v)
    @inbounds begin
        u[i, j, k] = ur
        v[i, j, k] = vr
    end
end

@kernel function _interpolate_velocities!(u, v, grid, uᶜᶜᶜ, vᶜᶜᶜ)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        u[i, j, k] = ℑxyᶠᶜᵃ(i, j, k, grid, uᶜᶜᶜ)
        v[i, j, k] = ℑxyᶜᶠᵃ(i, j, k, grid, vᶜᶜᶜ)
    end
end
