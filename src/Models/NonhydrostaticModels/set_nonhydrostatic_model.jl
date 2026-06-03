using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: compute!
using Oceananigans.TimeSteppers: update_state!
using Oceananigans.TurbulenceClosures: initialize_closure_fields!

import Oceananigans.Fields: set!

"""
    set!(model::NonhydrostaticModel; enforce_incompressibility=true, kwargs...)

Set velocity and tracer fields of `model`. The keyword arguments
`kwargs...` take the form `name=data`, where `name` refers to one of the
fields of `model.velocities` or `model.tracers`, and the `data` may be an array,
a function with arguments `(x, y, z)`, or any data type for which a
`set!(ϕ::AbstractField, data)` function exists.

Example
=======

```@example
using Oceananigans
grid = RectilinearGrid(size=(16, 16, 16), extent=(1, 1, 1))
model = NonhydrostaticModel(grid, tracers=:T)

# Set u to a parabolic function of z, v to random numbers damped
# at top and bottom, and T to some silly array of half zeros,
# half random numbers.

u₀(x, y, z) = z / model.grid.Lz * (1 + z / model.grid.Lz)
v₀(x, y, z) = 1e-3 * rand() * u₀(x, y, z)

T₀ = rand(size(model.grid)...)
T₀[T₀ .< 0.5] .= 0

set!(model, u=u₀, v=v₀, T=T₀)

model.tracers.T
```
"""
function set!(model::NonhydrostaticModel; enforce_incompressibility=true, kwargs...)
    velocity_names = propertynames(model.velocities)
    velocities_are_set = false
    set_u = :u in keys(kwargs)
    set_v = :v in keys(kwargs)

    if set_u || set_v
        u = set_u ? kwargs[:u] : Oceananigans.Fields.ZeroField()
        v = set_v ? kwargs[:v] : Oceananigans.Fields.ZeroField()

        if model.grid isa Oceananigans.Models.HydrostaticFreeSurfaceModels.IntrinsicCoordinateGrid
            Oceananigans.Models.HydrostaticFreeSurfaceModels.set_from_extrinsic_velocities!(model.velocities,
                                                                                            model.grid,
                                                                                            u,
                                                                                            v;
                                                                                            set_u,
                                                                                            set_v)
        else
            set!(model.velocities; u, v)
        end

        compute!(model.auxiliary_fields)
        fill_halo_regions!(model.velocities, model.clock, fields(model))
        velocities_are_set = true
    end

    for (fldname, value) in kwargs
        fldname in (:u, :v) && continue

        if fldname ∈ velocity_names
            ϕ = getproperty(model.velocities, fldname)
            velocities_are_set = true
        elseif fldname ∈ propertynames(model.tracers)
            ϕ = getproperty(model.tracers, fldname)
        elseif !isnothing(model.free_surface) && fldname ∈ propertynames(model.free_surface)
            ϕ = getproperty(model.free_surface, fldname)
        else
            throw(ArgumentError("name $fldname not found in model.velocities or model.tracers."))
        end
        set!(ϕ, value)

        compute!(model.auxiliary_fields)
        fill_halo_regions!(ϕ, model.clock, fields(model))
    end

    # Apply a mask
    foreach(mask_immersed_field!, model.tracers)
    foreach(mask_immersed_field!, model.velocities)
    update_state!(model)
    velocities_are_set && initialize_closure_fields!(model.closure_fields, model.closure, model)

    if enforce_incompressibility
        FT = eltype(model.grid)
        compute_pressure_correction!(model, one(FT))
        make_pressure_correction!(model, one(FT))
        update_state!(model)
        velocities_are_set && initialize_closure_fields!(model.closure_fields, model.closure, model)
    end

    return nothing
end
