using Oceananigans.Grids: ZDirection, validate_unit_vector

struct Buoyancy{M, G}
                   model :: M
     gravity_unit_vector :: G
end


"""
    Buoyancy(; model, gravity_unit_vector=ZDirection())

Uses a given buoyancy `model` to create buoyancy in a model. The optional keyword argument 
`gravity_unit_vector` can be used to specify the direction opposite to the gravitational
acceleration (which we take here to mean the "vertical" direction).

Example
=======

```julia
using Oceananigans

grid = RectilinearGrid(size=(1, 8, 8), extent=(1, 1000, 100))
θ = 45 # degrees
g̃ = (0, sind(θ), cosd(θ))

buoyancy = Buoyancy(model=BuoyancyTracer(), gravity_unit_vector=g̃)

model = NonhydrostaticModel(grid=grid, buoyancy=buoyancy, tracers=:b)
```
"""
function Buoyancy(; model, gravity_unit_vector=ZDirection())
    gravity_unit_vector = validate_unit_vector(gravity_unit_vector)
    return Buoyancy(model, gravity_unit_vector)
end


@inline ĝ_x(buoyancy) = @inbounds buoyancy.gravity_unit_vector[1]
@inline ĝ_y(buoyancy) = @inbounds buoyancy.gravity_unit_vector[2]
@inline ĝ_z(buoyancy) = @inbounds buoyancy.gravity_unit_vector[3]

@inline ĝ_x(::Buoyancy{M, ZDirection}) where M = 0
@inline ĝ_y(::Buoyancy{M, ZDirection}) where M = 0
@inline ĝ_z(::Buoyancy{M, ZDirection}) where M = 1

#####
##### For convenience
#####

@inline required_tracers(bm::Buoyancy) = required_tracers(bm.model)

@inline get_temperature_and_salinity(bm::Buoyancy, C) = get_temperature_and_salinity(bm.model, C)

@inline ∂x_b(i, j, k, grid, b::Buoyancy, C) = ∂x_b(i, j, k, grid, b.model, C)
@inline ∂y_b(i, j, k, grid, b::Buoyancy, C) = ∂y_b(i, j, k, grid, b.model, C)
@inline ∂z_b(i, j, k, grid, b::Buoyancy, C) = ∂z_b(i, j, k, grid, b.model, C)

@inline top_buoyancy_flux(i, j, grid, b::Buoyancy, args...) = top_buoyancy_flux(i, j, grid, b.model, args...)

regularize_buoyancy(b) = b
regularize_buoyancy(b::AbstractBuoyancyModel) = Buoyancy(model=b)
