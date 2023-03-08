using Oceananigans.Grids: NegativeZDirection, validate_unit_vector

struct Buoyancy{M, G}
    model :: M
    gravity_unit_vector :: G
end

"""
    Buoyancy(; model, gravity_unit_vector=NegativeZDirection())

Uses a given buoyancy `model` to create buoyancy in a model. The optional keyword argument
`gravity_unit_vector` can be used to specify the direction of gravity, and the buoyancy acceleration
will act in the opposite direction.

Example
=======

```jldoctest

using Oceananigans

grid = RectilinearGrid(size=(1, 8, 8), extent=(1, 1000, 100))

θ = 45 # degrees
g̃ = (0, sind(θ), cosd(θ))

buoyancy = Buoyancy(model=BuoyancyTracer(), gravity_unit_vector=g̃)

model = NonhydrostaticModel(grid=grid, buoyancy=buoyancy, tracers=:b)

# output

┌ Warning: The behavior of `gravity_unit_vector` changed in version 0.80.0.
│ Prior to this version, `gravity_unit_vector` indicated the direction _opposite_ to gravity.
│ After version 0.80.0, `gravity_unit_vector` indicates the direction of the gravitional acceleration
└ @ Oceananigans.BuoyancyModels ~/repos/Oceananigans.jl/src/BuoyancyModels/buoyancy.jl:44
NonhydrostaticModel{CPU, RectilinearGrid}(time = 0 seconds, iteration = 0)
├── grid: 1×8×8 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── timestepper: QuasiAdamsBashforth2TimeStepper
├── tracers: b
├── closure: Nothing
├── buoyancy: BuoyancyTracer with ĝ = Tuple{Int64, Float64, Float64}
└── coriolis: Nothing
```
"""
function Buoyancy(; model, gravity_unit_vector=NegativeZDirection())
    gravity_unit_vector != NegativeZDirection() &&
        @warn "The behavior of `gravity_unit_vector` changed in version 0.80.0.
Prior to this version, `gravity_unit_vector` indicated the direction _opposite_ to gravity.
After version 0.80.0, `gravity_unit_vector` indicates the direction of the gravitional acceleration"
    gravity_unit_vector = validate_unit_vector(gravity_unit_vector)
    return Buoyancy(model, gravity_unit_vector)
end


@inline ĝ_x(buoyancy) = @inbounds -buoyancy.gravity_unit_vector[1]
@inline ĝ_y(buoyancy) = @inbounds -buoyancy.gravity_unit_vector[2]
@inline ĝ_z(buoyancy) = @inbounds -buoyancy.gravity_unit_vector[3]

@inline ĝ_x(::Buoyancy{M, NegativeZDirection}) where M = 0
@inline ĝ_y(::Buoyancy{M, NegativeZDirection}) where M = 0
@inline ĝ_z(::Buoyancy{M, NegativeZDirection}) where M = 1

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

Base.summary(buoyancy::Buoyancy) = string(summary(buoyancy.model), " with ĝ = ", summary(buoyancy.gravity_unit_vector))

Base.show(io::IO, buoyancy::Buoyancy) = print(io, sprint(show, buoyancy.model), "\nwith `gravity_unit_vector` = ", summary(buoyancy.gravity_unit_vector))

