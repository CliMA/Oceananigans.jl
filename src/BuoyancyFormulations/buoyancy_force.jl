using Oceananigans.Utils
using Oceananigans.BoundaryConditions
using Oceananigans.Fields: XFaceField, YFaceField, ZFaceField
using Oceananigans.Grids: NegativeZDirection, validate_unit_vector, architecture
using Oceananigans.ImmersedBoundaries: retrieve_interior_active_cells_map, active_linear_index_to_tuple
using KernelAbstractions: @kernel, @index

using Adapt

struct BuoyancyForce{M, G, B}
    formulation :: M
    gravity_unit_vector :: G
    gradients :: B
end

"""
    BuoyancyForce(formulation; gravity_unit_vector=NegativeZDirection())

Construct a `buoyancy` given a buoyancy `model`. Optional keyword argument `gravity_unit_vector`
can be used to specify the direction of gravity (default `NegativeZDirection()`).
The buoyancy acceleration acts in the direction opposite to gravity.

Example
=======

```jldoctest
using Oceananigans

grid = RectilinearGrid(size=(1, 8, 8), extent=(1, 1, 1))

θ = 45 # degrees
g̃ = (0, -sind(θ), -cosd(θ))

buoyancy = BuoyancyForce(BuoyancyTracer(), gravity_unit_vector=g̃)

model = NonhydrostaticModel(; grid, buoyancy, tracers=:b)

# output

NonhydrostaticModel{CPU, RectilinearGrid}(time = 0 seconds, iteration = 0)
├── grid: 1×8×8 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 1×3×3 halo
├── timestepper: RungeKutta3TimeStepper
├── advection scheme: Centered(order=2)
├── tracers: b
├── closure: Nothing
├── buoyancy: BuoyancyTracer with ĝ = (0.0, -0.707107, -0.707107)
└── coriolis: Nothing
```
"""
function BuoyancyForce(grid; formulation, gravity_unit_vector=NegativeZDirection(), precompute_buoyancy_gradients=true)
    gravity_unit_vector = validate_unit_vector(gravity_unit_vector)

    if precompute_buoyancy_gradients
        ∂x_b = XFaceField(grid)
        ∂y_b = YFaceField(grid)
        ∂z_b = ZFaceField(grid)

        buoyancy_gradients = (; ∂x_b, ∂y_b, ∂z_b)
    else
        buoyancy_gradients = nothing
    end

    return BuoyancyForce(formulation, gravity_unit_vector, buoyancy_gradients)
end

Adapt.adapt_structure(to, buoyancy::BuoyancyForce) = 
    BuoyancyForce(Adapt.adapt(to, buoyancy.formulation), 
                  Adapt.adapt(to, buoyancy.gravity_unit_vector),
                  Adapt.adapt(to, buoyancy.gradients))

@inline ĝ_x(bf) = @inbounds - bf.gravity_unit_vector[1]
@inline ĝ_y(bf) = @inbounds - bf.gravity_unit_vector[2]
@inline ĝ_z(bf) = @inbounds - bf.gravity_unit_vector[3]

@inline ĝ_x(::BuoyancyForce{M, NegativeZDirection}) where M = 0
@inline ĝ_y(::BuoyancyForce{M, NegativeZDirection}) where M = 0
@inline ĝ_z(::BuoyancyForce{M, NegativeZDirection}) where M = 1

#####
##### For convenience
#####

@inline required_tracers(bf::BuoyancyForce) = required_tracers(bf.formulation)

@inline get_temperature_and_salinity(bf::BuoyancyForce, C) = get_temperature_and_salinity(bf.formulation, C)

@inline ∂x_b(i, j, k, grid, b::BuoyancyForce{<:Any, <:Any, Nothing}, C) = ∂x_b(i, j, k, grid, b.formulation, C)
@inline ∂y_b(i, j, k, grid, b::BuoyancyForce{<:Any, <:Any, Nothing}, C) = ∂y_b(i, j, k, grid, b.formulation, C)
@inline ∂z_b(i, j, k, grid, b::BuoyancyForce{<:Any, <:Any, Nothing}, C) = ∂z_b(i, j, k, grid, b.formulation, C)

@inline ∂x_b(i, j, k, grid, b::BuoyancyForce, C) = @inbounds b.gradients.∂x_b[i, j, k]
@inline ∂y_b(i, j, k, grid, b::BuoyancyForce, C) = @inbounds b.gradients.∂y_b[i, j, k]
@inline ∂z_b(i, j, k, grid, b::BuoyancyForce, C) = @inbounds b.gradients.∂z_b[i, j, k]

@inline top_buoyancy_flux(i, j, grid, b::BuoyancyForce, args...) = top_buoyancy_flux(i, j, grid, b.formulation, args...)

regularize_buoyancy(bf, grid) = bf
regularize_buoyancy(formulation::AbstractBuoyancyFormulation, grid) = BuoyancyForce(grid; formulation)

compute_buoyancy_gradients!(::BuoyancyForce{<:Any, <:Any, <:Nothing}, grid, tracers) = nothing

function compute_buoyancy_gradients!(buoyancy, grid, tracers; parameters=:xyz) 

    active_cells_map = retrieve_interior_active_cells_map(grid, Val(:interior))

    gradients = buoyancy.gradients
    formulation = buoyancy.formulation
    launch!(architecture(grid), grid, parameters, _compute_buoyancy_gradients!, 
            gradients, grid, active_cells_map, formulation, tracers; active_cells_map)

    fill_halo_regions!((∂x_b, ∂y_b, ∂z_b); only_local_halos = true)

    return nothing
end

@kernel function _compute_buoyancy_gradients!(gradients, grid, map, buoyancy, tracers)
    idx = @index(Global, Linear)
    i, j, k = active_linear_index_to_tuple(idx, map)
    @inbounds gradients.∂x_b[i, j, k] = ∂x_b(i, j, k, grid, buoyancy, tracers)
    @inbounds gradients.∂y_b[i, j, k] = ∂y_b(i, j, k, grid, buoyancy, tracers)
    @inbounds gradients.∂z_b[i, j, k] = ∂z_b(i, j, k, grid, buoyancy, tracers)
end

@kernel function _compute_buoyancy_gradients!(gradients, grid, ::Nothing, buoyancy, tracers)
    i, j, k = @index(Global, NTuple)
    @inbounds gradients.∂x_b[i, j, k] = ∂x_b(i, j, k, grid, buoyancy, tracers)
    @inbounds gradients.∂y_b[i, j, k] = ∂y_b(i, j, k, grid, buoyancy, tracers)
    @inbounds gradients.∂z_b[i, j, k] = ∂z_b(i, j, k, grid, buoyancy, tracers)
end

Base.summary(bf::BuoyancyForce) = string(summary(bf.formulation),
                                         " with ĝ = ",
                                         summarize_vector(bf.gravity_unit_vector))

summarize_vector(n) = string("(", prettysummary(n[1]), ", ",
                                  prettysummary(n[2]), ", ",
                                  prettysummary(n[3]), ")")
                             
summarize_vector(::NegativeZDirection) = "NegativeZDirection()"

function Base.show(io::IO, bf::BuoyancyForce)
    print(io, "BuoyancyForce:", '\n',
              "├── formulation: ", prettysummary(bf.formulation), '\n',
              "└── gravity_unit_vector: ", summarize_vector(bf.gravity_unit_vector))
end
