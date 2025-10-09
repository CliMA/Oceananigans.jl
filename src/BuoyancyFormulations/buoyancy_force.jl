using Oceananigans.Utils
using Oceananigans.Fields
using Oceananigans.Grids: NegativeZDirection, validate_unit_vector, architecture
using Oceananigans.BoundaryConditions

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
function BuoyancyForce(grid; formulation, gravity_unit_vector=NegativeZDirection(), precompute_gradients=true)
    gravity_unit_vector = validate_unit_vector(gravity_unit_vector)

    if precompute_gradients
        ∂x_b = XFaceField(grid)
        ∂y_b = YFaceField(grid)
        ∂z_b = ZFaceField(grid)

        gradients = (; ∂x_b, ∂y_b, ∂z_b)
    else
        gradients = nothing
    end

    return BuoyancyForce(formulation, gravity_unit_vector, gradients)
end

Adapt.adapt_structure(to, bf::BuoyancyForce) = 
    BuoyancyForce(Adapt.adapt(to, bf.formulation), 
                  Adapt.adapt(to, bf.gravity_unit_vector), 
                  Adapt.adapt(to, bf.gradients))

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

@inline top_buoyancy_flux(i, j, grid, b::BuoyancyForce, args...) = top_buoyancy_flux(i, j, grid, b.formulation, args...)

regularize_buoyancy(bf, grid; kw...) = bf
regularize_buoyancy(formulation::AbstractBuoyancyFormulation, grid; kw...) = BuoyancyForce(grid; formulation, kw...)

# Fallback
compute_buoyancy_gradients!(::BuoyancyForce{<:Any, <:Any, <:Nothing}, grid, tracers; kw...) = nothing
compute_buoyancy_gradients!(::Nothing, grid, tracers; kw...) = nothing     

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

#####
##### Some performance optimizations for models that compute gradients over and over...
#####

@inline ∂x_b(i, j, k, grid, b::BuoyancyForce, C) = @inbounds b.gradients.∂x_b[i, j, k]
@inline ∂y_b(i, j, k, grid, b::BuoyancyForce, C) = @inbounds b.gradients.∂y_b[i, j, k]
@inline ∂z_b(i, j, k, grid, b::BuoyancyForce, C) = @inbounds b.gradients.∂z_b[i, j, k]

function compute_buoyancy_gradients!(buoyancy, grid, tracers; parameters=:xyz)     
    gradients = buoyancy.gradients
    formulation = buoyancy.formulation
    launch!(architecture(grid), grid, parameters, _compute_buoyancy_gradients!, gradients, grid, formulation, tracers)
    fill_halo_regions!(gradients, only_local_halos=true)

    return nothing
end

@kernel function _compute_buoyancy_gradients!(g, grid, b, C)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        g.∂x_b[i, j, k] = ∂x_b(i, j, k, grid, b, C)
        g.∂y_b[i, j, k] = ∂y_b(i, j, k, grid, b, C)
        g.∂z_b[i, j, k] = ∂z_b(i, j, k, grid, b, C)
    end
end
