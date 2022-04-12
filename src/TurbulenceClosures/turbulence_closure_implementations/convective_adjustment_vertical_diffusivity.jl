using Oceananigans.Architectures: architecture, device_event, arch_array
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.BuoyancyModels: ∂z_b
using Oceananigans.Operators: ℑzᵃᵃᶜ

struct ConvectiveAdjustmentVerticalDiffusivity{TD, CK, CN, BK, BN} <: AbstractScalarDiffusivity{TD, VerticalFormulation}
    convective_κz :: CK
    convective_νz :: CN
    background_κz :: BK
    background_νz :: BN

    function ConvectiveAdjustmentVerticalDiffusivity{TD}(convective_κz::CK,
                                                         convective_νz::CN,
                                                         background_κz::BK,
                                                         background_νz::BN) where {TD, CK, CN, BK, BN}

        return new{TD, CK, CN, BK, BN}(convective_κz, convective_νz, background_κz, background_νz)
    end
end

"""
    ConvectiveAdjustmentVerticalDiffusivity([time_discretization = VerticallyImplicitTimeDiscretization(), FT=Float64;]
                                            convective_κz = 0,
                                            convective_νz = 0,
                                            background_κz = 0,
                                            background_νz = 0)

Return a convective adjustment vertical diffusivity closure that applies different values of diffusivity and/or viscosity depending
whether the region is statically stable (positive or zero buoyancy gradient) or statically unstable (negative buoyancy gradient).

Arguments
=========

* `time_discretization`: Either `ExplicitTimeDiscretization()` or `VerticallyImplicitTimeDiscretization()`;
                         default `VerticallyImplicitTimeDiscretization()`.

* `FT`: Float type; default `Float64`.

Keyword arguments
=================

* `convective_κz`: Vertical tracer diffusivity in regions with negative (unstable) buoyancy gradients. Either
                   a single number, function, array, field, or tuple of diffusivities for each tracer.

* `background_κz`: Vertical tracer diffusivity in regions with zero or positive (stable) buoyancy gradients.

* `convective_νz`: Vertical viscosity in regions with negative (unstable) buoyancy gradients. Either
                  a number, function, array, or field.

* `background_κz`: Vertical viscosity in regions with zero or positive (stable) buoyancy gradients.

Example
=======

```jldoctest
julia> using Oceananigans

julia> cavd = ConvectiveAdjustmentVerticalDiffusivity(convective_κz = 1)
ConvectiveAdjustmentVerticalDiffusivity{VerticallyImplicitTimeDiscretization}(background_κz=0.0 convective_κz=1 background_νz=0.0 convective_νz=0.0)
```
"""
function ConvectiveAdjustmentVerticalDiffusivity(time_discretization = VerticallyImplicitTimeDiscretization(), FT = Float64;
                                                 convective_κz = zero(FT),
                                                 convective_νz = zero(FT),
                                                 background_κz = zero(FT),
                                                 background_νz = zero(FT))

    return ConvectiveAdjustmentVerticalDiffusivity{typeof(time_discretization)}(convective_κz, convective_νz,
                                                                                background_κz, background_νz)
end

ConvectiveAdjustmentVerticalDiffusivity(FT::DataType; kwargs...) = ConvectiveAdjustmentVerticalDiffusivity(VerticallyImplicitTimeDiscretization(), FT; kwargs...)

const CAVD = ConvectiveAdjustmentVerticalDiffusivity

#####
##### Diffusivity field utilities
#####

# Support for "ManyIndependentColumnMode"
const CAVDArray = AbstractArray{<:CAVD}
const FlavorOfCAVD = Union{CAVD, CAVDArray}

with_tracers(tracers, closure::FlavorOfCAVD) = closure
diffusivity_fields(grid, tracer_names, bcs, closure::FlavorOfCAVD) = (; κ = ZFaceField(grid), ν = ZFaceField(grid))
@inline viscosity_location(::FlavorOfCAVD) = (Center(), Center(), Face())
@inline diffusivity_location(::FlavorOfCAVD) = (Center(), Center(), Face())
@inline viscosity(::FlavorOfCAVD, diffusivities) = diffusivities.ν
@inline diffusivity(::FlavorOfCAVD, diffusivities, id) = diffusivities.κ

function calculate_diffusivities!(diffusivities, closure::FlavorOfCAVD, model)

    arch = model.architecture
    grid = model.grid
    tracers = model.tracers
    buoyancy = model.buoyancy

    event = launch!(arch, grid, :xyz,
                    ## If we can figure out how to only precompute the "stability" of a cell:
                    # compute_stability!, diffusivities, grid, closure, tracers, buoyancy,
                    compute_convective_adjustment_diffusivities!, diffusivities, grid, closure, tracers, buoyancy,
                    dependencies = device_event(arch))

    wait(device(arch), event)

    return nothing
end

@inline is_stableᶜᶜᶠ(i, j, k, grid, tracers, buoyancy) = ∂z_b(i, j, k, grid, buoyancy, tracers) >= 0

@kernel function compute_convective_adjustment_diffusivities!(diffusivities, grid, closure, tracers, buoyancy)
    i, j, k, = @index(Global, NTuple)

    # Ensure this works with "ensembles" of closures, in addition to ordinary single closures
    closure_ij = getclosure(i, j, closure)

    stable_cell = is_stableᶜᶜᶠ(i, j, k, grid, tracers, buoyancy)

    @inbounds diffusivities.κ[i, j, k] = ifelse(stable_cell,
                                                closure_ij.background_κz,
                                                closure_ij.convective_κz)

    @inbounds diffusivities.ν[i, j, k] = ifelse(stable_cell,
                                                closure_ij.background_νz,
                                                closure_ij.convective_νz)
end

#=
## If we can figure out how to only precompute the "stability" of a cell:
@kernel function compute_stability!(diffusivities, grid, closure, tracers, buoyancy)
    i, j, k, = @index(Global, NTuple)
    @inbounds diffusivities.unstable_buoyancy_gradient[i, j, k] = is_unstableᶜᶜᶠ(i, j, k, grid, tracers, buoyancy)
end
=#

#####
##### Show
#####

function Base.summary(closure::ConvectiveAdjustmentVerticalDiffusivity{TD}) where TD
    return string("ConvectiveAdjustmentVerticalDiffusivity{$TD}" *
        "(background_κz=", prettysummary(closure.background_κz), " convective_κz=", prettysummary(closure.convective_κz),
        " background_νz=", prettysummary(closure.background_νz), " convective_νz=", prettysummary(closure.convective_νz), ")")
end

Base.show(io::IO, closure::ConvectiveAdjustmentVerticalDiffusivity) = print(io, summary(closure))

