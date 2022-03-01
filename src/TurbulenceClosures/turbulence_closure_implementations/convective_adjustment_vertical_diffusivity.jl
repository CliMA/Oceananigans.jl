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
    ConvectiveAdjustmentVerticalDiffusivity([FT=Float64;]
                                            convective_κz = 0,
                                            convective_νz = 0,
                                            background_κz = 0,
                                            background_νz = 0,
                                            time_discretization = VerticallyImplicittime_discretization())

The one positional argument determines the floating point type of the free parameters
of `ConvectiveAdjustmentVerticalDiffusivity`. The default is `Float64`.

Keyword arguments
=================

* `convective_κz`: Vertical tracer diffusivity in regions with negative (unstable) buoyancy gradients. Either
                   a single number, function, array, field, or tuple of diffusivities for each tracer.

* `background_κz`: Vertical tracer diffusivity in regions with zero or positive (stable) buoyancy gradients.

* `convective_νz`: Vertical viscosity in regions with negative (unstable) buoyancy gradients. Either
                  a number, function, array, or field.

* `background_κz`: Vertical viscosity in regions with zero or positive (stable) buoyancy gradients.

* `time_discretization`: Either `Explicit` or `VerticallyImplicit`.
"""

ConvectiveAdjustmentVerticalDiffusivity(FT::DataType; kwargs...) = ConvectiveAdjustmentVerticalDiffusivity(VerticallyImplicitTimeDiscretization(), FT; kwargs...)

function ConvectiveAdjustmentVerticalDiffusivity(time_discretization = VerticallyImplicitTimeDiscretization(), FT = Float64;
                                                 convective_κz = zero(FT),
                                                 convective_νz = zero(FT),
                                                 background_κz = zero(FT),
                                                 background_νz = zero(FT))

    return ConvectiveAdjustmentVerticalDiffusivity{typeof(time_discretization)}(convective_κz, convective_νz,
                                                                                background_κz, background_νz)
end

const CAVD = ConvectiveAdjustmentVerticalDiffusivity

#####
##### Diffusivity field utilities
#####

# Support for "ManyIndependentColumnMode"
const CAVDArray = AbstractArray{<:CAVD}

with_tracers(tracers, closure::CAVD{TD}) where TD =
    ConvectiveAdjustmentVerticalDiffusivity{TD}(closure.convective_κz,
                                                closure.convective_νz,
                                                closure.background_κz,
                                                closure.background_νz)

function with_tracers(tracers, closure_array::CAVDArray)
    arch = architecture(closure_array)
    Ex, Ey = size(closure_array)
    return arch_array(arch, [with_tracers(tracers, closure_array[i, j]) for i=1:Ex, j=1:Ey])
end

# Note: computing diffusivities at cell centers for now.
function DiffusivityFields(grid, tracer_names, bcs, closure::Union{CAVD, CAVDArray})
    ## If we can get away with only precomputing the "stability" of a cell:
    # data = new_data(Bool, arch, grid, (Center, Center, Center))
    κ = CenterField(grid)
    ν = CenterField(grid)
    return (; κ, ν)
end       

function calculate_diffusivities!(diffusivities, closure::Union{CAVD, CAVDArray}, model)

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
    closure_ij = get_closure_ij(i, j, closure)

    stable_cell = is_stableᶜᶜᶠ(i, j, k+1, grid, tracers, buoyancy) & 
                  is_stableᶜᶜᶠ(i, j, k,   grid, tracers, buoyancy)

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

@inline z_viscosity(::Union{CAVD, CAVDArray}, diffusivities, args...) = diffusivities.ν
@inline z_diffusivity(::Union{CAVD, CAVDArray}, ::Val{tracer_index}, diffusivities, args...) where tracer_index =
    diffusivities.κ[tracer_index]

#####
##### Show
#####

function Base.summary(closure::ConvectiveAdjustmentVerticalDiffusivity)
    TD = nameof(typeof(time_discretization(closure)))
    return string("ConvectiveAdjustmentVerticalDiffusivity{$TD}(",
                  "background_κz=$(closure.background_κz), convective_κz=$(closure.convective_κz), ",
                  "background_νz=$(closure.background_νz), convective_νz=$(closure.convective_νz))")
end

Base.show(io::IO, closure::ConvectiveAdjustmentVerticalDiffusivity) = print(io, summary(closure))
