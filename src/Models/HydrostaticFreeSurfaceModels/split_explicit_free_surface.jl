using Oceananigans
using Oceananigans.Architectures
using Oceananigans.Fields
using Oceananigans.Grids
using Oceananigans.Grids: AbstractGrid
using Oceananigans.AbstractOperations: Δz, GridMetricOperation

using Adapt
using Base
using KernelAbstractions: @index, @kernel

import Oceananigans.TimeSteppers: reset!

"""
    struct SplitExplicitFreeSurface

The split-explicit free surface solver.

$(FIELDS)
"""
struct SplitExplicitFreeSurface{𝒩, 𝒮, ℱ, 𝒫 ,ℰ} <: AbstractFreeSurface{𝒩, 𝒫}
    "The instantaneous free surface (`ReducedField`)"
    η :: 𝒩
    "The entire state for the split-explicit solver (`SplitExplicitState`)"
    state :: 𝒮
    "Parameters for timestepping split-explicit solver (`NamedTuple`)"
    auxiliary :: ℱ
    "Gravitational acceleration"
    gravitational_acceleration :: 𝒫
    "Settings for the split-explicit scheme (`NamedTuple`)"
    settings :: ℰ
end

"""
    SplitExplicitFreeSurface(; gravitational_acceleration = g_Earth, kwargs...) 

Return a `SplitExplicitFreeSurface` representing an explicit time discretization
of oceanic free surface dynamics with `gravitational_acceleration`.

Keyword Arguments
=================

- `substeps`: The number of substeps that divide the range `(t, t + 2Δt)`, where `Δt` is the baroclinic
              timestep. Note that some averaging functions do not require substepping until `2Δt`.
              The number of substeps is reduced automatically to the last index of `averaging_weights`
              for which `averaging_weights > 0`.

- `cfl`: If set then the number of `substeps` are computed based on the advective timescale imposed from the
         barotropic gravity-wave speed, computed with depth `grid.Lz`. If `fixed_Δt` is provided then the number of
         `substeps` will adapt to maintain an exact cfl. If not the effective cfl will be always lower than the 
         specified `cfl` provided that the baroclinic time step `Δt_baroclinic < fixed_Δt`

!!! info "Needed keyword arguments"
    Either `substeps` _or_ `cfl` (with `grid`) need to be prescribed.

- `grid`: Used to compute the corresponding barotropic surface wave speed.

- `fixed_Δt`: The maximum baroclinic timestep allowed. If `fixed_Δt` is a `nothing` and a cfl is provided, then
              the number of substeps will be computed on the fly from the baroclinic time step to maintain a constant cfl.

- `gravitational_acceleration`: the gravitational acceleration (default: `g_Earth`)

- `averaging_kernel`: function of `τ` used to average the barotropic transport `U` and free surface `η`
                      within the barotropic advancement. `τ` is the fractional substep going from 0 to 2
                      with the baroclinic time step `t + Δt` located at `τ = 1`. This function should be
                      centered at `τ = 1`, that is, ``∑ (aₘ m /M) = 1``. By default the averaging kernel
                      described by [Shchepetkin2005](@citet) is chosen.

- `timestepper`: Time stepping scheme used for the barotropic advancement. Choose one of:
<<<<<<< HEAD
  - `ForwardBackwardScheme()` (default): `ηᵐ⁺¹ = f(U)`   then `U = f(η)`,
  - `AdamsBashforth3Scheme()`: `η = f(U, Uᵐ⁻¹, Uᵐ⁻²)` then `U = f(η, ηᵐ, ηᵐ⁻¹, ηᵐ⁻²)`.
=======
  * `ForwardBackwardScheme()` (default): `η = f(U)`   then `U = f(η)`,
  * `AdamsBashforth3Scheme()`: `η = f(U, Uᵐ⁻¹, Uᵐ⁻²)` then `U = f(η, ηᵐ, ηᵐ⁻¹, ηᵐ⁻²)`.

References
==========

Shchepetkin, A. F., & McWilliams, J. C. (2005). The regional oceanic modeling system (ROMS): a split-explicit, free-surface, topography-following-coordinate oceanic model. Ocean Modelling, 9(4), 347-404.
>>>>>>> origin/ss-glw/time-bcs
"""
SplitExplicitFreeSurface(FT::DataType = Float64; gravitational_acceleration = g_Earth, kwargs...) = 
    SplitExplicitFreeSurface(nothing, nothing, nothing, convert(FT, gravitational_acceleration),
                             SplitExplicitSettings(FT; gravitational_acceleration, kwargs...))
                             
# The new constructor is defined later on after the state, settings, auxiliary have been defined
function FreeSurface(free_surface::SplitExplicitFreeSurface, velocities, grid)
    η =  FreeSurfaceDisplacementField(velocities, free_surface, grid)

    return SplitExplicitFreeSurface(η, SplitExplicitState(grid, free_surface.settings.timestepper),
                                    SplitExplicitAuxiliaryFields(grid),
                                    free_surface.gravitational_acceleration,
                                    free_surface.settings)
end

function SplitExplicitFreeSurface(grid; gravitational_acceleration = g_Earth,
    settings = SplitExplicitSettings(eltype(grid); gravitational_acceleration, substeps = 200))

    Nz = size(grid, 3)
    η  = ZFaceField(grid, indices = (:, :, Nz+1))
    gravitational_acceleration = convert(eltype(grid), gravitational_acceleration)

<<<<<<< HEAD
    return SplitExplicitFreeSurface(η, SplitExplicitState(grid, settings.timestepper), SplitExplicitAuxiliaryFields(grid),
           gravitational_acceleration, settings)
=======
    return SplitExplicitFreeSurface(η, SplitExplicitState(grid), SplitExplicitAuxiliaryFields(grid),
                                    gravitational_acceleration, settings)
>>>>>>> origin/ss-glw/time-bcs
end

"""
    struct SplitExplicitState

A type containing the state fields for the split-explicit free surface.

$(FIELDS)
"""
Base.@kwdef struct SplitExplicitState{CC, ACC, FC, AFC, CF, ACF}
    "The free surface at time `m`. (`ReducedField` over ``z``)"
    ηᵐ   :: ACC
    "The free surface at time `m-1`. (`ReducedField` over ``z``)"
    ηᵐ⁻¹ :: ACC
    "The free surface at time `m-2`. (`ReducedField` over ``z``)"
    ηᵐ⁻² :: ACC
    "The barotropic zonal velocity at time `m`. (`ReducedField` over ``z``)"
    U    :: FC
    "The barotropic zonal velocity at time `m-1`. (`ReducedField` over ``z``)"
    Uᵐ⁻¹ :: AFC
    "The barotropic zonal velocity at time `m-2`. (`ReducedField` over ``z``)"
    Uᵐ⁻² :: AFC
    "The barotropic meridional velocity at time `m`. (`ReducedField` over ``z``)"
    V    :: CF
    "The barotropic meridional velocity at time `m-1`. (`ReducedField` over ``z``)"
    Vᵐ⁻¹ :: ACF
    "The barotropic meridional velocity at time `m-2`. (`ReducedField` over ``z``)"
    Vᵐ⁻² :: ACF
    "The time-filtered free surface. (`ReducedField` over ``z``)"
    η̅    :: CC
    "The time-filtered barotropic zonal velocity. (`ReducedField` over ``z``)"
    U̅    :: FC
    "The time-filtered barotropic meridional velocity. (`ReducedField` over ``z``)"
    V̅    :: CF
end

"""
    SplitExplicitState(grid, timestepper)

Return the split-explicit state for `grid`.

Note that `η̅` is solely used for setting the `η` at the next substep iteration -- it essentially
acts as a filter for `η`. Values with superscripts `m-1` and `m-2` correspond to previous stored
time steps to allow using a higher-order time stepping scheme, e.g., `AdamsBashforth3Scheme`.
"""
function SplitExplicitState(grid::AbstractGrid, timestepper)
    
    Nz = size(grid, 3)
    
    η̅ = ZFaceField(grid, indices = (:, :, Nz+1))

    ηᵐ   = auxiliary_free_surface_field(grid, timestepper)
    ηᵐ⁻¹ = auxiliary_free_surface_field(grid, timestepper)
    ηᵐ⁻² = auxiliary_free_surface_field(grid, timestepper)
          
    U    = XFaceField(grid, indices = (:, :, Nz))
    V    = YFaceField(grid, indices = (:, :, Nz))

    Uᵐ⁻¹ = auxiliary_barotropic_U_field(grid, timestepper)
    Vᵐ⁻¹ = auxiliary_barotropic_V_field(grid, timestepper)
    Uᵐ⁻² = auxiliary_barotropic_U_field(grid, timestepper)
    Vᵐ⁻² = auxiliary_barotropic_V_field(grid, timestepper)
          
    U̅ = XFaceField(grid, indices = (:, :, Nz))
    V̅ = YFaceField(grid, indices = (:, :, Nz))
    
    return SplitExplicitState(; ηᵐ, ηᵐ⁻¹, ηᵐ⁻², U, Uᵐ⁻¹, Uᵐ⁻², V, Vᵐ⁻¹, Vᵐ⁻², η̅, U̅, V̅)
end

"""
    SplitExplicitAuxiliaryFields

A type containing auxiliary fields for the split-explicit free surface.

The barotropic time stepping is launched on a grid `(kernel_size[1], kernel_size[2])`
large (or `:xy` in case of a serial computation), and start computing from 
`(i - kernel_offsets[1], j - kernel_offsets[2])`.

$(FIELDS)
"""
Base.@kwdef struct SplitExplicitAuxiliaryFields{𝒞ℱ, ℱ𝒞, 𝒦}
    "Vertically-integrated slow barotropic forcing function for `U` (`ReducedField` over ``z``)"
    Gᵁ :: ℱ𝒞
    "Vertically-integrated slow barotropic forcing function for `V` (`ReducedField` over ``z``)"
    Gⱽ :: 𝒞ℱ
    "Depth at `(Face, Center)` (`ReducedField` over ``z``)"
    Hᶠᶜ :: ℱ𝒞
    "Depth at `(Center, Face)` (`ReducedField` over ``z``)"
    Hᶜᶠ :: 𝒞ℱ
    "kernel size for barotropic time stepping"
    kernel_parameters :: 𝒦
end

"""
    SplitExplicitAuxiliaryFields(grid)

Return the `SplitExplicitAuxiliaryFields` for `grid`.
"""
function SplitExplicitAuxiliaryFields(grid::AbstractGrid)

    Gᵁ = Field((Face,   Center, Nothing), grid)
    Gⱽ = Field((Center, Face,   Nothing), grid)

    Hᶠᶜ = Field((Face,   Center, Nothing), grid)
    Hᶜᶠ = Field((Center, Face,   Nothing), grid)

    dz = GridMetricOperation((Face, Center, Center), Δz, grid)
    sum!(Hᶠᶜ, dz)

    dz = GridMetricOperation((Center, Face, Center), Δz, grid)
    sum!(Hᶜᶠ, dz)

    fill_halo_regions!((Hᶠᶜ, Hᶜᶠ))

    kernel_parameters = :xy
<<<<<<< HEAD
    
    return SplitExplicitAuxiliaryFields(Gᵁ, Gⱽ, Hᶠᶜ, Hᶜᶠ, kernel_parameters)
=======

    return SplitExplicitAuxiliaryFields(Gᵁ, Gⱽ, Hᶠᶜ, Hᶜᶠ, Hᶜᶜ, kernel_parameters)
>>>>>>> origin/ss-glw/time-bcs
end

"""
    struct SplitExplicitSettings

A type containing settings for the split-explicit free surface.

$(FIELDS)
"""
struct SplitExplicitSettings{𝒩, 𝒮}
    substepping :: 𝒩 # Either `FixedSubstepNumber` or `FixedTimeStepSize`"
    timestepper :: 𝒮 # time-stepping scheme
end

struct AdamsBashforth3Scheme end
struct ForwardBackwardScheme end


auxiliary_free_surface_field(grid, ::AdamsBashforth3Scheme) = ZFaceField(grid, indices = (:, :, size(grid, 3)+1))
auxiliary_free_surface_field(grid, ::ForwardBackwardScheme) = nothing

auxiliary_barotropic_U_field(grid, ::AdamsBashforth3Scheme) = XFaceField(grid, indices = (:, :, size(grid, 3)))
auxiliary_barotropic_U_field(grid, ::ForwardBackwardScheme) = nothing
auxiliary_barotropic_V_field(grid, ::AdamsBashforth3Scheme) = YFaceField(grid, indices = (:, :, size(grid, 3)))
auxiliary_barotropic_V_field(grid, ::ForwardBackwardScheme) = nothing

# (p = 2, q = 4, r = 0.18927) minimize dispersion error from Shchepetkin and McWilliams (2005): https://doi.org/10.1016/j.ocemod.2004.08.002 
@inline function averaging_shape_function(τ::FT; p = 2, q = 4, r = FT(0.18927)) where FT 
    τ₀ = (p + 2) * (p + q + 2) / (p + 1) / (p + q + 1)

    return (τ / τ₀)^p * (1 - (τ / τ₀)^q) - r * (τ / τ₀)
end

@inline   cosine_averaging_kernel(τ::FT) where FT = τ ≥ 0.5 && τ ≤ 1.5 ? convert(FT, 1 + cos(2π * (τ - 1))) : zero(FT)
@inline constant_averaging_kernel(τ::FT) where FT = convert(FT, 1)

""" An internal type for the `SplitExplicitFreeSurface` that allows substepping with
a fixed `Δt_barotopic` based on a CFL condition """
struct FixedTimeStepSize{B, F}
    Δt_barotropic    :: B
    averaging_kernel :: F
end

""" An internal type for the `SplitExplicitFreeSurface` that allows substepping with
a fixed number of substeps with time step size of `fractional_step_size * Δt_baroclinic` """
struct FixedSubstepNumber{B, F}
    fractional_step_size :: B
    averaging_weights    :: F
end

function FixedTimeStepSize(FT::DataType = Float64;
                           cfl = 0.7, 
                           grid, 
                           averaging_kernel = averaging_shape_function, 
                           gravitational_acceleration = g_Earth)
    Δx⁻² = topology(grid)[1] == Flat ? 0 : 1 / minimum_xspacing(grid)^2
    Δy⁻² = topology(grid)[2] == Flat ? 0 : 1 / minimum_yspacing(grid)^2
    Δs   = sqrt(1 / (Δx⁻² + Δy⁻²))

    wave_speed = sqrt(gravitational_acceleration * grid.Lz)

    Δt_barotropic = convert(FT, cfl * Δs / wave_speed)

    return FixedTimeStepSize(Δt_barotropic, averaging_kernel)
end

@inline function weights_from_substeps(FT, substeps, averaging_kernel)

    τᶠ = range(FT(0), FT(2), length = substeps+1)
    Δτ = τᶠ[2] - τᶠ[1]

    averaging_weights = map(averaging_kernel, τᶠ[2:end])
    idx = searchsortedlast(averaging_weights, 0, rev=true)
    substeps = idx

    averaging_weights = averaging_weights[1:idx]
    averaging_weights ./= sum(averaging_weights)

    return Δτ, tuple(averaging_weights...)
end

function SplitExplicitSettings(FT::DataType=Float64;
                               substeps = nothing, 
                               cfl      = nothing,
                               grid     = nothing,
                               fixed_Δt = nothing,
                               gravitational_acceleration = g_Earth,
                               averaging_kernel = averaging_shape_function,
                               timestepper = ForwardBackwardScheme())
    
    if (!isnothing(substeps) && !isnothing(cfl)) || (isnothing(substeps) && isnothing(cfl))
        throw(ArgumentError("either specify a cfl or a number of substeps"))
    end

    if !isnothing(grid) && eltype(grid) !== FT
        throw(ArgumentError("Prescribed FT was different that the one used in `grid`."))
    end

    if !isnothing(cfl)
        if isnothing(grid)
            throw(ArgumentError("Need to specify the grid kwarg to calculate the barotropic substeps from the cfl"))
        end
        substepping = FixedTimeStepSize(FT; cfl, grid, gravitational_acceleration, averaging_kernel)
        if isnothing(fixed_Δt)
            return SplitExplicitSettings(substepping, timestepper)
        else
            substeps = ceil(Int, 2 * fixed_Δt / substepping.Δt_barotropic)
        end
    end

    fractional_step_size, averaging_weights = weights_from_substeps(FT, substeps, averaging_kernel)
    substepping = FixedSubstepNumber(fractional_step_size, averaging_weights)

    return SplitExplicitSettings(substepping, timestepper)
end

# Convenience Functions for grabbing free surface
free_surface(free_surface::SplitExplicitFreeSurface) = free_surface.η

# extend
@inline explicit_barotropic_pressure_x_gradient(i, j, k, grid, ::SplitExplicitFreeSurface) = zero(grid)
@inline explicit_barotropic_pressure_y_gradient(i, j, k, grid, ::SplitExplicitFreeSurface) = zero(grid)

# convenience functor
(sefs::SplitExplicitFreeSurface)(settings::SplitExplicitSettings) =
    SplitExplicitFreeSurface(sefs.η, sefs.state, sefs.auxiliary, sefs.gravitational_acceleration, settings)

Base.summary(s::FixedTimeStepSize)  = string("Barotropic time step equal to $(s.Δt_barotopic)")
Base.summary(s::FixedSubstepNumber) = string("Barotropic fractional step equal to $(s.fractional_step_size) times the baroclinic step")

Base.summary(sefs::SplitExplicitFreeSurface) = string("SplitExplicitFreeSurface with $(sefs.settings.substepping)")
Base.show(io::IO, sefs::SplitExplicitFreeSurface) = print(io, "$(summary(sefs))\n")

function reset!(sefs::SplitExplicitFreeSurface)
    for name in propertynames(sefs.state)
        var = getproperty(sefs.state, name)
        fill!(var, 0)
    end

    fill!(sefs.auxiliary.Gᵁ, 0)
    fill!(sefs.auxiliary.Gⱽ, 0)

    return nothing
end

# Adapt
Adapt.adapt_structure(to, free_surface::SplitExplicitFreeSurface) =
    SplitExplicitFreeSurface(Adapt.adapt(to, free_surface.η), nothing, nothing,
                             free_surface.gravitational_acceleration, nothing)
