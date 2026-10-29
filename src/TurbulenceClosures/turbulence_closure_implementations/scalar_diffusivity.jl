using Oceananigans.Utils: prettysummary

"""
    SecondOrderDiffusion()

Second-order (three-point Laplacian) discretization of diffusive fluxes. Default for `ScalarDiffusivity`.
"""
struct SecondOrderDiffusion end

"""
    FourthOrderDiffusion()

Fourth-order (five-point Laplacian) discretization of diffusive fluxes, with second-order fallback
next to walls and immersed boundaries. See [`ScalarDiffusivity`](@ref).
"""
struct FourthOrderDiffusion end

diffusion_discretization(order::Int) = order == 2 ? SecondOrderDiffusion() :
                                       order == 4 ? FourthOrderDiffusion() :
                                       throw(ArgumentError("ScalarDiffusivity order must be 2 or 4; got order = $order"))

@inline diffusion_order(::SecondOrderDiffusion) = Val(2)
@inline diffusion_order(::FourthOrderDiffusion) = Val(4)

discretization_halo_size(::SecondOrderDiffusion) = 1
discretization_halo_size(::FourthOrderDiffusion) = 2

struct ScalarDiffusivity{TD, F, N, V, K, D} <: AbstractScalarDiffusivity{TD, F, N}
    ν :: V
    κ :: K
    discretization :: D
    ScalarDiffusivity{TD, F, N}(ν::V, κ::K, discretization::D=SecondOrderDiffusion()) where {TD, F, N, V, K, D} =
        new{TD, F, N, V, K, D}(ν, κ, discretization)
end

@inline diffusion_order(closure::ScalarDiffusivity) = diffusion_order(closure.discretization)

"""
    ScalarDiffusivity(time_discretization = ExplicitTimeDiscretization(),
                      formulation = ThreeDimensionalFormulation(), FT = Float64;
                      ν = nothing,
                      κ = nothing,
                      discrete_form = false,
                      loc = (nothing, nothing, nothing),
                      parameters = nothing,
                      order = 2)

Return `ScalarDiffusivity` turbulence closure with viscosity `ν` and tracer diffusivities `κ`
for each tracer field in `tracers`. If a single `κ` is provided, it is applied to all tracers.
Otherwise `κ` must be a `NamedTuple` with values for every tracer individually.

Arguments
=========

* `time_discretization`: either `ExplicitTimeDiscretization()` (default)
  or `VerticallyImplicitTimeDiscretization()`.

* `formulation`:
  - `HorizontalFormulation()` for diffusivity applied in the horizontal direction(s)
  - `VerticalFormulation()` for diffusivity applied in the vertical direction,
  - `ThreeDimensionalFormulation()` (default) for diffusivity applied isotropically to all directions

* `FT`: the float datatype (default: `Float64`)

Keyword arguments
=================

* `ν`: Viscosity. `Number`, `AbstractArray`, `Field`, `Function`, or `nothing`.
       With `ν = nothing` the closure does not act on momentum and no viscous fluxes are computed.

* `κ`: Diffusivity. `Number`, `AbstractArray`, `Field`, `Function`, `nothing`, or
       `NamedTuple` of diffusivities with entries for each tracer.
       With `κ = nothing` the closure does not act on tracers and no diffusive fluxes are computed.

* `discrete_form`: `Boolean`; default: `false`.

When prescribing the viscosities or diffusivities as functions, depending on the
value of keyword argument `discrete_form`, the constructor expects:

* `discrete_form = false` (default): functions of the grid's native coordinates
  and time, e.g., `(x, y, z, t)` for a `RectilinearGrid` or `(λ, φ, z, t)` for
  a `LatitudeLongitudeGrid`.

* `discrete_form = true`:
  - with `loc = (nothing, nothing, nothing)` and `parameters = nothing` (default):
    functions of `(i, j, k, grid, ℓx, ℓy, ℓz, clock, fields)` with `ℓx`, `ℓy`,
    and `ℓz` either `Face()` or `Center()`.
  - with `loc = (ℓx, ℓy, ℓz)` with `ℓx`, `ℓy`, and `ℓz` either
    `Face()` or `Center()` and `parameters = nothing`: functions of `(i, j, k, grid, clock, fields)`.
  - with `loc = (nothing, nothing, nothing)` and specified `parameters`:
    functions of `(i, j, k, grid, ℓx, ℓy, ℓz, clock, fields, parameters)`.
  - with `loc = (ℓx, ℓy, ℓz)` and specified `parameters`:
    functions of `(i, j, k, grid, clock, fields, parameters)`.

* `required_halo_size = 1`: the required halo size for the closure. This value should be an integer.
  change only if using a function for `ν` or `κ` that requires a halo size larger than 1 to compute.

* `parameters`: `NamedTuple` with parameters used by the functions
  that compute viscosity and/or diffusivity; default: `nothing`.

* `order`: order of accuracy of the discrete Laplacian, either `2` (default) or `4`.
  With `order = 4`, diffusive fluxes use the face derivative
  `(15 (φᵢ - φᵢ₋₁) - (φᵢ₊₁ - φᵢ₋₂)) / 12Δ`, whose flux divergence is the fourth-order
  five-point Laplacian. Fourth-order fluxes are used only in directions in which the grid is a
  regularly-spaced `RectilinearGrid`, and fall back to second order next to `Bounded` walls and
  immersed boundaries (where the four-point stencil would include inactive nodes), and in directions
  with variable spacing. Viscous fluxes compute `∂ⱼuᵢ` at fourth order and the transpose term `∂ᵢuⱼ`
  of the strain rate at second order, so that the viscous term reduces to the fourth-order Laplacian
  `ν ∇²uᵢ` for constant `ν` and a velocity field that is discretely divergence-free.
  Diffusivities stored in arrays or `Field`s are interpolated to faces at second order.
  `order = 4` requires a halo size of at least 2 and is supported for `ThreeDimensionalFormulation`
  and `VerticalFormulation` with `ExplicitTimeDiscretization`.

Examples
========

```jldoctest ScalarDiffusivity
julia> using Oceananigans

julia> ScalarDiffusivity(ν=1000, κ=2000)
ScalarDiffusivity{ExplicitTimeDiscretization}(ν=1000.0, κ=2000.0)
```

```jldoctest ScalarDiffusivity
julia> ScalarDiffusivity(ν=1e-6, κ=1e-7, order=4)
ScalarDiffusivity{ExplicitTimeDiscretization}(ν=1.0e-6, κ=1.0e-7, order=4)
```

```jldoctest ScalarDiffusivity
julia> const depth_scale = 100;

julia> @inline ν(x, y, z, t) = 1000 * exp(z / depth_scale)
ν (generic function with 1 method)

julia> ScalarDiffusivity(ν=ν)
ScalarDiffusivity{ExplicitTimeDiscretization}(ν=ν (generic function with 1 method), κ=nothing)
```

```jldoctest ScalarDiffusivity
julia> using Oceananigans.Grids: znode

julia> @inline function κ(i, j, k, grid, ℓx, ℓy, ℓz, clock, fields)
           z = znode(i, j, k, grid, ℓx, ℓy, ℓz)
           return 2000 * exp(z / depth_scale)
       end
κ (generic function with 1 method)

julia> ScalarDiffusivity(κ=κ, discrete_form=true)
ScalarDiffusivity{ExplicitTimeDiscretization}(ν=nothing, κ=Oceananigans.TurbulenceClosures.DiscreteDiffusionFunction{Nothing, Nothing, Nothing, Nothing, typeof(κ)})
```

```jldoctest ScalarDiffusivity
julia> @inline function another_κ(i, j, k, grid, clock, fields, p)
           z = znode(i, j, k, grid, Center(), Center(), Face())
           return 2000 * exp(z / p.depth_scale)
       end
another_κ (generic function with 1 method)

julia> ScalarDiffusivity(κ=another_κ, discrete_form=true, loc=(Center, Center, Face), parameters=(; depth_scale = 120.0))
ScalarDiffusivity{ExplicitTimeDiscretization}(ν=nothing, κ=Oceananigans.TurbulenceClosures.DiscreteDiffusionFunction{Center, Center, Face, @NamedTuple{depth_scale::Float64}, typeof(another_κ)})
```
"""
function ScalarDiffusivity(time_discretization=ExplicitTimeDiscretization(),
                           formulation=ThreeDimensionalFormulation(),
                           FT=Oceananigans.defaults.FloatType;
                           ν=nothing, κ=nothing,
                           discrete_form = false,
                           loc = (nothing, nothing, nothing),
                           parameters = nothing,
                           required_halo_size::Int = 1,
                           order::Int = 2)

    if formulation == HorizontalFormulation() && time_discretization == VerticallyImplicitTimeDiscretization()
      throw(ArgumentError("VerticallyImplicitTimeDiscretization is only supported for \
          `VerticalFormulation` or `ThreeDimensionalFormulation`"))
    end

    discretization = diffusion_discretization(order)

    if discretization isa FourthOrderDiffusion
        formulation isa Union{ThreeDimensionalFormulation, VerticalFormulation} ||
            throw(ArgumentError("order = 4 is only supported for ThreeDimensionalFormulation and VerticalFormulation"))

        time_discretization isa ExplicitTimeDiscretization ||
            throw(ArgumentError("order = 4 is only supported with ExplicitTimeDiscretization"))

    end

    required_halo_size = max(required_halo_size, discretization_halo_size(discretization))

    κ = convert_diffusivity(FT, κ; discrete_form, loc, parameters)
    ν = convert_diffusivity(FT, ν; discrete_form, loc, parameters)

    # Force a type-stable constructor if ν and κ are numbers
    # This particular short-circuiting of the required_halo_size kwargs is necessary to perform parameter
    # estimation of the diffusivity coefficients using autodiff.
    if ν isa Number && κ isa Number
      N = discretization_halo_size(discretization)
      return ScalarDiffusivity{typeof(time_discretization), typeof(formulation), N}(ν, κ, discretization)
    end

    return ScalarDiffusivity{typeof(time_discretization), typeof(formulation), required_halo_size}(ν, κ, discretization)
end

# Explicit default
@inline ScalarDiffusivity(formulation::AbstractDiffusivityFormulation, FT=Oceananigans.defaults.FloatType; kw...) =
    ScalarDiffusivity(ExplicitTimeDiscretization(), formulation, FT; kw...)

const VerticalScalarDiffusivity{TD} = ScalarDiffusivity{TD, VerticalFormulation} where TD
const HorizontalScalarDiffusivity{TD} = ScalarDiffusivity{TD, HorizontalFormulation} where TD
const HorizontalDivergenceScalarDiffusivity{TD} = ScalarDiffusivity{TD, HorizontalDivergenceFormulation} where TD

"""
    VerticalScalarDiffusivity([time_discretization=ExplicitTimeDiscretization(),
                              FT::DataType=Float64;]
                              kwargs...)

Shorthand for a `ScalarDiffusivity` with `VerticalFormulation()`. See [`ScalarDiffusivity`](@ref).
"""
@inline VerticalScalarDiffusivity(time_discretization=ExplicitTimeDiscretization(),
                                  FT::DataType=Oceananigans.defaults.FloatType; kwargs...) =
    ScalarDiffusivity(time_discretization, VerticalFormulation(), FT; kwargs...)

"""
    HorizontalScalarDiffusivity([time_discretization=ExplicitTimeDiscretization(),
                                FT::DataType=Float64;]
                                kwargs...)

Shorthand for a `ScalarDiffusivity` with `HorizontalFormulation()`. See [`ScalarDiffusivity`](@ref).
"""
@inline HorizontalScalarDiffusivity(time_discretization=ExplicitTimeDiscretization(),
                                    FT::DataType=Oceananigans.defaults.FloatType; kwargs...) =
    ScalarDiffusivity(time_discretization, HorizontalFormulation(), FT; kwargs...)

"""
    HorizontalDivergenceScalarDiffusivity([time_discretization=ExplicitTimeDiscretization(),
                                          FT::DataType=Float64;]
                                          kwargs...)

Shorthand for a `ScalarDiffusivity` with `HorizontalDivergenceFormulation()`. See [`ScalarDiffusivity`](@ref).
"""
@inline HorizontalDivergenceScalarDiffusivity(time_discretization=ExplicitTimeDiscretization(),
                                              FT::DataType=Oceananigans.defaults.FloatType; kwargs...) =
    ScalarDiffusivity(time_discretization, HorizontalDivergenceFormulation(), FT; kwargs...)

# Aliases that allow specify the floating type, assuming that the discretization is Explicit in time
                    ScalarDiffusivity(FT::DataType; kwargs...) = ScalarDiffusivity(ExplicitTimeDiscretization(), ThreeDimensionalFormulation(), FT; kwargs...)
    @inline VerticalScalarDiffusivity(FT::DataType; kwargs...) = ScalarDiffusivity(ExplicitTimeDiscretization(), VerticalFormulation(), FT; kwargs...)
          HorizontalScalarDiffusivity(FT::DataType; kwargs...) = ScalarDiffusivity(ExplicitTimeDiscretization(), HorizontalFormulation(), FT; kwargs...)
HorizontalDivergenceScalarDiffusivity(FT::DataType; kwargs...) = ScalarDiffusivity(ExplicitTimeDiscretization(), HorizontalDivergenceFormulation(), FT; kwargs...)

@inline function Utils.with_tracers(tracers, closure::ScalarDiffusivity{TD, F, N}) where {TD, F, N}
    κ = tracer_diffusivities(tracers, closure.κ)
    return ScalarDiffusivity{TD, F, N}(closure.ν, κ, closure.discretization)
end

@inline viscosity(closure::ScalarDiffusivity, K) = closure.ν
@inline diffusivity(closure::ScalarDiffusivity, K, ::Val{id}) where id = closure.κ[id]

compute_closure_fields!(closure_fields, ::ScalarDiffusivity, args...) = nothing

function Base.summary(closure::ScalarDiffusivity)
    TD = summary(TimeSteppers.time_discretization(closure))
    prefix = replace(summary(formulation(closure)), "Formulation" => "")
    prefix === "ThreeDimensional" && (prefix = "")

    if closure.κ == NamedTuple()
        summary_str = string(prefix, "ScalarDiffusivity{$TD}(ν=", prettysummary(closure.ν), ")")
    else
        summary_str = string(prefix, "ScalarDiffusivity{$TD}(ν=", prettysummary(closure.ν), ", κ=", prettysummary(closure.κ), ")")
    end

    if diffusion_order(closure) == Val(4)
        summary_str = summary_str[1:end-1] * ", order=4)"
    end

    return summary_str
end

Base.show(io::IO, closure::ScalarDiffusivity) = print(io, summary(closure))

function Adapt.adapt_structure(to, closure::ScalarDiffusivity{TD, F, N}) where {TD, F, N}
    ν = Adapt.adapt(to, closure.ν)
    κ = Adapt.adapt(to, closure.κ)
    return ScalarDiffusivity{TD, F, N}(ν, κ, closure.discretization)
end

function Architectures.on_architecture(to, closure::ScalarDiffusivity{TD, F, N}) where {TD, F, N}
    ν = on_architecture(to, closure.ν)
    κ = on_architecture(to, closure.κ)
    return ScalarDiffusivity{TD, F, N}(ν, κ, closure.discretization)
end

#####
##### No viscosity or no tracer diffusivity
#####

const NoViscosityID = ScalarDiffusivity{<:Any, <:ThreeDimensionalFormulation,     <:Any, Nothing}
const NoViscosityHD = ScalarDiffusivity{<:Any, <:HorizontalFormulation,           <:Any, Nothing}
const NoViscosityDD = ScalarDiffusivity{<:Any, <:HorizontalDivergenceFormulation, <:Any, Nothing}
const NoViscosityVD = ScalarDiffusivity{<:Any, <:VerticalFormulation,             <:Any, Nothing}

const NoTracerDiffusivityID = ScalarDiffusivity{<:Any, <:ThreeDimensionalFormulation, <:Any, <:Any, Nothing}
const NoTracerDiffusivityHD = ScalarDiffusivity{<:Any, <:HorizontalFormulation,       <:Any, <:Any, Nothing}
const NoTracerDiffusivityVD = ScalarDiffusivity{<:Any, <:VerticalFormulation,         <:Any, <:Any, Nothing}

for flux in (:ux, :vx, :wx, :uy, :vy, :wy, :uz, :vz, :wz)
    viscous_flux = Symbol(:viscous_flux_, flux)
    @eval @inline $viscous_flux(i, j, k, grid, ::NoViscosityID, K, clk, fields, b) = zero(grid)
end

for flux in (:ux, :vx, :wx, :uy, :vy, :wy)
    viscous_flux = Symbol(:viscous_flux_, flux)
    @eval @inline $viscous_flux(i, j, k, grid, ::NoViscosityHD, K, clk, fields, b) = zero(grid)
end

for flux in (:ux, :vy)
    viscous_flux = Symbol(:viscous_flux_, flux)
    @eval @inline $viscous_flux(i, j, k, grid, ::NoViscosityDD, K, clk, fields, b) = zero(grid)
end

for flux in (:uz, :vz, :wz)
    viscous_flux = Symbol(:viscous_flux_, flux)
    @eval @inline $viscous_flux(i, j, k, grid, ::NoViscosityVD, K, clk, fields, b) = zero(grid)
end

@inline diffusive_flux_x(i, j, k, grid, ::NoTracerDiffusivityID, K, id, c, clk, fields, b) = zero(grid)
@inline diffusive_flux_y(i, j, k, grid, ::NoTracerDiffusivityID, K, id, c, clk, fields, b) = zero(grid)
@inline diffusive_flux_z(i, j, k, grid, ::NoTracerDiffusivityID, K, id, c, clk, fields, b) = zero(grid)
@inline diffusive_flux_x(i, j, k, grid, ::NoTracerDiffusivityHD, K, id, c, clk, fields, b) = zero(grid)
@inline diffusive_flux_y(i, j, k, grid, ::NoTracerDiffusivityHD, K, id, c, clk, fields, b) = zero(grid)
@inline diffusive_flux_z(i, j, k, grid, ::NoTracerDiffusivityVD, K, id, c, clk, fields, b) = zero(grid)

@inline diffusivity(::ScalarDiffusivity{<:Any, <:Any, <:Any, <:Any, Nothing}, K, ::Val) = nothing
