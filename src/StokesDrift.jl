module StokesDrift

export
    UniformStokesDrift,
    ∂t_uˢ,
    ∂t_vˢ,
    ∂t_wˢ,
    x_curl_Uˢ_cross_U,
    y_curl_Uˢ_cross_U,
    z_curl_Uˢ_cross_U

using Oceananigans.Grids: AbstractGrid
using Oceananigans.Fields
using Oceananigans.Operators
using Oceananigans.Utils: prettysummary

"""
    abstract type AbstractStokesDrift end

Parent type for parameter structs for Stokes drift fields
associated with surface waves.
"""
abstract type AbstractStokesDrift end

#####
##### Functions for "no surface waves"
#####

@inline ∂t_uˢ(i, j, k, grid::AbstractGrid{FT}, ::Nothing, time) where FT = zero(FT)
@inline ∂t_vˢ(i, j, k, grid::AbstractGrid{FT}, ::Nothing, time) where FT = zero(FT)
@inline ∂t_wˢ(i, j, k, grid::AbstractGrid{FT}, ::Nothing, time) where FT = zero(FT)

@inline x_curl_Uˢ_cross_U(i, j, k, grid::AbstractGrid{FT}, ::Nothing, U, time) where FT = zero(FT)
@inline y_curl_Uˢ_cross_U(i, j, k, grid::AbstractGrid{FT}, ::Nothing, U, time) where FT = zero(FT)
@inline z_curl_Uˢ_cross_U(i, j, k, grid::AbstractGrid{FT}, ::Nothing, U, time) where FT = zero(FT)

#####
##### Uniform Stokes drift for homogeneous surface waves
#####

"""
    UniformStokesDrift{UZ, VZ, UT, VT} <: AbstractStokesDrift

Parameter struct for Stokes drift fields associated with surface waves.
"""
struct UniformStokesDrift{UZ, VZ, UT, VT} <: AbstractStokesDrift
    ∂z_uˢ :: UZ
    ∂z_vˢ :: VZ
    ∂t_uˢ :: UT
    ∂t_vˢ :: VT
end

"""
    UniformStokesDrift(grid=nothing; kw...)

Construct four `Field`s that describe the Stokes drift shear
and tendency beneath a uniform surface gravity wave field.

The keyword arguments are the four `Field`s with default values:

* `∂z_uˢ = Field{Nothing, Nothing, Face}(grid)`: Stokes shear in x-direction
* `∂z_vˢ = Field{Nothing, Nothing, Face}(grid)`: Stokes shear in y-direction
* `∂t_uˢ = Field{Nothing, Nothing, Center}(grid)`: Stokes tendency in x-direction
* `∂t_vˢ = Field{Nothing, Nothing, Center}(grid)`: Stokes tendency in y-direction

Memory allocation for any of these fields is avoided by setting them to `nothing`.
*Tip*: if no `Field`s are required, omit `grid` from the constructor.

Notes
=====

* If the Stokes drift changes in time, the Stokes shear and tendency must be
  updated by adding a `Callback` to `Simulation.callbacks`.

* On that note, [time-dependent Stokes drift is a _source of momentum_ (Wagner et al. 2021).
  Take care that the total momentum flux into your simulation
  (boundary conditions + Stokes drift + internal forcing) is accurately specified!

Examples
========

Construct Stokes drift from a function:

```jldoctest stokes_drift
a = 1.0 # m
k = 2π / 200 # m
g = Oceananigans.Buoyancy.g_Earth
@inline ∂z_uˢ(z, t) = 2 * (a * k)^2 * sqrt(g * k) * exp(2k * z)

stokes_drift = UniformStokesDrift(∂z_uˢ=∂z_uˢ)

# output
```

Construct `UniformStokesDrift` with `Field` shear and tendency:

```jldoctest stokes_drift
using Oceananigans

grid = RectilinearGrid(size=(3, 3, 3), extent=(3, 3, 3))
stokes_drift = UniformStokesDrift(grid; ∂z_vˢ=nothing, ∂t_uˢ=nothing, ∂t_vˢ=nothing)

# output
UniformStokesDrift:
├── ∂z_uˢ: 1×1×4 Field{Nothing, Nothing, Face} reduced over dims = (1, 2) on RectilinearGrid on CPU
├── ∂z_vˢ: Nothing
├── ∂t_uˢ: Nothing
└── ∂t_vˢ: Nothing
```

Construct `UniformStokesDrift`, setting y-shear and tendencies to `nothing`:

```jldoctest stokes_drift
stokes_drift = UniformStokesDrift(grid; ∂z_vˢ=nothing, ∂t_uˢ=nothing, ∂t_vˢ=nothing)

# output
UniformStokesDrift:
├── ∂z_uˢ: 1×1×4 Field{Nothing, Nothing, Face} reduced over dims = (1, 2) on RectilinearGrid on CPU
├── ∂z_vˢ: Nothing
├── ∂t_uˢ: Nothing
└── ∂t_vˢ: Nothing
```
"""
function UniformStokesDrift(grid::AbstractGrid;
                            ∂z_uˢ = Field{Nothing, Nothing, Face}(grid),
                            ∂z_vˢ = Field{Nothing, Nothing, Face}(grid),
                            ∂t_uˢ = Field{Nothing, Nothing, Center}(grid),
                            ∂t_vˢ = Field{Nothing, Nothing, Center}(grid))

    return UniformStokesDrift(∂z_uˢ, ∂z_vˢ, ∂t_uˢ, ∂t_vˢ)
end

UniformStokesDrift(; ∂z_uˢ=nothing, ∂z_vˢ=nothing, ∂t_uˢ=nothing, ∂t_vˢ=nothing) =
    UniformStokesDrift(∂z_uˢ, ∂z_vˢ, ∂t_uˢ, ∂t_vˢ)

const USD = UniformStokesDrift

# Some helpers for three cases: Nothing, AbstractArray, or fallback (function)
@inline ∂z_Uᵃᵃᶜ(i, j, k, grid, sd::USD, ∂z_Uˢ, time) = ∂z_Uˢ(znode(Center(), k, grid), time)
@inline ∂z_Uᵃᵃᶜ(i, j, k, grid, sd::USD, ∂z_Uˢ::AbstractArray, time) = ℑzᵃᵃᶜ(i, j, k, grid, ∂z_Uˢ)
@inline ∂z_Uᵃᵃᶜ(i, j, k, grid, sd::USD, ::Nothing, time) = zero(eltype(grid))

@inline ∂z_Uᵃᵃᶠ(i, j, k, grid, sd::USD, ∂z_Uˢ, time) = ∂z_Uˢ(znode(Face(), k, grid), time)
@inline ∂z_Uᵃᵃᶠ(i, j, k, grid, sd::USD, ∂z_Uˢ::AbstractArray, time) = @inbounds ∂z_Uˢ[i, j, k]
@inline ∂z_Uᵃᵃᶠ(i, j, k, grid, sd::USD, ::Nothing, time) = zero(eltype(grid))

@inline ∂t_U(i, j, k, grid, sd::USD, ∂t_Uˢ, time) = ∂t_Uˢ(znode(Center(), k, grid), time)
@inline ∂t_U(i, j, k, grid, sd::USD, ∂t_Uˢ::AbstractArray, time) = @inbounds ∂t_Uˢ[i, j, k]
@inline ∂t_U(i, j, k, grid, sd::USD, ::Nothing, time) = zero(eltype(grid))

# Kernel functions
@inline ∂t_uˢ(i, j, k, grid, sd::USD, time) = ∂t_U(i, j, k, grid, sd, sd.∂t_uˢ, time)
@inline ∂t_vˢ(i, j, k, grid, sd::USD, time) = ∂t_U(i, j, k, grid, sd, sd.∂t_vˢ, time)
@inline ∂t_wˢ(i, j, k, grid::AbstractGrid{FT}, sd::USD, time) where FT = zero(FT)

@inline x_curl_Uˢ_cross_U(i, j, k, grid, sd::USD, U, time) =
    ℑxzᶠᵃᶜ(i, j, k, grid, U.w) * ∂z_Uᵃᵃᶜ(i, j, k, grid, sd, sd.∂z_uˢ, time)

@inline y_curl_Uˢ_cross_U(i, j, k, grid, sd::USD, U, time) =
    ℑyzᵃᶠᶜ(i, j, k, grid, U.w) * ∂z_Uᵃᵃᶜ(i, j, k, grid, sd, sd.∂z_vˢ, time)

@inline z_curl_Uˢ_cross_U(i, j, k, grid, sd::USD, U, time) = (
    - ℑxzᶜᵃᶠ(i, j, k, grid, U.u) * ∂z_Uᵃᵃᶠ(i, j, k, grid, sd, sd.∂z_uˢ, time)
    - ℑyzᵃᶜᶠ(i, j, k, grid, U.v) * ∂z_Uᵃᵃᶠ(i, j, k, grid, sd, sd.∂z_vˢ, time))

Base.show(io::IO, stokes_drift::USD) =
    print(io, "UniformStokesDrift:", '\n',
              "├── ∂z_uˢ: ", prettysummary(stokes_drift.∂z_uˢ), '\n',
              "├── ∂z_vˢ: ", prettysummary(stokes_drift.∂z_vˢ), '\n',
              "├── ∂t_uˢ: ", prettysummary(stokes_drift.∂t_uˢ), '\n',
              "└── ∂t_vˢ: ", prettysummary(stokes_drift.∂t_vˢ))

end # module

