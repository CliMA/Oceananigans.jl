using Adapt: Adapt, adapt
using DocStringExtensions: SIGNATURES
using KernelAbstractions: @kernel, @index
using Oceananigans.Architectures: AbstractArchitecture, CPU
using Oceananigans.Grids: Grids, Bounded, Center, Face, OrthogonalSphericalShellGrid,
                          architecture, cpu_face_constructor_z,
                          validate_topology, validate_size, validate_halo,
                          validate_dimension_specification, topology
using Oceananigans.Utils: KernelParameters, launch!, prettysummary
using OrderedCollections: OrderedDict

struct LambertConformalConic{FT}
    central_longitude :: FT
    latitude_of_origin :: FT
    standard_parallel_1 :: FT
    standard_parallel_2 :: FT
    false_easting :: FT
    false_northing :: FT
    radius :: FT
    cone_constant :: FT
    scale_constant :: FT
    origin_radius :: FT
    x₁ :: FT
    y₁ :: FT
    Δx :: FT
    Δy :: FT
end

const LambertConformalConicGrid{FT, TX, TY, TZ, Z} =
    OrthogonalSphericalShellGrid{FT, TX, TY, TZ, Z,
                                 <:LambertConformalConic} where {FT, TX, TY, TZ, Z}

@inline degrees_to_radians(::Type{FT}, θ) where FT = convert(FT, θ) * convert(FT, π) / convert(FT, 180)
@inline radians_to_degrees(::Type{FT}, θ) where FT = convert(FT, θ) * convert(FT, 180) / convert(FT, π)
@inline lcc_half(::Type{FT}) where FT = one(FT) / convert(FT, 2)
@inline lcc_halfπ(::Type{FT}) where FT = convert(FT, π) / convert(FT, 2)
@inline lcc_quarterπ(::Type{FT}) where FT = convert(FT, π) / convert(FT, 4)
@inline lcc_tangent(φ) = tan(lcc_quarterπ(typeof(φ)) + lcc_half(typeof(φ)) * φ)
@inline lcc_wrap_to_π(λ) = atan(sin(λ), cos(λ))

function normalize_standard_parallels(standard_parallel, standard_parallels)
    if !isnothing(standard_parallel) && !isnothing(standard_parallels)
        throw(ArgumentError("Specify either standard_parallel or standard_parallels, not both."))
    elseif isnothing(standard_parallel) && isnothing(standard_parallels)
        throw(ArgumentError("Must supply standard_parallel or standard_parallels."))
    elseif !isnothing(standard_parallel)
        return standard_parallel, standard_parallel
    elseif standard_parallels isa Number
        return standard_parallels, standard_parallels
    elseif standard_parallels isa Tuple
        length(standard_parallels) == 2 ||
            throw(ArgumentError("standard_parallels must be a number or a 2-tuple."))
        return standard_parallels
    else
        throw(ArgumentError("standard_parallels must be a number or a 2-tuple."))
    end
end

function validate_lcc_angles(FT, φ₀, φ₁, φ₂)
    halfπ = lcc_halfπ(FT)
    tolerance = sqrt(eps(FT))

    isfinite(φ₀) || throw(ArgumentError("latitude_of_origin must be finite."))
    isfinite(φ₁) || throw(ArgumentError("standard parallels must be finite."))
    isfinite(φ₂) || throw(ArgumentError("standard parallels must be finite."))
    abs(φ₀) <= halfπ || throw(ArgumentError("latitude_of_origin must lie between -90 and 90 degrees."))
    abs(φ₁) <= halfπ || throw(ArgumentError("standard parallels must lie between -90 and 90 degrees."))
    abs(φ₂) <= halfπ || throw(ArgumentError("standard parallels must lie between -90 and 90 degrees."))

    # When a standard parallel sits exactly at a pole the projection becomes
    # polar stereographic; in that case both parallels must coincide with the
    # same pole.
    φ₁_at_pole = abs(abs(φ₁) - halfπ) <= tolerance
    φ₂_at_pole = abs(abs(φ₂) - halfπ) <= tolerance
    if φ₁_at_pole || φ₂_at_pole
        (φ₁_at_pole && φ₂_at_pole && sign(φ₁) == sign(φ₂)) ||
            throw(ArgumentError("when a standard parallel is set to ±90° (polar " *
                                "stereographic limit), both standard parallels must " *
                                "coincide with the same pole."))
    else
        abs(φ₁ + φ₂) > tolerance ||
            throw(ArgumentError("standard parallels cannot be symmetric about the equator."))
    end

    return nothing
end

function validate_lcc_scalar(value, name, FT; positive = false)
    value = try
        convert(FT, value)
    catch
        throw(ArgumentError("$name must be convertible to $FT."))
    end

    isfinite(value) || throw(ArgumentError("$name must be finite."))
    positive && value ≤ zero(FT) && throw(ArgumentError("$name must be positive."))

    return value
end

function lcc_cone_constant(FT, φ₁, φ₂)
    tolerance = sqrt(eps(FT))

    n = if isapprox(φ₁, φ₂; atol = tolerance, rtol = zero(FT))
        sin(φ₁)
    else
        log(cos(φ₁) / cos(φ₂)) / log(lcc_tangent(φ₂) / lcc_tangent(φ₁))
    end

    abs(n) > tolerance || throw(ArgumentError("Lambert conformal cone constant n is too close to zero."))

    return n
end

"""
    $(SIGNATURES)

Return a spherical Lambert conformal conic projection map.

```jldoctest
using Oceananigans

map = LambertConformalConic(Float64;
                            standard_parallels = (30, 60),
                            central_longitude = -105,
                            latitude_of_origin = 40,
                            x₁ = -1000, y₁ = -1000,
                            Δx = 100, Δy = 100)

show(round(lcc_scale_factor(map, 30); digits=12))

# output
1.0
```
"""
function LambertConformalConic(FT::DataType = Oceananigans.defaults.FloatType;
                               standard_parallel = nothing,
                               standard_parallels = nothing,
                               central_longitude,
                               latitude_of_origin,
                               false_easting = 0,
                               false_northing = 0,
                               radius = Oceananigans.defaults.planet_radius,
                               x₁,
                               y₁,
                               Δx,
                               Δy)

    standard_parallel_1, standard_parallel_2 =
        normalize_standard_parallels(standard_parallel, standard_parallels)

    central_longitude = validate_lcc_scalar(central_longitude, "central_longitude", FT)
    latitude_of_origin = validate_lcc_scalar(latitude_of_origin, "latitude_of_origin", FT)
    standard_parallel_1 = validate_lcc_scalar(standard_parallel_1, "standard parallels", FT)
    standard_parallel_2 = validate_lcc_scalar(standard_parallel_2, "standard parallels", FT)
    false_easting = validate_lcc_scalar(false_easting, "false_easting", FT)
    false_northing = validate_lcc_scalar(false_northing, "false_northing", FT)
    radius = validate_lcc_scalar(radius, "radius", FT; positive = true)
    x₁ = validate_lcc_scalar(x₁, "x₁", FT)
    y₁ = validate_lcc_scalar(y₁, "y₁", FT)
    Δx = validate_lcc_scalar(Δx, "Δx", FT; positive = true)
    Δy = validate_lcc_scalar(Δy, "Δy", FT; positive = true)

    λ₀ = degrees_to_radians(FT, central_longitude)
    φ₀ = degrees_to_radians(FT, latitude_of_origin)
    φ₁ = degrees_to_radians(FT, standard_parallel_1)
    φ₂ = degrees_to_radians(FT, standard_parallel_2)

    validate_lcc_angles(FT, φ₀, φ₁, φ₂)

    n = lcc_cone_constant(FT, φ₁, φ₂)
    T₁ = lcc_tangent(φ₁)
    # Polar-stereographic limit: when |φ₁| = π/2 the cone degenerates to a
    # plane tangent at the pole. cos(φ₁) → 0 and T₁^n → ∞, but the product
    # cos(φ₁) · T₁^n / n approaches the finite limit 2·sign(n). Take that limit
    # directly to avoid 0·∞ = NaN.
    halfπ = lcc_halfπ(FT)
    polar_tolerance = sqrt(eps(FT))
    is_polar = abs(abs(φ₁) - halfπ) <= polar_tolerance
    F = is_polar ? convert(FT, 2) * sign(n) : cos(φ₁) * T₁^n / n
    ρ₀ = radius * F / lcc_tangent(φ₀)^n

    isfinite(F) ||
        throw(ArgumentError("Lambert conformal scale constant F is not finite."))

    isfinite(ρ₀) ||
        throw(ArgumentError("Lambert conformal radius at latitude_of_origin is not finite."))

    return LambertConformalConic{FT}(λ₀, φ₀, φ₁, φ₂,
                                     false_easting, false_northing,
                                     radius, n, F, ρ₀,
                                     x₁, y₁, Δx, Δy)
end

function Adapt.adapt_structure(to, map::LambertConformalConic)
    return LambertConformalConic(adapt(to, map.central_longitude),
                                 adapt(to, map.latitude_of_origin),
                                 adapt(to, map.standard_parallel_1),
                                 adapt(to, map.standard_parallel_2),
                                 adapt(to, map.false_easting),
                                 adapt(to, map.false_northing),
                                 adapt(to, map.radius),
                                 adapt(to, map.cone_constant),
                                 adapt(to, map.scale_constant),
                                 adapt(to, map.origin_radius),
                                 adapt(to, map.x₁),
                                 adapt(to, map.y₁),
                                 adapt(to, map.Δx),
                                 adapt(to, map.Δy))
end

on_architecture(::AbstractArchitecture, map::LambertConformalConic) = map

function Base.show(io::IO, map::LambertConformalConic{FT}) where FT
    λ₀ = prettysummary(radians_to_degrees(FT, map.central_longitude))
    φ₀ = prettysummary(radians_to_degrees(FT, map.latitude_of_origin))
    φ₁ = prettysummary(radians_to_degrees(FT, map.standard_parallel_1))
    φ₂ = prettysummary(radians_to_degrees(FT, map.standard_parallel_2))
    Δx = prettysummary(map.Δx)
    Δy = prettysummary(map.Δy)

    return print(io, "LambertConformalConic(λ₀=", λ₀, "°, φ₀=", φ₀,
                 "°, standard_parallels=(", φ₁, "°, ", φ₂, "°), Δ=(", Δx, ", ", Δy, ") m)")
end

@inline function lcc_radius(map::LambertConformalConic{FT}, φ) where FT
    F = map.scale_constant
    n = map.cone_constant
    return map.radius * F / lcc_tangent(φ)^n
end

"""
    $(SIGNATURES)

Return the projected coordinates `(x, y)` corresponding to longitude `λ` and latitude `φ`, in degrees.

```jldoctest
using Oceananigans

map = LambertConformalConic(Float64;
                            standard_parallels = (30, 60),
                            central_longitude = -105,
                            latitude_of_origin = 40,
                            x₁ = -1000, y₁ = -1000,
                            Δx = 100, Δy = 100)

x, y = lcc_forward(map, -105, 40)
show((round(x; digits=12), round(y; digits=12)))

# output
(0.0, 0.0)
```
"""
@inline function lcc_forward(map::LambertConformalConic{FT}, λ, φ) where FT
    λ = degrees_to_radians(FT, λ)
    φ = degrees_to_radians(FT, φ)
    ρ = lcc_radius(map, φ)
    n = map.cone_constant
    ρ₀ = map.origin_radius
    θ = n * lcc_wrap_to_π(λ - map.central_longitude)

    x = map.false_easting + ρ * sin(θ)
    y = map.false_northing + ρ₀ - ρ * cos(θ)

    return x, y
end

"""
    $(SIGNATURES)

Return the longitude and latitude `(λ, φ)`, in degrees, corresponding to projected coordinates `(x, y)`.

```jldoctest
using Oceananigans

map = LambertConformalConic(Float64;
                            standard_parallels = (30, 60),
                            central_longitude = -105,
                            latitude_of_origin = 40,
                            x₁ = -1000, y₁ = -1000,
                            Δx = 100, Δy = 100)

x, y = lcc_forward(map, -100, 45)
λ, φ = lcc_inverse(map, x, y)
show((round(λ; digits=10), round(φ; digits=10)))

# output
(-100.0, 45.0)
```
"""
@inline function lcc_inverse(map::LambertConformalConic{FT}, x, y) where FT
    n = map.cone_constant
    F = map.scale_constant
    ρ₀ = map.origin_radius
    x′ = convert(FT, x) - map.false_easting
    y′ = convert(FT, y) - map.false_northing
    ρ_abs = sqrt(x′^2 + (ρ₀ - y′)^2)
    sign_n = ifelse(n < zero(FT), -one(FT), one(FT))
    ρ = sign_n * ρ_abs
    ρ_for_inverse = ifelse(ρ_abs == zero(FT),
                            sign_n,
                            ρ)
    θ = atan(sign_n * x′, sign_n * (ρ₀ - y′))
    T = (map.radius * F / ρ_for_inverse)^(one(FT) / n)
    φ = convert(FT, 2) * atan(T) - lcc_halfπ(FT)
    λ = map.central_longitude + θ / n

    φ_apex = ifelse(n < zero(FT), -lcc_halfπ(FT), lcc_halfπ(FT))
    φ = ifelse(ρ_abs == zero(FT), φ_apex, φ)
    λ = ifelse(ρ_abs == zero(FT), map.central_longitude, λ)
    λ = lcc_wrap_to_π(λ)

    return radians_to_degrees(FT, λ), radians_to_degrees(FT, φ)
end

"""
    $(SIGNATURES)

Return the Lambert conformal conic scale factor at latitude `φ`, in degrees.

```jldoctest
using Oceananigans

map = LambertConformalConic(Float64;
                            standard_parallels = (30, 60),
                            central_longitude = -105,
                            latitude_of_origin = 40,
                            x₁ = -1000, y₁ = -1000,
                            Δx = 100, Δy = 100)

show(round(lcc_scale_factor(map, 60); digits=12))

# output
1.0
```
"""
@inline function lcc_scale_factor(map::LambertConformalConic{FT}, φ) where FT
    φ = degrees_to_radians(FT, φ)
    ρ = lcc_radius(map, φ)
    n = map.cone_constant
    return abs(n * ρ / (map.radius * cos(φ)))
end

@inline function lcc_xnode(i, ::Center, map::LambertConformalConic{FT}) where FT
    return map.x₁ + (convert(FT, i) - lcc_half(FT)) * map.Δx
end

@inline function lcc_xnode(i, ::Face, map::LambertConformalConic{FT}) where FT
    return map.x₁ + (convert(FT, i) - one(FT)) * map.Δx
end

@inline function lcc_ynode(j, ::Center, map::LambertConformalConic{FT}) where FT
    return map.y₁ + (convert(FT, j) - lcc_half(FT)) * map.Δy
end

@inline function lcc_ynode(j, ::Face, map::LambertConformalConic{FT}) where FT
    return map.y₁ + (convert(FT, j) - 1) * map.Δy
end

function validate_lcc_topology(topology)
    TX, TY, TZ = validate_topology(topology)
    TX === Bounded || throw(ArgumentError("LambertConformalConicGrid requires Bounded topology in x."))
    TY === Bounded || throw(ArgumentError("LambertConformalConicGrid requires Bounded topology in y."))
    return TX, TY, TZ
end

function validate_lcc_tuple(specification, name, FT; positive = false)
    x, y = try
        if specification isa Number
            value = convert(FT, specification)
            value, value
        elseif specification isa Tuple
            length(specification) == 2 || throw(ArgumentError("$name must be a number or a 2-tuple."))
            convert(FT, specification[1]), convert(FT, specification[2])
        else
            throw(ArgumentError("$name must be a number or a 2-tuple."))
        end
    catch err
        err isa ArgumentError && rethrow()
        throw(ArgumentError("$name must be a number or a 2-tuple convertible to $FT."))
    end

    isfinite(x) || throw(ArgumentError("$name entries must be finite."))
    isfinite(y) || throw(ArgumentError("$name entries must be finite."))

    if positive
        x > zero(FT) || throw(ArgumentError("$name entries must be positive."))
        y > zero(FT) || throw(ArgumentError("$name entries must be positive."))
    end

    return x, y
end

function validate_lcc_center(center, FT)
    center isa Tuple && length(center) == 2 ||
        throw(ArgumentError("center must be a 2-tuple of longitude and latitude."))

    λ, φ = try
        convert(FT, center[1]), convert(FT, center[2])
    catch
        throw(ArgumentError("center entries must be convertible to $FT."))
    end

    isfinite(λ) || throw(ArgumentError("center entries must be finite."))
    isfinite(φ) || throw(ArgumentError("center entries must be finite."))

    abs(φ) <= convert(FT, 90) ||
        throw(ArgumentError("center latitude must lie between -90 and 90 degrees."))

    return λ, φ
end

function parse_lcc_domain(FT, map, size; x, y, center, extent, spacing)
    has_x = !isnothing(x)
    has_y = !isnothing(y)
    has_center = !isnothing(center)
    has_extent = !isnothing(extent)
    has_spacing = !isnothing(spacing)

    has_x == has_y || throw(ArgumentError("LambertConformalConicGrid requires both x and y, or neither."))

    xy_mode = has_x && has_y && !has_center && !has_extent && !has_spacing
    center_extent_mode = has_center && has_extent && !has_x && !has_y && !has_spacing
    center_spacing_mode = has_center && has_spacing && !has_x && !has_y && !has_extent
    domain_modes = (xy_mode, center_extent_mode, center_spacing_mode)

    count(domain_modes) == 1 ||
        throw(ArgumentError("Specify exactly one domain mode: x/y, center/extent, or center/spacing."))

    Nx, Ny, _ = size

    if has_x
        x₁, x₂ = validate_lcc_tuple(x, "x", FT)
        y₁, y₂ = validate_lcc_tuple(y, "y", FT)
        x₂ > x₁ || throw(ArgumentError("x must be an increasing interval."))
        y₂ > y₁ || throw(ArgumentError("y must be an increasing interval."))
        Δx = (x₂ - x₁) / convert(FT, Nx)
        Δy = (y₂ - y₁) / convert(FT, Ny)
    else
        x_center, y_center = lcc_forward(map, center...)

        if has_extent
            Lx, Ly = validate_lcc_tuple(extent, "extent", FT; positive = true)
            Δx = Lx / convert(FT, Nx)
            Δy = Ly / convert(FT, Ny)
        else
            Δx, Δy = validate_lcc_tuple(spacing, "spacing", FT; positive = true)
            Lx = convert(FT, Nx) * Δx
            Ly = convert(FT, Ny) * Δy
        end

        x₁ = x_center - Lx / convert(FT, 2)
        y₁ = y_center - Ly / convert(FT, 2)
    end

    return x₁, y₁, Δx, Δy
end

function lcc_coordinate_matches_node(coordinate, first_face, spacing, N, ::Face)
    FT = typeof(spacing)
    fractional_index = one(FT) + (coordinate - first_face) / spacing
    tolerance = sqrt(eps(FT))

    return one(FT) <= fractional_index <= convert(FT, N + 1) &&
           abs(fractional_index - round(fractional_index)) <= tolerance
end

function lcc_coordinate_matches_node(coordinate, first_face, spacing, N, ::Center)
    FT = typeof(spacing)
    fractional_index = lcc_half(FT) + (coordinate - first_face) / spacing
    tolerance = sqrt(eps(FT))

    return one(FT) <= fractional_index <= convert(FT, N) &&
           abs(fractional_index - round(fractional_index)) <= tolerance
end

function validate_lcc_projected_domain(map::LambertConformalConic{FT}, size) where FT
    Nx, Ny, _ = size
    x₂ = map.x₁ + convert(FT, Nx) * map.Δx
    y₂ = map.y₁ + convert(FT, Ny) * map.Δy

    for x in (map.x₁, x₂), y in (map.y₁, y₂)
        λ, φ = lcc_inverse(map, x, y)

        isfinite(λ) && isfinite(φ) ||
            throw(ArgumentError("LambertConformalConicGrid domain inverse-projects to nonfinite " *
                                "longitude or latitude."))
    end

    x_apex = map.false_easting
    y_apex = map.false_northing + map.origin_radius
    apex_in_x_domain = map.x₁ <= x_apex <= x₂
    apex_in_y_domain = map.y₁ <= y_apex <= y₂

    apex_on_x_node = lcc_coordinate_matches_node(x_apex, map.x₁, map.Δx, Nx, Center()) ||
                     lcc_coordinate_matches_node(x_apex, map.x₁, map.Δx, Nx, Face())

    apex_on_y_node = lcc_coordinate_matches_node(y_apex, map.y₁, map.Δy, Ny, Center()) ||
                     lcc_coordinate_matches_node(y_apex, map.y₁, map.Δy, Ny, Face())

    if apex_in_x_domain && apex_in_y_domain
        node_message = ifelse(apex_on_x_node && apex_on_y_node, " on a grid node", "")
        @warn "LambertConformalConicGrid contains the cone apex / pole$(node_message); longitude is singular there."
    end

    return nothing
end

"""
    $(SIGNATURES)

Return a regional `LambertConformalConicGrid`, represented internally as an
`OrthogonalSphericalShellGrid` with 2D longitude, latitude, and spherical metric arrays.
The computational grid is regular in projected `x/y` coordinates measured in meters,
while the physical grid is a thin spherical shell in geographic longitude/latitude.

Specify exactly one horizontal domain mode:

- `center` and `spacing`, where `center = (longitude, latitude)` in degrees and
  `spacing` is a projected meter spacing.
- `center` and `extent`, where `extent` is the projected meter domain size.
- `x` and `y`, where both are projected-coordinate intervals in meters.

Use `standard_parallel` for the tangent one-standard-parallel case, or
`standard_parallels` for either one or two standard parallels. Horizontal topology
defaults to `(Bounded, Bounded, Bounded)` and periodic horizontal topology is not
supported. Lambert conformal conic is conformal but not equal-area, is intended for
regional domains rather than global domains, and has a singular longitude at the cone
apex / pole. A warning is emitted when the projected rectangle contains the cone
apex / pole.

# Polar stereographic limit

For pole-centred grids, set `standard_parallel = 90` (or `-90` for the South Pole)
along with `latitude_of_origin = 90` (or `-90`). This is the degenerate case of LCC
in which the cone is tangent to the sphere at the pole — the cone constant is `n = 1`
(or `-1`) and the projection becomes polar stereographic, with no missing-wedge
artefact at the antemeridian. For LCC with standard parallels strictly between `-90`
and `90` (`|n| < 1`), the cone unrolls to a sector of `2nπ` rather than `2π`, so a
small `2π(1-n)` wedge of physical longitude has no representation in projected
space; this is only a problem for grids that contain the projected apex, and is
removed entirely by using the polar stereographic limit above.

```jldoctest
using Oceananigans

grid = LambertConformalConicGrid(size = (8, 6, 1),
                                 center = (-105, 40),
                                 spacing = 10000,
                                 standard_parallels = (30, 60),
                                 z = (-100, 0))

print(summary(grid))

# output
8×6×1 OrthogonalSphericalShellGrid{Float64, Bounded, Bounded, Bounded} on CPU with 3×3×1 halo
```

```jldoctest
using Oceananigans

grid = LambertConformalConicGrid(size = (4, 4, 1),
                                 center = (-105, 40),
                                 extent = (80e3, 60e3),
                                 standard_parallels = (30, 60),
                                 z = (-1, 0))

print(summary(grid))

# output
4×4×1 OrthogonalSphericalShellGrid{Float64, Bounded, Bounded, Bounded} on CPU with 3×3×1 halo
```

```jldoctest
using Oceananigans

grid = LambertConformalConicGrid(size = (4, 4, 1),
                                 x = (-40e3, 40e3),
                                 y = (-30e3, 30e3),
                                 standard_parallels = (30, 60),
                                 central_longitude = -105,
                                 latitude_of_origin = 40,
                                 z = (-1, 0))

print(summary(grid))

# output
4×4×1 OrthogonalSphericalShellGrid{Float64, Bounded, Bounded, Bounded} on CPU with 3×3×1 halo
```

```jldoctest
using Oceananigans

grid = LambertConformalConicGrid(size = (4, 4, 1),
                                 center = (-100, 45),
                                 spacing = 10000,
                                 standard_parallel = 45,
                                 z = (-1, 0))

print(summary(grid))

# output
4×4×1 OrthogonalSphericalShellGrid{Float64, Bounded, Bounded, Bounded} on CPU with 3×3×1 halo
```
"""
function LambertConformalConicGrid(arch::AbstractArchitecture = CPU(),
                                   FT::DataType = Oceananigans.defaults.FloatType;
                                   size,
                                   z,
                                   center = nothing,
                                   spacing = nothing,
                                   extent = nothing,
                                   x = nothing,
                                   y = nothing,
                                   standard_parallel = nothing,
                                   standard_parallels = nothing,
                                   central_longitude = nothing,
                                   latitude_of_origin = nothing,
                                   false_easting = 0,
                                   false_northing = 0,
                                   radius = Oceananigans.defaults.planet_radius,
                                   halo = nothing,
                                   topology = (Bounded, Bounded, Bounded))

    if !isnothing(center)
        center = validate_lcc_center(center, FT)
    end

    TX, TY, TZ = validate_lcc_topology(topology)
    size = validate_size(TX, TY, TZ, size)
    halo = validate_halo(TX, TY, TZ, size, halo)
    z = validate_dimension_specification(TZ, z, :z, size[3], FT)

    if isnothing(central_longitude)
        isnothing(center) && throw(ArgumentError("central_longitude is required when center is not supplied."))
        central_longitude = center[1]
    end

    if isnothing(latitude_of_origin)
        isnothing(center) && throw(ArgumentError("latitude_of_origin is required when center is not supplied."))
        latitude_of_origin = center[2]
    end

    temporary_map = LambertConformalConic(FT;
                                          standard_parallel,
                                          standard_parallels,
                                          central_longitude,
                                          latitude_of_origin,
                                          false_easting,
                                          false_northing,
                                          radius,
                                          x₁ = zero(FT),
                                          y₁ = zero(FT),
                                          Δx = one(FT),
                                          Δy = one(FT))

    x₁, y₁, Δx, Δy = parse_lcc_domain(FT, temporary_map, size; x, y, center, extent, spacing)

    conformal_mapping = LambertConformalConic(FT;
                                              standard_parallel,
                                              standard_parallels,
                                              central_longitude,
                                              latitude_of_origin,
                                              false_easting,
                                              false_northing,
                                              radius,
                                              x₁, y₁, Δx, Δy)

    validate_lcc_projected_domain(conformal_mapping, size)

    grid = OrthogonalSphericalShellGrid(arch, FT; size, z, radius,
                                        halo, topology,
                                        conformal_mapping)

    fill_lcc_coordinates_and_metrics!(grid)

    return grid
end

LambertConformalConicGrid(FT::DataType; kwargs...) = LambertConformalConicGrid(CPU(), FT; kwargs...)

function fill_lcc_coordinates_and_metrics!(grid::LambertConformalConicGrid)
    arch = architecture(grid)
    Nx, Ny, _ = size(grid)
    Hx, Hy, _ = halo_size(grid)

    coordinate_parameters = KernelParameters(-Hx-1:Nx+Hx+2, -Hy-1:Ny+Hy+2)
    launch!(arch, grid, coordinate_parameters, _fill_lcc_coordinates!, grid)

    metric_parameters = KernelParameters(-Hx:Nx+Hx+1, -Hy:Ny+Hy+1)
    launch!(arch, grid, metric_parameters, _calculate_lcc_metrics!, grid)

    return nothing
end

@kernel function _fill_lcc_coordinates!(grid)
    i, j = @index(Global, NTuple)
    map = grid.conformal_mapping

    xᶜ = lcc_xnode(i, Center(), map)
    xᶠ = lcc_xnode(i, Face(), map)
    yᶜ = lcc_ynode(j, Center(), map)
    yᶠ = lcc_ynode(j, Face(), map)

    @inbounds begin
        λ, φ = lcc_inverse(map, xᶜ, yᶜ)
        grid.λᶜᶜᵃ[i, j] = λ
        grid.φᶜᶜᵃ[i, j] = φ

        λ, φ = lcc_inverse(map, xᶠ, yᶜ)
        grid.λᶠᶜᵃ[i, j] = λ
        grid.φᶠᶜᵃ[i, j] = φ

        λ, φ = lcc_inverse(map, xᶜ, yᶠ)
        grid.λᶜᶠᵃ[i, j] = λ
        grid.φᶜᶠᵃ[i, j] = φ

        λ, φ = lcc_inverse(map, xᶠ, yᶠ)
        grid.λᶠᶠᵃ[i, j] = λ
        grid.φᶠᶠᵃ[i, j] = φ
    end
end

@inline function spherical_distance(λ₁, φ₁, λ₂, φ₂, radius)
    FT = typeof(radius)
    λ₁ = degrees_to_radians(FT, λ₁)
    φ₁ = degrees_to_radians(FT, φ₁)
    λ₂ = degrees_to_radians(FT, λ₂)
    φ₂ = degrees_to_radians(FT, φ₂)

    half = lcc_half(FT)
    Δλ = lcc_wrap_to_π(λ₂ - λ₁)
    Δφ = φ₂ - φ₁
    haversine_argument = sin(half * Δφ)^2 + cos(φ₁) * cos(φ₂) * sin(half * Δλ)^2
    haversine_argument = min(max(haversine_argument, zero(FT)), one(FT))
    central_angle = convert(FT, 2) * atan(sqrt(haversine_argument), sqrt(one(FT) - haversine_argument))

    return radius * central_angle
end

@inline function spherical_unit_vector(λ, φ, ::Type{FT}) where FT
    λ = degrees_to_radians(FT, λ)
    φ = degrees_to_radians(FT, φ)
    cosφ = cos(φ)

    return cos(λ) * cosφ, sin(λ) * cosφ, sin(φ)
end

@inline dot_product(a, b) = a[1] * b[1] + a[2] * b[2] + a[3] * b[3]

@inline function cross_product(a, b)
    return (a[2] * b[3] - a[3] * b[2],
            a[3] * b[1] - a[1] * b[3],
            a[1] * b[2] - a[2] * b[1])
end

@inline function spherical_triangle_area(a, b, c)
    numerator = abs(dot_product(a, cross_product(b, c)))
    denominator = one(numerator) + dot_product(a, b) + dot_product(b, c) + dot_product(c, a)
    return convert(typeof(numerator), 2) * atan(numerator, denominator)
end

@inline function spherical_quadrilateral_area(λa, φa, λb, φb, λc, φc, λd, φd, radius)
    FT = typeof(radius)
    a = spherical_unit_vector(λa, φa, FT)
    b = spherical_unit_vector(λb, φb, FT)
    c = spherical_unit_vector(λc, φc, FT)
    d = spherical_unit_vector(λd, φd, FT)

    return (spherical_triangle_area(a, b, c) + spherical_triangle_area(a, c, d)) * radius^2
end

# This duplicates TripolarGrid's spherical metric conventions until the shared
# spherical-shell metric generation kernel is factored out.
@kernel function _calculate_lcc_metrics!(grid)
    i, j = @index(Global, NTuple)
    R = grid.radius

    @inbounds begin
        grid.Δxᶜᶜᵃ[i, j] = spherical_distance(grid.λᶠᶜᵃ[i+1, j], grid.φᶠᶜᵃ[i+1, j],
                                              grid.λᶠᶜᵃ[i,   j], grid.φᶠᶜᵃ[i,   j], R)

        grid.Δxᶠᶜᵃ[i, j] = spherical_distance(grid.λᶜᶜᵃ[i,   j], grid.φᶜᶜᵃ[i,   j],
                                              grid.λᶜᶜᵃ[i-1, j], grid.φᶜᶜᵃ[i-1, j], R)

        grid.Δxᶜᶠᵃ[i, j] = spherical_distance(grid.λᶠᶠᵃ[i+1, j], grid.φᶠᶠᵃ[i+1, j],
                                              grid.λᶠᶠᵃ[i,   j], grid.φᶠᶠᵃ[i,   j], R)

        grid.Δxᶠᶠᵃ[i, j] = spherical_distance(grid.λᶜᶠᵃ[i,   j], grid.φᶜᶠᵃ[i,   j],
                                              grid.λᶜᶠᵃ[i-1, j], grid.φᶜᶠᵃ[i-1, j], R)

        grid.Δyᶜᶜᵃ[i, j] = spherical_distance(grid.λᶜᶠᵃ[i, j+1], grid.φᶜᶠᵃ[i, j+1],
                                              grid.λᶜᶠᵃ[i, j  ], grid.φᶜᶠᵃ[i, j  ], R)

        grid.Δyᶠᶜᵃ[i, j] = spherical_distance(grid.λᶠᶠᵃ[i, j+1], grid.φᶠᶠᵃ[i, j+1],
                                              grid.λᶠᶠᵃ[i, j  ], grid.φᶠᶠᵃ[i, j  ], R)

        grid.Δyᶜᶠᵃ[i, j] = spherical_distance(grid.λᶜᶜᵃ[i, j  ], grid.φᶜᶜᵃ[i, j  ],
                                              grid.λᶜᶜᵃ[i, j-1], grid.φᶜᶜᵃ[i, j-1], R)

        grid.Δyᶠᶠᵃ[i, j] = spherical_distance(grid.λᶠᶜᵃ[i, j  ], grid.φᶠᶜᵃ[i, j  ],
                                              grid.λᶠᶜᵃ[i, j-1], grid.φᶠᶜᵃ[i, j-1], R)

        grid.Azᶜᶜᵃ[i, j] = spherical_quadrilateral_area(grid.λᶠᶠᵃ[i,   j  ], grid.φᶠᶠᵃ[i,   j  ],
                                                        grid.λᶠᶠᵃ[i+1, j  ], grid.φᶠᶠᵃ[i+1, j  ],
                                                        grid.λᶠᶠᵃ[i+1, j+1], grid.φᶠᶠᵃ[i+1, j+1],
                                                        grid.λᶠᶠᵃ[i,   j+1], grid.φᶠᶠᵃ[i,   j+1], R)

        grid.Azᶠᶜᵃ[i, j] = grid.Δyᶠᶜᵃ[i, j] * grid.Δxᶠᶜᵃ[i, j]
        grid.Azᶜᶠᵃ[i, j] = grid.Δyᶜᶠᵃ[i, j] * grid.Δxᶜᶠᵃ[i, j]

        grid.Azᶠᶠᵃ[i, j] = spherical_quadrilateral_area(grid.λᶜᶜᵃ[i-1, j-1], grid.φᶜᶜᵃ[i-1, j-1],
                                                        grid.λᶜᶜᵃ[i,   j-1], grid.φᶜᶜᵃ[i,   j-1],
                                                        grid.λᶜᶜᵃ[i,   j  ], grid.φᶜᶜᵃ[i,   j  ],
                                                        grid.λᶜᶜᵃ[i-1, j  ], grid.φᶜᶜᵃ[i-1, j  ], R)
    end
end

function Grids.with_halo(new_halo, old_grid::LambertConformalConicGrid)
    arch = architecture(old_grid)
    FT = eltype(old_grid)
    Nx, Ny, Nz = size(old_grid)
    topo = topology(old_grid)
    grid_size = Grids.pop_flat_elements((Nx, Ny, Nz), topo)
    halo = length(new_halo) == 3 ? Grids.pop_flat_elements(new_halo, topo) : new_halo
    map = old_grid.conformal_mapping
    z = cpu_face_constructor_z(old_grid)

    x = (map.x₁, map.x₁ + convert(FT, Nx) * map.Δx)
    y = (map.y₁, map.y₁ + convert(FT, Ny) * map.Δy)

    return LambertConformalConicGrid(arch, FT;
                                     size = grid_size,
                                     x, y, z,
                                     standard_parallels = (radians_to_degrees(FT, map.standard_parallel_1),
                                                           radians_to_degrees(FT, map.standard_parallel_2)),
                                     central_longitude = radians_to_degrees(FT, map.central_longitude),
                                     latitude_of_origin = radians_to_degrees(FT, map.latitude_of_origin),
                                     false_easting = map.false_easting,
                                     false_northing = map.false_northing,
                                     radius = map.radius,
                                     halo,
                                     topology = topo)
end

function Grids.constructor_arguments(grid::LambertConformalConicGrid)
    arch = architecture(grid)
    FT = eltype(grid)
    args = OrderedDict(:architecture => arch, :number_type => FT)

    topo = topology(grid)
    size = (grid.Nx, grid.Ny, grid.Nz)
    halo = (grid.Hx, grid.Hy, grid.Hz)
    size = Grids.pop_flat_elements(size, topo)
    halo = Grids.pop_flat_elements(halo, topo)

    map = grid.conformal_mapping
    x = (map.x₁, map.x₁ + convert(FT, grid.Nx) * map.Δx)
    y = (map.y₁, map.y₁ + convert(FT, grid.Ny) * map.Δy)

    kwargs = Dict(:size => size,
                  :halo => halo,
                  :x => x,
                  :y => y,
                  :z => cpu_face_constructor_z(grid),
                  :topology => topo,
                  :standard_parallels => (radians_to_degrees(FT, map.standard_parallel_1),
                                          radians_to_degrees(FT, map.standard_parallel_2)),
                  :central_longitude => radians_to_degrees(FT, map.central_longitude),
                  :latitude_of_origin => radians_to_degrees(FT, map.latitude_of_origin),
                  :false_easting => map.false_easting,
                  :false_northing => map.false_northing,
                  :radius => map.radius)

    return args, kwargs
end

function Base.similar(grid::LambertConformalConicGrid)
    args, kwargs = Grids.constructor_arguments(grid)
    arch = args[:architecture]
    FT = args[:number_type]
    return LambertConformalConicGrid(arch, FT; kwargs...)
end

function Grids.with_number_type(FT, grid::LambertConformalConicGrid)
    args, kwargs = Grids.constructor_arguments(grid)
    arch = args[:architecture]
    return LambertConformalConicGrid(arch, FT; kwargs...)
end
