#####
##### Tidal harmonics
#####

struct TidalHarmonics{N, FT, C}
    constituents :: C
    frequencies :: NTuple{N, FT}
    phases :: NTuple{N, FT}
    nodal_factors :: NTuple{N, FT}
    equilibrium_amplitudes :: NTuple{N, FT}
    species :: NTuple{N, Int}
    ramp_time :: FT
end

"""
$(TYPEDSIGNATURES)

A set of tidal constituents, each an oscillation of frequency ``ω`` [rad s⁻¹] holding phase ``V + u``
[rad] at ``t = 0``, so that a constituent of amplitude ``A`` and phase lag ``G`` contributes
``f A \\cos(ω t + V + u - G)`` for a nodal factor ``f``. Model time is seconds from the epoch the
phases were computed for.

`constituents` are the labels `tidal_boundary_conditions` passes to `tidal_atlas_constants` to look
up harmonic constants. `equilibrium_amplitudes` [m] and `species` set the equilibrium tide that
`tidal_forcing` differentiates: species 2 is semidiurnal, with latitude structure ``\\cos^2 φ``;
1 is diurnal, ``\\sin 2φ``; and 0 is long-period, ``\\frac{1}{2} - \\frac{3}{2} \\sin^2 φ``. These are
the degree-2 components of the tidal potential of a distant body.

`nodal_factors` default to one: no slow modulation of amplitude. `ramp_time` eases the tide in from
rest as ``\\tanh(t / T)``, which a basin started impulsively needs to avoid ringing its gravest
gravity mode.

A single `TidalHarmonics` feeds both `tidal_forcing` and `tidal_boundary_conditions`, which keeps
the equilibrium tide and the boundary tide in phase with each other.

```jldoctest
using Oceananigans

TidalHarmonics(constituents = (:M2,),
               frequencies = (1.405189e-4,),
               phases = (1.7,),
               equilibrium_amplitudes = (0.168,),
               species = (2,))

# output
TidalHarmonics with M2
```
"""
function TidalHarmonics(; constituents,
                          frequencies,
                          phases,
                          equilibrium_amplitudes,
                          species,
                          nodal_factors = map(one, Tuple(frequencies)),
                          ramp_time = 0)

    FT = Oceananigans.defaults.FloatType

    return TidalHarmonics(Tuple(constituents),
                          map(FT, Tuple(frequencies)),
                          map(FT, Tuple(phases)),
                          map(FT, Tuple(nodal_factors)),
                          map(FT, Tuple(equilibrium_amplitudes)),
                          map(Int, Tuple(species)),
                          FT(ramp_time))
end

Adapt.adapt_structure(to, harmonics::TidalHarmonics) =
    TidalHarmonics(nothing,
                   harmonics.frequencies,
                   harmonics.phases,
                   harmonics.nodal_factors,
                   harmonics.equilibrium_amplitudes,
                   harmonics.species,
                   harmonics.ramp_time)

Base.summary(harmonics::TidalHarmonics) =
    string("TidalHarmonics with ", join(harmonics.constituents, ", "))

Base.show(io::IO, harmonics::TidalHarmonics) = print(io, summary(harmonics))

@inline tidal_ramp(harmonics, t) = ifelse(harmonics.ramp_time > 0, tanh(t / harmonics.ramp_time), one(t))

#####
##### The equilibrium tide
#####

# Elevation [m] of the equilibrium tide at a cell center. The latitude structure of a constituent
# follows from its species: cos²φ semidiurnal, sin2φ diurnal, and ½ - 3/2 sin²φ long-period.
@inline function equilibrium_tide(i, j, k, grid, clock, harmonics)
    λ = deg2rad(λnode(i, j, k, grid, Center(), Center(), Center()))
    φ = deg2rad(φnode(i, j, k, grid, Center(), Center(), Center()))
    t = clock.time

    η = zero(grid)

    @inbounds for n in eachindex(harmonics.frequencies)
        s = harmonics.species[n]
        structure = ifelse(s == 2, cos(φ)^2, ifelse(s == 1, sin(2φ), (1 - 3 * sin(φ)^2) / 2))
        η += harmonics.nodal_factors[n] * harmonics.equilibrium_amplitudes[n] * structure *
             cos(harmonics.frequencies[n] * t + harmonics.phases[n] + s * λ)
    end

    return η
end

#####
##### Boundary conditions from a tidal atlas
#####

"""
$(TYPEDSIGNATURES)

Complex harmonic constants of `constituent` from `dataset`, at each `(longitude, latitude)` point in
`nodes`: sea surface height [m], eastward transport [m² s⁻¹] and northward transport [m² s⁻¹], one
vector per quantity, in the order of `nodes`. A constant pairs with its Greenwich phase lag ``G`` as
``A e^{-i G}``.

Sampling `dataset` at arbitrary points — its native grid, staggering and interpolation — is entirely
the implementation's concern; `nodes` may fall anywhere, including on land, which the implementation
also decides how to handle.
"""
function tidal_atlas_constants end

# Greenwich phase lags enter as complex constants, so a constituent contributes
# real(f A e^{-iG} e^{i(ωt + V + u)}) to the transport and to the elevation alike.
@inline function tidal_transport_and_elevation(i, j, grid, clock, fields, p)
    t = clock.time
    harmonics = p.harmonics

    U = zero(grid)
    η = zero(grid)

    @inbounds for n in eachindex(harmonics.frequencies)
        rotation = harmonics.nodal_factors[n] * cis(harmonics.frequencies[n] * t + harmonics.phases[n])
        U += real(p.transport[i, n] * rotation)
        η += real(p.elevation[i, n] * rotation)
    end

    ramp = tidal_ramp(harmonics, t)

    return (ramp * U, ramp * η)
end

# A (n_nodes, n_constituents) matrix of one field's constants, one column per constituent.
constants_matrix(field, per_constituent) = reduce(hcat, getproperty.(per_constituent, field))

function flather_condition(harmonics, dataset, transport_field, arch, transport_nodes, elevation_nodes; kw...)
    at_transport = map(name -> tidal_atlas_constants(dataset, transport_nodes, name; kw...), harmonics.constituents)
    at_elevation = map(name -> tidal_atlas_constants(dataset, elevation_nodes, name; kw...), harmonics.constituents)

    parameters = (; harmonics,
                  transport = on_architecture(arch, constants_matrix(transport_field, at_transport)),
                  elevation = on_architecture(arch, constants_matrix(:sea_surface_height, at_elevation)))

    return GravityWaveRadiationBoundaryCondition(tidal_transport_and_elevation; discrete_form = true, parameters)
end

"""
$(TYPEDSIGNATURES)

Boundary conditions `(U, V, η)` that drive the barotropic tide of `harmonics` through every lateral
boundary of `grid`, with the tidal transports and elevations of `dataset`: Flather conditions
[Flather (1976)](@cite flather1976tidal) on the transports `U` and `V`, which carry the tide in, and
radiation conditions [Chapman (1985)](@cite chapman1985numerical) on the free surface `η`.

The atlas is sampled at the boundary nodes themselves, the transports where the model holds `U` and
`V` and the elevations at the adjacent cell centers, which the Flather condition compares with its
own.

The transports enter unconverted, so the tidal mass flux is preserved where the atlas and the model
disagree about depth. Keyword arguments reach `dataset`, such as the `dir` it reads.
"""
function tidal_boundary_conditions(grid, harmonics::TidalHarmonics, dataset; kw...)

    arch = architecture(grid)
    λᶜ, λᶠ = λnodes(grid, Center()), λnodes(grid, Face())
    φᶜ, φᶠ = φnodes(grid, Center()), φnodes(grid, Face())

    west = flather_condition(harmonics, dataset, :eastward_transport, arch,
                             [(first(λᶠ), φ) for φ in φᶜ], [(first(λᶜ), φ) for φ in φᶜ]; kw...)

    east = flather_condition(harmonics, dataset, :eastward_transport, arch,
                             [(last(λᶠ), φ) for φ in φᶜ], [(last(λᶜ), φ) for φ in φᶜ]; kw...)

    south = flather_condition(harmonics, dataset, :northward_transport, arch,
                              [(λ, first(φᶠ)) for λ in λᶜ], [(λ, first(φᶜ)) for λ in λᶜ]; kw...)

    north = flather_condition(harmonics, dataset, :northward_transport, arch,
                              [(λ, last(φᶠ)) for λ in λᶜ], [(λ, last(φᶜ)) for λ in λᶜ]; kw...)

    U = FieldBoundaryConditions(grid, (Face(), Center(), nothing); west, east)
    V = FieldBoundaryConditions(grid, (Center(), Face(), nothing); south, north)

    radiation = SurfaceWaveRadiationBoundaryCondition()
    η = FieldBoundaryConditions(grid, (Center(), Center(), Face());
                                west = radiation, east = radiation, south = radiation, north = radiation)

    return (; U, V, η)
end
