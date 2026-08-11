#####
##### Paired baroclinic + barotropic open boundary conditions
#####

# A side spec is `nothing`/`false` (closed), `true` (open, use the shared external velocity),
# or a value/function/array (open, used as that side's external boundary-normal velocity).
open_side_external_velocity(::Nothing, shared_external_velocity) = nothing
open_side_external_velocity(open::Bool, shared_external_velocity) = open ? shared_external_velocity : nothing
open_side_external_velocity(side_external_velocity, shared_external_velocity) = side_external_velocity

maybe_normal_radiation_boundary_condition(::Nothing, radiation) = nothing
maybe_normal_radiation_boundary_condition(external_velocity, radiation) = NormalFlowBoundaryCondition(external_velocity; scheme = radiation)

maybe_gravity_wave_boundary_condition(::Nothing, flather) = nothing
maybe_gravity_wave_boundary_condition(external_velocity, flather) = flather

function open_field_boundary_conditions(sides::NamedTuple)
    present = NamedTuple(name => bc for (name, bc) in pairs(sides) if !isnothing(bc))
    return isempty(present) ? nothing : FieldBoundaryConditions(; present...)
end

"""
    BarotropicBaroclinicBoundaryConditions(; west = nothing, east = nothing,
                                             south = nothing, north = nothing,
                                             external_state = 0,
                                             external_transport = 0,
                                             external_free_surface = 0,
                                             inflow_timescale = 0,
                                             outflow_timescale = Inf,
                                             gravitational_acceleration = defaults.gravitational_acceleration,
                                             use_boundary_velocity = false)

Build the paired open boundary conditions that a `HydrostaticFreeSurfaceModel` with a
`SplitExplicitFreeSurface` needs at an open boundary: one on the three-dimensional (baroclinic)
velocity and one on the depth-integrated (barotropic) transport. It removes the need to construct
and pass both by hand.

Returns a `NamedTuple` suitable for the `boundary_conditions` keyword of `HydrostaticFreeSurfaceModel`.
For every open side it sets

  - a [`NormalFlowBoundaryCondition`](@ref) with the [`NormalRadiation`](@ref) scheme on the
    boundary-normal baroclinic velocity (`u` for `west`/`east`, `v` for `south`/`north`), and
  - a [`GravityWaveRadiationBoundaryCondition`](@ref) on the barotropic transport (`U`/`V`).

The companion [`SurfaceWaveRadiation`](@ref) condition on the free surface `η` is added automatically
by the model, so it is not returned here.

Open sides are selected with the `west`, `east`, `south`, `north` keywords. A side left at `nothing`
(or `false`) is closed. A side set to `true` is open and uses the shared `external_state`; a side set
to a number, array, or function is open and uses that value as its external boundary-normal baroclinic
velocity.

Keyword arguments
=================

  - `external_state`: external boundary-normal baroclinic velocity used by sides set to `true` [m s⁻¹].
  - `external_transport`: external barotropic transport prescribed to `GravityWaveRadiation` [m² s⁻¹].
  - `external_free_surface`: external free surface displacement prescribed to `GravityWaveRadiation` [m].
  - `inflow_timescale`, `outflow_timescale`: `NormalRadiation` nudging timescales [s].
  - `gravitational_acceleration`: passed to `GravityWaveRadiation`.
  - `use_boundary_velocity`: passed to `NormalRadiation`.

For per-side external transports or free surfaces, or for BC values that need `discrete_form`,
construct the `u`/`v` and `U`/`V` boundary conditions directly instead.

Example
=======

```jldoctest
using Oceananigans
using Oceananigans.BoundaryConditions: BarotropicBaroclinicBoundaryConditions

momentum = BarotropicBaroclinicBoundaryConditions(west = true, east = true, outflow_timescale = 100)

momentum.u.east.classification.scheme

# output
NormalRadiation{Float64}
├── inflow_timescale: 0.0
├── outflow_timescale: 100.0
└── use_boundary_velocity: false
```
"""
function BarotropicBaroclinicBoundaryConditions(; west = nothing, east = nothing,
                                                  south = nothing, north = nothing,
                                                  external_state = 0,
                                                  external_transport = 0,
                                                  external_free_surface = 0,
                                                  inflow_timescale = 0,
                                                  outflow_timescale = Inf,
                                                  gravitational_acceleration = defaults.gravitational_acceleration,
                                                  use_boundary_velocity = false)

    uʷ = open_side_external_velocity(west,  external_state)
    uᵉ = open_side_external_velocity(east,  external_state)
    vˢ = open_side_external_velocity(south, external_state)
    vⁿ = open_side_external_velocity(north, external_state)

    radiation = NormalRadiation(; inflow_timescale, outflow_timescale, use_boundary_velocity)
    flather   = GravityWaveRadiationBoundaryCondition((external_transport, external_free_surface); gravitational_acceleration)

    bcs = (u = open_field_boundary_conditions((west  = maybe_normal_radiation_boundary_condition(uʷ, radiation),
                                               east  = maybe_normal_radiation_boundary_condition(uᵉ, radiation))),
           U = open_field_boundary_conditions((west  = maybe_gravity_wave_boundary_condition(uʷ, flather),
                                               east  = maybe_gravity_wave_boundary_condition(uᵉ, flather))),
           v = open_field_boundary_conditions((south = maybe_normal_radiation_boundary_condition(vˢ, radiation),
                                               north = maybe_normal_radiation_boundary_condition(vⁿ, radiation))),
           V = open_field_boundary_conditions((south = maybe_gravity_wave_boundary_condition(vˢ, flather),
                                               north = maybe_gravity_wave_boundary_condition(vⁿ, flather))))

    return NamedTuple(name => bc for (name, bc) in pairs(bcs) if !isnothing(bc))
end
