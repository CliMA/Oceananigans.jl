using DocStringExtensions: TYPEDEF, TYPEDFIELDS
using Oceananigans.BuoyancyFormulations: ∂x_b, ∂y_b, ∂z_b, top_buoyancy_flux
using Oceananigans.Coriolis: fᶠᶠᵃ
using Oceananigans.Fields: VelocityFields
using Oceananigans.Grids: Flat, XFlatGrid, YFlatGrid, halo_size, topology, inactive_node, peripheral_node
using Oceananigans.Operators: Δxᶠᶜᶜ, Δyᶠᶜᶜ, Δxᶜᶠᶜ, Δyᶜᶠᶜ, Δzᶜᶜᶠ, Δz⁻¹ᶜᶠᶜ, Δx_qᶜᶠᶠ
using Oceananigans.Utils: KernelParameters

using .TKEBasedVerticalDiffusivities: friction_velocity, get_top_tracer_bcs

using Oceananigans: prognostic_state, restore_prognostic_state!

"""
$(TYPEDEF)

Frontal length scale ``L_f = \\bar{N} H / \\sqrt{f² + τ_f^{-2}}`` set by the deformation radius of the
mixed layer (Fox-Kemper et al. 2011).

Fields
======

$(TYPEDFIELDS)
"""
struct DeformationFrontalLength{FT}
    "Efficiency coefficient ``Cₑ`` of the overturning streamfunction"
    eddy_coefficient :: FT
    "Time scale ``τ_f`` (s) regularizing the inertial time scale near the equator"
    frontal_timescale :: FT
    "Lower bound (m) of the frontal length scale"
    minimum_frontal_length :: FT
end

"""
$(TYPEDSIGNATURES)

Return a `DeformationFrontalLength` with the frontal length scale ``L_f = \\max(\\bar{N} H / f̃, L_{f, min})``,
where ``f̃ = \\sqrt{f² + τ_f^{-2}}``.

Keyword arguments
=================

- `eddy_coefficient`: default 0.0625.
- `frontal_timescale`: default 1 day.
- `minimum_frontal_length`: default 1 km.

```jldoctest
julia> using Oceananigans.TurbulenceClosures: DeformationFrontalLength

julia> DeformationFrontalLength(eddy_coefficient = 0.07)
DeformationFrontalLength{Float64}(0.07, 86400.0, 1000.0)
```
"""
function DeformationFrontalLength(FT = Oceananigans.defaults.FloatType;
                                  eddy_coefficient = 0.0625,
                                  frontal_timescale = 86400,
                                  minimum_frontal_length = 1000)

    return DeformationFrontalLength{FT}(convert(FT, eddy_coefficient),
                                        convert(FT, frontal_timescale),
                                        convert(FT, minimum_frontal_length))
end

"""
$(TYPEDEF)

Frontal length scale ``L_f ∝ \\overline{w'u'} / (f² h)`` of fronts whose frontogenesis is arrested by
boundary-layer turbulence (Bodner et al. 2023).

Fields
======

$(TYPEDFIELDS)
"""
struct ArrestedFrontalLength{FT}
    "Efficiency coefficient ``C_r`` of the overturning streamfunction"
    efficiency_coefficient :: FT
    "Coefficient ``m_★`` of the mechanically driven turbulent momentum flux"
    mechanical_coefficient :: FT
    "Coefficient ``n_★`` of the convectively driven turbulent momentum flux"
    convective_coefficient :: FT
    "Lower bound (m² s⁻²) of the turbulent momentum flux"
    minimum_momentum_flux :: FT
end

"""
$(TYPEDSIGNATURES)

Return an `ArrestedFrontalLength` with the frontal length scale set by the turbulent momentum flux of the
boundary layer,

```math
\\overline{w'u'} = \\max \\left [ (m_★ u_★³ + n_★ w_★³)^{2/3}, \\overline{w'u'}_{min} \\right ] ,
```

where ``u_★`` is the friction velocity and ``w_★³ = \\max(0, J_b) h`` is the convective velocity scale
of the instantaneous mixed layer of depth ``h`` under the upward surface buoyancy flux ``J_b``.
The default `efficiency_coefficient` and `minimum_momentum_flux` are the values of the CESM3 ocean configuration.

Keyword arguments
=================

- `efficiency_coefficient`: default 0.01.
- `mechanical_coefficient`: default 0.5.
- `convective_coefficient`: default 0.066.
- `minimum_momentum_flux`: default 1e-9 m² s⁻².

```jldoctest
julia> using Oceananigans.TurbulenceClosures: ArrestedFrontalLength

julia> ArrestedFrontalLength(efficiency_coefficient = 0.03)
ArrestedFrontalLength{Float64}(0.03, 0.5, 0.066, 1.0e-9)
```
"""
function ArrestedFrontalLength(FT = Oceananigans.defaults.FloatType;
                               efficiency_coefficient = 0.01,
                               mechanical_coefficient = 0.5,
                               convective_coefficient = 0.066,
                               minimum_momentum_flux = 1e-9)

    return ArrestedFrontalLength{FT}(convert(FT, efficiency_coefficient),
                                     convert(FT, mechanical_coefficient),
                                     convert(FT, convective_coefficient),
                                     convert(FT, minimum_momentum_flux))
end

"""
$(TYPEDEF)

Parameterization of the restratifying overturning of unresolved mixed-layer eddies
following Fox-Kemper et al. (2011), with the mixed-layer depth diagnosed from the potential
energy needed to homogenize the surface layer (Reichl et al. 2022).

Fields
======

$(TYPEDFIELDS)
"""
struct MixedLayerEddyRestratification{FT, L} <: AbstractTurbulenceClosure{ExplicitTimeDiscretization, 2}
    "Scaling of the frontal length, `DeformationFrontalLength` or `ArrestedFrontalLength`"
    frontal_length :: L
    "Potential energy (J m⁻²) that defines the mixed-layer depth"
    mixed_layer_energy :: FT
    "Reference density (kg m⁻³) converting `mixed_layer_energy` to buoyancy units"
    reference_density :: FT
    "Time scale (s) over which the filtered mixed-layer depth shoals; deepening is instantaneous"
    mixed_layer_decay_time :: FT
    "Upper bound (m) of the grid spacing that scales the overturning"
    maximum_grid_spacing :: FT
    "Upper bound (m s⁻¹) of the horizontal eddy velocity"
    maximum_eddy_velocity :: FT
end

const MLER = MixedLayerEddyRestratification

"""
$(TYPEDSIGNATURES)

Return a closure that advects all tracers with the eddy-induced velocity of the
mixed-layer eddy overturning streamfunction of Fox-Kemper et al. (2011),

```math
𝚿 = \\frac{Δs H² ∇\\bar{b} × \\hat{𝐳}}{𝒮} μ(z) ,
\\qquad μ = \\max \\left \\{ 0, [1 - (2ζ + 1)²] [1 + \\tfrac{5}{21} (2ζ + 1)²] \\right \\} ,
```

where ``ζ = (z - η) / H``, ``∇\\bar{b}`` is the buoyancy gradient averaged over the mixed layer
and ``Δs = \\min(\\sqrt{Δx Δy}, Δs_{max})``. The `frontal_length` sets the scale ``𝒮``:

- `DeformationFrontalLength` (Fox-Kemper et al. 2011): ``𝒮 = L_f f̃ / Cₑ`` with ``f̃ = \\sqrt{f² + τ_f^{-2}}``
  and ``L_f = \\max(\\bar{N} H / f̃, L_{f, min})``.
- `ArrestedFrontalLength` (Bodner et al. 2023): ``𝒮 = \\overline{w'u'} / (C_r |f| h)``, where ``h`` is the
  instantaneous mixed-layer depth and ``\\overline{w'u'}`` the turbulent momentum flux of the boundary layer.

The instantaneous mixed-layer depth ``h`` is the depth where the potential energy needed to mix
the layer above reaches `mixed_layer_energy`. Buoyancy is integrated from the local buoyancy frequency
so that ``h`` is independent of the compressibility of the equation of state.
``H`` follows ``h`` immediately when ``h`` deepens and relaxes towards it over `mixed_layer_decay_time`
when it shoals. ``H`` is updated once per time step and is part of the checkpointed state.
The magnitude of the streamfunction is limited so that the horizontal eddy velocity does not exceed
`maximum_eddy_velocity`.

The eddy velocities advect the tracers that `closure_auxiliary_velocity` reaches;
they do not advect the TKE of `CATKEVerticalDiffusivity`.

Keyword arguments
=================

- `frontal_length`: default `DeformationFrontalLength(FT)`.
- `mixed_layer_energy`: default 25 J m⁻².
- `reference_density`: default 1020 kg m⁻³.
- `mixed_layer_decay_time`: default 5 days.
- `maximum_grid_spacing`: default 100 km.
- `maximum_eddy_velocity`: default 0.2 m s⁻¹.

```jldoctest
julia> using Oceananigans

julia> using Oceananigans.TurbulenceClosures: MixedLayerEddyRestratification, ArrestedFrontalLength

julia> closure = MixedLayerEddyRestratification(frontal_length = ArrestedFrontalLength(), mixed_layer_energy = 10)
MixedLayerEddyRestratification:
├── frontal_length: ArrestedFrontalLength{Float64}(0.01, 0.5, 0.066, 1.0e-9)
├── mixed_layer_energy: 10.0
├── reference_density: 1020.0
├── mixed_layer_decay_time: 432000.0
├── maximum_grid_spacing: 100000.0
└── maximum_eddy_velocity: 0.2
```

References
==========

Bodner, A. S., Fox-Kemper, B., Johnson, L., Van Roekel, L. P., McWilliams, J. C., Sullivan, P. P.,
    Hall, P. S. and Dong, J. (2023). Modifying the mixed layer eddy parameterization to include
    frontogenesis arrest by boundary layer turbulence. Journal of Physical Oceanography, 53, 323–339.

Fox-Kemper, B., Danabasoglu, G., Ferrari, R., Griffies, S. M., Hallberg, R. W., Holland, M. M.,
    Maltrud, M. E., Peacock, S. and Samuels, B. L. (2011). Parameterization of mixed layer eddies.
    III: Implementation and impact in global ocean climate simulations. Ocean Modelling, 39, 61–78.

Reichl, B. G., Adcroft, A., Griffies, S. M. and Hallberg, R. (2022). A potential energy analysis of
    ocean surface mixed layers. Journal of Geophysical Research: Oceans, 127, e2021JC018140.
"""
function MixedLayerEddyRestratification(FT = Oceananigans.defaults.FloatType;
                                        frontal_length = DeformationFrontalLength(FT),
                                        mixed_layer_energy = 25,
                                        reference_density = 1020,
                                        mixed_layer_decay_time = 5 * 86400,
                                        maximum_grid_spacing = 100000,
                                        maximum_eddy_velocity = 0.2)

    return MixedLayerEddyRestratification{FT, typeof(frontal_length)}(frontal_length,
                                                                      convert(FT, mixed_layer_energy),
                                                                      convert(FT, reference_density),
                                                                      convert(FT, mixed_layer_decay_time),
                                                                      convert(FT, maximum_grid_spacing),
                                                                      convert(FT, maximum_eddy_velocity))
end

#####
##### Closure fields
#####

function build_closure_fields(grid, clock, tracer_names, bcs, ::MLER)
    h  = Field{Center, Center, Nothing}(grid)
    H  = Field{Center, Center, Nothing}(grid)
    Ψˣ = Field{Face, Center, Nothing}(grid)
    Ψʸ = Field{Center, Face, Nothing}(grid)
    w′u′ = Field{Center, Center, Nothing}(grid)
    U  = VelocityFields(grid)
    return (; h, H, Ψˣ, Ψʸ, w′u′, u=U.u, v=U.v, w=U.w)
end

const MixedLayerEddyRestratificationFields = NamedTuple{(:h, :H, :Ψˣ, :Ψʸ, :w′u′, :u, :v, :w)}

Oceananigans.prognostic_state(closure_fields::MixedLayerEddyRestratificationFields) = (; H = prognostic_state(closure_fields.H))

function Oceananigans.restore_prognostic_state!(restored::MixedLayerEddyRestratificationFields, from)
    restore_prognostic_state!(restored.H, from.H)
    return restored
end

Oceananigans.restore_prognostic_state!(::MixedLayerEddyRestratificationFields, ::Nothing) = nothing

@inline closure_auxiliary_velocity(::MLER, K, val_tracer_name) = (u = K.u, v = K.v, w = K.w)

@inline diffusive_flux_x(i, j, k, grid, ::MLER, args...) = zero(grid)
@inline diffusive_flux_y(i, j, k, grid, ::MLER, args...) = zero(grid)
@inline diffusive_flux_z(i, j, k, grid, ::MLER, args...) = zero(grid)

@inline viscous_flux_ux(i, j, k, grid, ::MLER, args...) = zero(grid)
@inline viscous_flux_uy(i, j, k, grid, ::MLER, args...) = zero(grid)
@inline viscous_flux_uz(i, j, k, grid, ::MLER, args...) = zero(grid)
@inline viscous_flux_vx(i, j, k, grid, ::MLER, args...) = zero(grid)
@inline viscous_flux_vy(i, j, k, grid, ::MLER, args...) = zero(grid)
@inline viscous_flux_vz(i, j, k, grid, ::MLER, args...) = zero(grid)
@inline viscous_flux_wx(i, j, k, grid, ::MLER, args...) = zero(grid)
@inline viscous_flux_wy(i, j, k, grid, ::MLER, args...) = zero(grid)
@inline viscous_flux_wz(i, j, k, grid, ::MLER, args...) = zero(grid)

#####
##### Energy-based mixed-layer depth
#####

"""
$(TYPEDSIGNATURES)

Return the depth below the free surface at which the potential energy per unit area,
divided by the reference density, needed to homogenize the layer above reaches `ε★`.
Return the depth of the active column when `ε★` is never reached.
"""
@inline function energy_mixed_layer_depth(i, j, grid, ε★, buoyancy, tracers)
    Nz = size(grid, 3)
    η = znode(i, j, Nz+1, grid, Center(), Center(), Face())

    b = zero(grid)
    D = zero(grid)
    B = zero(grid)
    M = zero(grid)
    Z = zero(grid)
    ε⁻ = zero(grid)
    h = zero(grid)
    crossed = false

    for k in Nz:-1:1
        active = !inactive_node(i, j, k, grid, Center(), Center(), Center())
        Δz = ifelse(active, Δzᶜᶜᶜ(i, j, k, grid), zero(grid))

        # Buoyancy relative to the top cell, integrated from the local N² of the face above
        N² = ifelse(k == Nz, zero(grid), ∂z_b(i, j, k+1, grid, buoyancy, tracers))
        b -= N² * Δzᶜᶜᶠ(i, j, k+1, grid)

        z = znode(i, j, k, grid, Center(), Center(), Center()) - η
        D += Δz
        B += b * Δz
        M += b * z * Δz
        Z += z * Δz

        ε = ifelse(D > 0, M - B * Z / D, zero(grid))
        crossing = !crossed & active & (ε ≥ ε★)
        dᵗ = η - znode(i, j, k+1, grid, Center(), Center(), Face())
        h = ifelse(crossing, dᵗ + Δz * (ε★ - ε⁻) / (ε - ε⁻), h)
        crossed = crossed | crossing
        ε⁻ = ε
    end

    return ifelse(crossed, h, D)
end

@kernel function _compute_mixed_layer_depth!(h, H, grid, closure, buoyancy, tracers, Δt)
    i, j = @index(Global, NTuple)

    ε★ = closure.mixed_layer_energy / closure.reference_density
    τ = closure.mixed_layer_decay_time

    hᵢⱼ = energy_mixed_layer_depth(i, j, grid, ε★, buoyancy, tracers)

    @inbounds begin
        Hᵢⱼ = H[i, j, 1]
        h[i, j, 1] = hᵢⱼ
        H[i, j, 1] = ifelse(hᵢⱼ > Hᵢⱼ, hᵢⱼ, Hᵢⱼ + Δt / (τ + Δt) * (hᵢⱼ - Hᵢⱼ))
    end
end

function compute_mixed_layer_depth!(closure_fields, closure::MLER, model, Δt)
    grid = model.grid
    buoyancy = model.buoyancy.formulation

    launch!(architecture(grid), grid, :xy, _compute_mixed_layer_depth!,
            closure_fields.h, closure_fields.H, grid, closure, buoyancy, model.tracers, Δt)

    return nothing
end

#####
##### Turbulent momentum flux of the boundary layer
#####

@kernel function _compute_boundary_layer_momentum_flux!(w′u′, grid, scaling, h, buoyancy,
                                                        top_tracer_bcs, top_velocity_bcs, clock, fields)
    i, j = @index(Global, NTuple)

    u★ = friction_velocity(i, j, grid, clock, fields, top_velocity_bcs)
    Jᵇ = top_buoyancy_flux(i, j, grid, buoyancy, top_tracer_bcs, clock, fields)
    w★³ = @inbounds max(zero(grid), Jᵇ) * h[i, j, 1]

    m★ = scaling.mechanical_coefficient
    n★ = scaling.convective_coefficient

    @inbounds w′u′[i, j, 1] = max(cbrt(m★ * u★^3 + n★ * w★³)^2, scaling.minimum_momentum_flux)
end

compute_boundary_layer_momentum_flux!(closure_fields, ::DeformationFrontalLength, model) = nothing

function compute_boundary_layer_momentum_flux!(closure_fields, scaling::ArrestedFrontalLength, model)
    grid = model.grid
    buoyancy = buoyancy_force(model)
    top_tracer_bcs = get_top_tracer_bcs(buoyancy, buoyancy_tracers(model))
    top_velocity_bcs = (u = model.velocities.u.boundary_conditions.top,
                        v = model.velocities.v.boundary_conditions.top)

    launch!(architecture(grid), grid, :xy, _compute_boundary_layer_momentum_flux!,
            closure_fields.w′u′, grid, scaling, closure_fields.h, buoyancy,
            top_tracer_bcs, top_velocity_bcs, model.clock, fields(model))

    return nothing
end

function TimeSteppers.step_closure_prognostics!(closure_fields, closure::MLER, model, Δt)
    compute_mixed_layer_depth!(closure_fields, closure, model, Δt)
    compute_boundary_layer_momentum_flux!(closure_fields, closure.frontal_length, model)
    fill_halo_regions!((closure_fields.h, closure_fields.H, closure_fields.w′u′))
    return nothing
end

#####
##### Overturning streamfunction
#####

# Filtered mixed-layer depth at u- and v-faces
@inline Hᶠᶜ(i, j, grid, H) = @inbounds min(H[i-1, j, 1], H[i, j, 1])
@inline Hᶜᶠ(i, j, grid, H) = @inbounds min(H[i, j-1, 1], H[i, j, 1])
@inline Hᶠᶜ(i, j, grid::XFlatGrid, H) = @inbounds H[i, j, 1]
@inline Hᶜᶠ(i, j, grid::YFlatGrid, H) = @inbounds H[i, j, 1]

# Vertical structure of the streamfunction, ζ = (z - η) / H
@inline function mixed_layer_structure(ζ)
    s² = (2ζ + 1)^2
    return max(zero(ζ), (1 - s²) * (1 + 5 * s² / 21))
end

# Inverse of the scale 𝒮 that converts Δs H² ∇b̄ into the streamfunction amplitude
@inline function inverse_frontal_scale(scaling::DeformationFrontalLength, H̃, h̃, N̄², f, w′u′)
    f̃ = sqrt(f^2 + 1 / scaling.frontal_timescale^2)
    Lf = max(sqrt(max(N̄², zero(N̄²))) * H̃ / f̃, scaling.minimum_frontal_length)
    return scaling.eddy_coefficient / (Lf * f̃)
end

# The bound guards `w′u′` before it is first computed by `step_closure_prognostics!`
@inline inverse_frontal_scale(scaling::ArrestedFrontalLength, H̃, h̃, N̄², f, w′u′) =
    scaling.efficiency_coefficient * abs(f) * h̃ / max(w′u′, scaling.minimum_momentum_flux)

# Amplitude of the streamfunction from the mixed-layer-averaged buoyancy gradient `b̄ₕ` and stratification `N̄²`
@inline function streamfunction_amplitude(closure, H̃, h̃, b̄ₕ, N̄², f, Δs, w′u′)
    Δs = min(Δs, closure.maximum_grid_spacing)
    Ψ = - Δs * H̃^2 * b̄ₕ * inverse_frontal_scale(closure.frontal_length, H̃, h̃, N̄², f, w′u′)

    # The largest |dμ/dζ| ≈ 5 is attained at the top and base of the mixed layer
    Ψₘₐₓ = closure.maximum_eddy_velocity * H̃ / 5
    return clamp(Ψ, -Ψₘₐₓ, Ψₘₐₓ)
end

@inline function mixed_layer_streamfunctionᶠᶜ(i, j, grid, closure, h, H, w′u′, buoyancy, tracers, coriolis)
    Nz = size(grid, 3)
    H̃ = Hᶠᶜ(i, j, grid, H)
    η = znode(i, j, Nz+1, grid, Face(), Center(), Face())
    zᴴ = η - H̃

    D = zero(grid)
    Σbₓ = zero(grid)
    ΣN² = zero(grid)

    k = Nz
    zᵗ = η

    # Only cells above the base of the mixed layer contribute
    while (k ≥ 1) & (zᵗ > zᴴ)
        zᵇ = znode(i, j, k, grid, Face(), Center(), Face())
        δ = zᵗ - max(zᵇ, zᴴ)
        δ = ifelse(peripheral_node(i, j, k, grid, Face(), Center(), Center()), zero(grid), δ)

        D += δ
        Σbₓ += ∂x_b(i, j, k, grid, buoyancy, tracers) * δ
        ΣN² += ℑxzᶠᵃᶜ(i, j, k, grid, ∂z_b, buoyancy, tracers) * δ

        zᵗ = zᵇ
        k -= 1
    end

    b̄ₓ = ifelse(D > 0, Σbₓ / D, zero(grid))
    N̄² = ifelse(D > 0, ΣN² / D, zero(grid))
    f = ℑyᵃᶜᵃ(i, j, 1, grid, fᶠᶠᵃ, coriolis)
    Δs = sqrt(Δxᶠᶜᶜ(i, j, Nz, grid) * Δyᶠᶜᶜ(i, j, Nz, grid))
    h̃ = min(Hᶠᶜ(i, j, grid, h), D)
    w̃′u′ = ℑxᶠᵃᵃ(i, j, 1, grid, w′u′)

    return streamfunction_amplitude(closure, D, h̃, b̄ₓ, N̄², f, Δs, w̃′u′)
end

@inline function mixed_layer_streamfunctionᶜᶠ(i, j, grid, closure, h, H, w′u′, buoyancy, tracers, coriolis)
    Nz = size(grid, 3)
    H̃ = Hᶜᶠ(i, j, grid, H)
    η = znode(i, j, Nz+1, grid, Center(), Face(), Face())
    zᴴ = η - H̃

    D = zero(grid)
    Σbᵧ = zero(grid)
    ΣN² = zero(grid)

    k = Nz
    zᵗ = η

    while (k ≥ 1) & (zᵗ > zᴴ)
        zᵇ = znode(i, j, k, grid, Center(), Face(), Face())
        δ = zᵗ - max(zᵇ, zᴴ)
        δ = ifelse(peripheral_node(i, j, k, grid, Center(), Face(), Center()), zero(grid), δ)

        D += δ
        Σbᵧ += ∂y_b(i, j, k, grid, buoyancy, tracers) * δ
        ΣN² += ℑyzᵃᶠᶜ(i, j, k, grid, ∂z_b, buoyancy, tracers) * δ

        zᵗ = zᵇ
        k -= 1
    end

    b̄ᵧ = ifelse(D > 0, Σbᵧ / D, zero(grid))
    N̄² = ifelse(D > 0, ΣN² / D, zero(grid))
    f = ℑxᶜᵃᵃ(i, j, 1, grid, fᶠᶠᵃ, coriolis)
    Δs = sqrt(Δxᶜᶠᶜ(i, j, Nz, grid) * Δyᶜᶠᶜ(i, j, Nz, grid))
    h̃ = min(Hᶜᶠ(i, j, grid, h), D)
    w̃′u′ = ℑyᵃᶠᵃ(i, j, 1, grid, w′u′)

    return streamfunction_amplitude(closure, D, h̃, b̄ᵧ, N̄², f, Δs, w̃′u′)
end

@kernel function _compute_mixed_layer_streamfunction!(Ψˣ, Ψʸ, grid, closure, h, H, w′u′, buoyancy, tracers, coriolis)
    i, j = @index(Global, NTuple)

    @inbounds begin
        Ψˣ[i, j, 1] = mixed_layer_streamfunctionᶠᶜ(i, j, grid, closure, h, H, w′u′, buoyancy, tracers, coriolis)
        Ψʸ[i, j, 1] = mixed_layer_streamfunctionᶜᶠ(i, j, grid, closure, h, H, w′u′, buoyancy, tracers, coriolis)
    end
end

# The streamfunction at face i needs H and tracers at i-1, so it is computed everywhere but the first halo point
function mixed_layer_streamfunction_parameters(grid)
    Nx, Ny, _ = size(grid)
    Hx, Hy, _ = halo_size(grid)
    TX, TY, _ = topology(grid)

    ii = ifelse(TX == Flat, 1:Nx, -Hx+2:Nx+Hx)
    jj = ifelse(TY == Flat, 1:Ny, -Hy+2:Ny+Hy)

    return KernelParameters(ii, jj)
end

#####
##### Eddy velocities
#####

# Streamfunction at (F, C, F) and (C, F, F), zero on peripheries and outside the mixed layer
@inline function ψˣᶠᶜᶠ(i, j, k, grid, H, Ψˣ)
    Nz = size(grid, 3)
    H̃ = Hᶠᶜ(i, j, grid, H)
    z = znode(i, j, k, grid, Face(), Center(), Face()) - znode(i, j, Nz+1, grid, Face(), Center(), Face())
    ζ = ifelse(H̃ > 0, z / H̃, - one(grid))
    ψ = @inbounds Ψˣ[i, j, 1] * mixed_layer_structure(ζ)
    return ifelse(peripheral_node(i, j, k, grid, Face(), Center(), Face()), zero(grid), ψ)
end

@inline function ψʸᶜᶠᶠ(i, j, k, grid, H, Ψʸ)
    Nz = size(grid, 3)
    H̃ = Hᶜᶠ(i, j, grid, H)
    z = znode(i, j, k, grid, Center(), Face(), Face()) - znode(i, j, Nz+1, grid, Center(), Face(), Face())
    ζ = ifelse(H̃ > 0, z / H̃, - one(grid))
    ψ = @inbounds Ψʸ[i, j, 1] * mixed_layer_structure(ζ)
    return ifelse(peripheral_node(i, j, k, grid, Center(), Face(), Face()), zero(grid), ψ)
end

@kernel function _compute_mixed_layer_eddy_velocities!(uₑ, vₑ, wₑ, grid, H, Ψˣ, Ψʸ)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        uₑ[i, j, k] = - δzᵃᵃᶜ(i, j, k, grid, ψˣᶠᶜᶠ, H, Ψˣ) * Δz⁻¹ᶠᶜᶜ(i, j, k, grid)
        vₑ[i, j, k] = - δzᵃᵃᶜ(i, j, k, grid, ψʸᶜᶠᶠ, H, Ψʸ) * Δz⁻¹ᶜᶠᶜ(i, j, k, grid)

        wˣ = δxᶜᵃᵃ(i, j, k, grid, Δy_qᶠᶜᶠ, ψˣᶠᶜᶠ, H, Ψˣ)
        wʸ = δyᵃᶜᵃ(i, j, k, grid, Δx_qᶜᶠᶠ, ψʸᶜᶠᶠ, H, Ψʸ)

        wₑ[i, j, k] = (wˣ + wʸ) * Az⁻¹ᶜᶜᶠ(i, j, k, grid)
    end
end

function compute_closure_fields!(closure_fields, closure::MLER, model; parameters = :xyz)
    grid = model.grid
    arch = architecture(grid)
    buoyancy = model.buoyancy.formulation
    tracers = model.tracers
    H = closure_fields.H
    Ψˣ = closure_fields.Ψˣ
    Ψʸ = closure_fields.Ψʸ

    launch!(arch, grid, mixed_layer_streamfunction_parameters(grid),
            _compute_mixed_layer_streamfunction!, Ψˣ, Ψʸ, grid, closure,
            closure_fields.h, H, closure_fields.w′u′, buoyancy, tracers, model.coriolis)

    launch!(arch, grid, parameters, _compute_mixed_layer_eddy_velocities!,
            closure_fields.u, closure_fields.v, closure_fields.w, grid, H, Ψˣ, Ψʸ)

    return nothing
end

#####
##### Show
#####

Base.summary(::MLER) = "MixedLayerEddyRestratification"

function Base.show(io::IO, closure::MLER)
    print(io, summary(closure), ":", '\n',
              "├── frontal_length: ", closure.frontal_length, '\n',
              "├── mixed_layer_energy: ", prettysummary(closure.mixed_layer_energy), '\n',
              "├── reference_density: ", prettysummary(closure.reference_density), '\n',
              "├── mixed_layer_decay_time: ", prettysummary(closure.mixed_layer_decay_time), '\n',
              "├── maximum_grid_spacing: ", prettysummary(closure.maximum_grid_spacing), '\n',
              "└── maximum_eddy_velocity: ", prettysummary(closure.maximum_eddy_velocity))
end
