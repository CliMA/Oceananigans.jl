using KernelAbstractions: @kernel, @index
using Oceananigans.Architectures: architecture
using Oceananigans.BoundaryConditions: FieldBoundaryConditions, fill_halo_regions!, regularize_field_boundary_conditions
using Oceananigans.Fields: XFaceField, YFaceField
using Oceananigans.Utils: launch!

"""
    CDScheme(grid; relaxation_time = 10 * 86400)

Coriolis scheme of Adcroft, Hill and Marshall (1999), Mon. Wea. Rev. 127, 1928-1936.
The Coriolis force on the C-grid velocities uses D-grid velocities located at the same points,
`vᴰ` at `u` points and `uᴰ` at `v` points, so that `f × u` is evaluated without spatial averaging.
The D-grid velocities are advanced together with the C-grid velocities at every stage of the
`SplitRungeKuttaTimeStepper` of a `HydrostaticFreeSurfaceModel`: their Coriolis force uses the
co-located C-grid velocities, and every other term is the C-grid tendency averaged to the D-grid points.

Keyword arguments
=================

- `relaxation_time`: time scale [s] over which the D-grid velocities relax toward the averaged C-grid velocities.
  Default: 10 days. `Inf` disables the relaxation.
"""
struct CDScheme{U, V, FT, S}
    uᴰ :: U
    vᴰ :: V
    relaxation_rate :: FT
    state :: S
end

function CDScheme(grid; relaxation_time = 10 * 86400)
    x_velocity_boundary_conditions = regularize_field_boundary_conditions(FieldBoundaryConditions(), grid, :u)
    y_velocity_boundary_conditions = regularize_field_boundary_conditions(FieldBoundaryConditions(), grid, :v)
    uᴰ, uᴰ⁰, uᴰ⁺ = Tuple(YFaceField(grid; boundary_conditions=y_velocity_boundary_conditions) for _ in 1:3)
    vᴰ, vᴰ⁰, vᴰ⁺ = Tuple(XFaceField(grid; boundary_conditions=x_velocity_boundary_conditions) for _ in 1:3)

    return CDScheme(uᴰ, vᴰ, convert(eltype(grid), 1 / relaxation_time), (; uᴰ⁰, vᴰ⁰, uᴰ⁺, vᴰ⁺))
end

Base.summary(::CDScheme) = "CDScheme"

Adapt.adapt_structure(to, scheme::CDScheme) = CDScheme(Adapt.adapt(to, scheme.uᴰ), Adapt.adapt(to, scheme.vᴰ), scheme.relaxation_rate, nothing)

const CDS = AbstractRotation{<:CDScheme}

Oceananigans.prognostic_state(coriolis::CDS) = (; uᴰ = Oceananigans.prognostic_state(coriolis.scheme.uᴰ),
                                                  vᴰ = Oceananigans.prognostic_state(coriolis.scheme.vᴰ))

function Oceananigans.restore_prognostic_state!(coriolis::CDS, from::NamedTuple)
    Oceananigans.restore_prognostic_state!(coriolis.scheme.uᴰ, from.uᴰ)
    Oceananigans.restore_prognostic_state!(coriolis.scheme.vᴰ, from.vᴰ)
    return coriolis
end

@inline x_f_cross_U(i, j, k, grid, coriolis::CDS, U) = @inbounds - ℑyᵃᶜᵃ(i, j, k, grid, fᶠᶠᵃ, coriolis) * coriolis.scheme.vᴰ[i, j, k]
@inline y_f_cross_U(i, j, k, grid, coriolis::CDS, U) = @inbounds + ℑxᶜᵃᵃ(i, j, k, grid, fᶠᶠᵃ, coriolis) * coriolis.scheme.uᴰ[i, j, k]

#####
##### Split Runge-Kutta substep of the D-grid velocities, uᴰᵐ⁺¹ = uᴰ⁰ + Δτ (f vᵐ - (uᴰᵐ - ⟨uᵐ⟩) / τ + ⟨G⟩), in two parts:
##### the Coriolis and relaxation terms before the C-grid substep, the averaged C-grid tendency G without Coriolis after it
#####

reconcile_coriolis!(coriolis, velocities) = nothing
cache_coriolis_fields!(coriolis) = nothing
rk_substep_coriolis!(coriolis, velocities, Δτ) = nothing
add_c_grid_increment!(coriolis, velocities, cached_velocities, Δτ) = nothing

@kernel function _reconcile_coriolis!(uᴰ, vᴰ, grid, u, v)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        uᴰ[i, j, k] = ℑxyᶜᶠᵃ(i, j, k, grid, u) * !peripheral_node(i, j, k, grid, center, face, center)
        vᴰ[i, j, k] = ℑxyᶠᶜᵃ(i, j, k, grid, v) * !peripheral_node(i, j, k, grid, face, center, center)
    end
end

function reconcile_coriolis!(coriolis::CDS, velocities)
    (; uᴰ, vᴰ) = coriolis.scheme
    grid = uᴰ.grid
    launch!(architecture(grid), grid, :xyz, _reconcile_coriolis!, uᴰ, vᴰ, grid, velocities.u, velocities.v)
    fill_halo_regions!((uᴰ, vᴰ))
    return nothing
end

function cache_coriolis_fields!(coriolis::CDS)
    (; uᴰ, vᴰ, state) = coriolis.scheme
    parent(state.uᴰ⁰) .= parent(uᴰ)
    parent(state.vᴰ⁰) .= parent(vᴰ)
    return nothing
end

@kernel function _rk_substep_coriolis!(uᴰ⁺, vᴰ⁺, uᴰ⁰, vᴰ⁰, uᴰ, vᴰ, grid, coriolis, u, v, Δτ)
    i, j, k = @index(Global, NTuple)
    r = coriolis.scheme.relaxation_rate
    @inbounds begin
        uᴰ⁺[i, j, k] = uᴰ⁰[i, j, k] + Δτ * (ℑxᶜᵃᵃ(i, j, k, grid, fᶠᶠᵃ, coriolis) * v[i, j, k] - r * (uᴰ[i, j, k] - ℑxyᶜᶠᵃ(i, j, k, grid, u)))
        vᴰ⁺[i, j, k] = vᴰ⁰[i, j, k] - Δτ * (ℑyᵃᶜᵃ(i, j, k, grid, fᶠᶠᵃ, coriolis) * u[i, j, k] + r * (vᴰ[i, j, k] - ℑxyᶠᶜᵃ(i, j, k, grid, v)))
    end
end

function rk_substep_coriolis!(coriolis::CDS, velocities, Δτ)
    (; uᴰ, vᴰ, state) = coriolis.scheme
    grid = uᴰ.grid
    launch!(architecture(grid), grid, :xyz, _rk_substep_coriolis!, state.uᴰ⁺, state.vᴰ⁺, state.uᴰ⁰, state.vᴰ⁰, uᴰ, vᴰ,
            grid, coriolis, velocities.u, velocities.v, convert(eltype(grid), Δτ))
    return nothing
end

# Realized C-grid increment of the stage minus its Coriolis term, which is evaluated with the D-grid velocities
@inline x_velocity_increment(i, j, k, grid, u, u⁰, vᴰ, coriolis, Δτ) = @inbounds (u[i, j, k] - u⁰[i, j, k] - Δτ * ℑyᵃᶜᵃ(i, j, k, grid, fᶠᶠᵃ, coriolis) * vᴰ[i, j, k]) * !peripheral_node(i, j, k, grid, face, center, center)
@inline y_velocity_increment(i, j, k, grid, v, v⁰, uᴰ, coriolis, Δτ) = @inbounds (v[i, j, k] - v⁰[i, j, k] + Δτ * ℑxᶜᵃᵃ(i, j, k, grid, fᶠᶠᵃ, coriolis) * uᴰ[i, j, k]) * !peripheral_node(i, j, k, grid, center, face, center)

@kernel function _add_c_grid_increment!(uᴰ⁺, vᴰ⁺, uᴰ, vᴰ, grid, coriolis, u, v, u⁰, v⁰, Δτ)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        uᴰ⁺[i, j, k] = (uᴰ⁺[i, j, k] + ℑxyᶜᶠᵃ(i, j, k, grid, x_velocity_increment, u, u⁰, vᴰ, coriolis, Δτ)) * !peripheral_node(i, j, k, grid, center, face, center)
        vᴰ⁺[i, j, k] = (vᴰ⁺[i, j, k] + ℑxyᶠᶜᵃ(i, j, k, grid, y_velocity_increment, v, v⁰, uᴰ, coriolis, Δτ)) * !peripheral_node(i, j, k, grid, face, center, center)
    end
end

function add_c_grid_increment!(coriolis::CDS, velocities, cached_velocities, Δτ)
    (; uᴰ, vᴰ, state) = coriolis.scheme
    u, v = velocities.u, velocities.v
    grid = u.grid

    fill_halo_regions!((u, v))
    launch!(architecture(grid), grid, :xyz, _add_c_grid_increment!, state.uᴰ⁺, state.vᴰ⁺, uᴰ, vᴰ,
            grid, coriolis, u, v, cached_velocities.u, cached_velocities.v, convert(eltype(grid), Δτ))

    parent(uᴰ) .= parent(state.uᴰ⁺)
    parent(vᴰ) .= parent(state.vᴰ⁺)
    fill_halo_regions!((uᴰ, vᴰ))

    return nothing
end
