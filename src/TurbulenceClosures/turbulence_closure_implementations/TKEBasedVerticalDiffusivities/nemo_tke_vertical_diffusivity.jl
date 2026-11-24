#####
##### NEMO's TKE vertical mixing as a flavour of CATKEVerticalDiffusivity
#####
#
# The base scheme of NEMO's zdftke.F90 (Gaspar et al. 1990; Blanke and Delecluse 1993; Madec et al., NEMO ocean engine),
# with the reference namelist (namzdf_tke, namzdf) as defaults:
#
#   ∂t e = ∂z(κᵉ ∂z e) + νᵉ S² − κᶜ N² − Cᴰ e^{3/2} / ℓ,       κᵉ = νᵉ,  νᵉ = max(Cᵏ ℓ √e, νᵇ)
#   κᶜ = max(Pr⁻¹ max(Cᵏ ℓ √e, κᵇ), κᵇ),   Pr⁻¹ = max(1 / Prᵐᵃˣ, Riᶜ / max(Riᶜ, Ri))                (nn_pdl = 1)
#   ℓ = √(2e) / N on faces, then limited so that it grows by at most the distance between faces away from the sea
#       floor (ℓᵐⁱⁿ) and the surface (ℓ₀ = Cᴿ u★², the stress-dependent surface length of ln_mxl0)         (nn_mxl = 2)
#   e = e₀ = max(e₀ᵐⁱⁿ, Cᵉᵇ u★²) at the surface (rn_ebb: wave breaking, kept under ice as in NEMO), e ≥ minimum_tke
#   κᶜ = κᵉᵛᵈ where N² < −10⁻¹² (enhanced vertical diffusion, ln_zdfevd; νᵉᵛᵈ for momentum, off by default: nn_evdm = 0)
#
# The Langmuir-cell and TKE-penetration terms (ln_lc, nn_etau) are not included: NEMO switches them off under sea ice.
# As in NEMO 3.x (GLORYS12), the buoyancy term of the TKE equation uses κᶜ including the enhanced vertical diffusion.
#
# Oceananigans keeps e at cell centres (NEMO at w points): ℓ is computed on faces from e interpolated to the face, the
# dissipation length at centres is the average of the two faces, and the surface value e₀ is imposed on the top cell
# by an implicit relaxation through the half cell above its centre, ∂t e = 2κᵉ (e₀ − e) / Δz², with κᵉ on the face
# below. Everything else (time stepping, the implicit solve, shear production) is CATKE's.
#
# Usage: closure = NEMOTKEVerticalDiffusivity(), or CATKEVerticalDiffusivity(; mixing_length = NEMOMixingLength(),
# turbulent_kinetic_energy_equation = NEMOTKEEquation(), minimum_tke = 1e-6). Vertically implicit time discretization
# only; a single closure (not an array of closures).

using Oceananigans.Utils: KernelParameters

"""
    NEMOMixingLength{FT}

Mixing length and eddy diffusivities of NEMO's TKE scheme (zdftke.F90, `nn_mxl = 2`, `nn_pdl = 1`, `ln_mxl0 = true`),
with NEMO's reference values as defaults:

- `Cᵏ`: eddy diffusivity coefficient, νᵉ = Cᵏ ℓ √e (`rn_ediff` = 0.1)
- `Cᴿ`: surface mixing length ℓ₀ = Cᴿ u★², the von Kármán constant times Stacey's 2 × 10⁵ over g (`ln_mxl0`)
- `ℓᵐⁱⁿ`: smallest mixing length, 10⁻⁶ / (rn_ediff √rn_emin) = 0.01 m
- `Riᶜ`: critical Richardson number of the Prandtl number, 2 / (2 + rn_ediss / rn_ediff) = 2/9
- `Prᵐᵃˣ`: largest Prandtl number (10)
- `νᵇ`, `κᵇ`: background viscosity and diffusivity (`rn_avm0` = 1.2 × 10⁻⁴, `rn_avt0` = 1.2 × 10⁻⁵ m² s⁻¹)
- `κᵉᵛᵈ`: tracer diffusivity where the water column is unstable (`rn_evd` = 100 m² s⁻¹; 0 switches it off)
- `νᵉᵛᵈ`: viscosity where the water column is unstable (0: off, as `nn_evdm = 0`)
"""
Base.@kwdef struct NEMOMixingLength{FT}
    Cᵏ    :: FT = 0.1
    Cᴿ    :: FT = 0.4 * 2e5 / 9.80665
    ℓᵐⁱⁿ  :: FT = 0.01
    Riᶜ   :: FT = 2 / 9
    Prᵐᵃˣ :: FT = 10.0
    νᵇ    :: FT = 1.2e-4
    κᵇ    :: FT = 1.2e-5
    κᵉᵛᵈ  :: FT = 100.0
    νᵉᵛᵈ  :: FT = 0.0
end

"""
    NEMOTKEEquation{FT}

Turbulent kinetic energy equation of NEMO's TKE scheme, with NEMO's reference values as defaults:

- `Cᴰ`: dissipation coefficient, ϵ = Cᴰ e^{3/2} / ℓ (`rn_ediss` = 0.7)
- `Cᵉᵇ`: surface TKE e₀ = Cᵉᵇ u★² (`rn_ebb` = 67.83, wave breaking; Gaspar et al.'s value without waves is 3.75)
- `e₀ᵐⁱⁿ`: smallest surface TKE (`rn_emin0` = 10⁻⁴ m² s⁻²)
"""
Base.@kwdef struct NEMOTKEEquation{FT}
    Cᴰ    :: FT = 0.7
    Cᵉᵇ   :: FT = 67.83
    e₀ᵐⁱⁿ :: FT = 1e-4
end

const NEMOTKEVD = CATKEVerticalDiffusivity{<:Any, <:NEMOMixingLength, <:Any, <:Any, <:NEMOTKEEquation}

for S in (:NEMOMixingLength, :NEMOTKEEquation)
    @eval @inline convert_eltype(::Type{FT}, s::$S) where FT =
        $S{FT}(; Dict(p => getproperty(s, p) for p in propertynames(s))...)

    @eval @inline convert_eltype(::Type{FT}, s::$S{FT}) where FT = s
end

"""
    NEMOTKEVerticalDiffusivity([time_discretization = VerticallyImplicitTimeDiscretization(),
                                FT = Oceananigans.defaults.FloatType;]
                               mixing_length = NEMOMixingLength(),
                               turbulent_kinetic_energy_equation = NEMOTKEEquation(),
                               minimum_tke = 1e-6,
                               kw...)

Return a `CATKEVerticalDiffusivity` that mixes with the base scheme of NEMO's TKE closure (zdftke.F90): mixing
length √(2e)/N limited by the distance to the boundaries, eddy viscosity 0.1 ℓ √e, a Richardson-number-dependent
Prandtl number, the surface value 67.83 u★², and enhanced vertical diffusion of tracers where N² < 0.
`minimum_tke` is NEMO's `rn_emin`. Other keyword arguments go to `CATKEVerticalDiffusivity`.
"""
function NEMOTKEVerticalDiffusivity(time_discretization = VerticallyImplicitTimeDiscretization(),
                                    FT = Oceananigans.defaults.FloatType;
                                    mixing_length = NEMOMixingLength(),
                                    turbulent_kinetic_energy_equation = NEMOTKEEquation(),
                                    minimum_tke = 1e-6,
                                    kw...)

    return CATKEVerticalDiffusivity(time_discretization, FT; mixing_length, turbulent_kinetic_energy_equation,
                                    minimum_tke, kw...)
end

NEMOTKEVerticalDiffusivity(FT::DataType; kw...) =
    NEMOTKEVerticalDiffusivity(VerticallyImplicitTimeDiscretization(), FT; kw...)

#####
##### Closure fields: CATKE's, plus the mixing length on faces and the surface TKE
#####

struct NEMOTKEClosureFields{K, L, M, E, U, KC, LC}
    κu :: K
    κc :: K
    κe :: K
    Le :: L
    ℓ  :: M
    e₀ :: E
    previous_velocities :: U
    _tupled_tracer_diffusivities :: KC
    _tupled_implicit_linear_coefficients :: LC
end

Adapt.adapt_structure(to, fields::NEMOTKEClosureFields) =
    NEMOTKEClosureFields(adapt(to, fields.κu),
                         adapt(to, fields.κc),
                         adapt(to, fields.κe),
                         adapt(to, fields.Le),
                         adapt(to, fields.ℓ),
                         adapt(to, fields.e₀),
                         adapt(to, fields.previous_velocities),
                         adapt(to, fields._tupled_tracer_diffusivities),
                         adapt(to, fields._tupled_implicit_linear_coefficients))

function BoundaryConditions.fill_halo_regions!(fields::NEMOTKEClosureFields, args...; kw...)
    κ = (fields.κu, fields.κc, fields.κe)
    return fill_halo_regions!(κ, args...; kw...)
end

Base.@constprop :aggressive function TurbulenceClosures.build_closure_fields(grid, clock, tracer_names, bcs,
                                                                              closure::NEMOTKEVD)

    default_diffusivity_bcs = (κu = FieldBoundaryConditions(grid, (Center(), Center(), Face())),
                               κc = FieldBoundaryConditions(grid, (Center(), Center(), Face())),
                               κe = FieldBoundaryConditions(grid, (Center(), Center(), Face())))

    bcs = merge(default_diffusivity_bcs, bcs)

    κu = ZFaceField(grid, boundary_conditions=bcs.κu)
    κc = ZFaceField(grid, boundary_conditions=bcs.κc)
    κe = ZFaceField(grid, boundary_conditions=bcs.κe)
    Le = CenterField(grid)
    ℓ  = ZFaceField(grid)
    e₀ = Field{Center, Center, Nothing}(grid)

    u⁻ = XFaceField(grid)
    v⁻ = YFaceField(grid)
    previous_velocities = (; u=u⁻, v=v⁻)

    _tupled_tracer_diffusivities = named_tuple(tracer_names) do name
        Base.@constprop :aggressive
        name === :e ? κe : κc
    end

    _tupled_implicit_linear_coefficients = named_tuple(tracer_names) do name
        Base.@constprop :aggressive
        name === :e ? Le : ZeroField()
    end

    return NEMOTKEClosureFields(κu, κc, κe, Le, ℓ, e₀, previous_velocities,
                                _tupled_tracer_diffusivities,
                                _tupled_implicit_linear_coefficients)
end

function TimeSteppers.reset!(closure_fields, ::NEMOTKEVD)
    fields = (closure_fields.κu, closure_fields.κc, closure_fields.κe, closure_fields.ℓ, closure_fields.e₀,
              closure_fields.previous_velocities...)

    for field in fields
        fill!(field, 0)
    end

    return nothing
end

Oceananigans.prognostic_state(cf::NEMOTKEClosureFields) =
    (previous_velocities = prognostic_state(cf.previous_velocities),
     κu = prognostic_state(cf.κu),
     κc = prognostic_state(cf.κc),
     κe = prognostic_state(cf.κe))

function Oceananigans.restore_prognostic_state!(restored::NEMOTKEClosureFields, from)
    restore_prognostic_state!(restored.previous_velocities, from.previous_velocities)
    restore_prognostic_state!(restored.κu, from.κu)
    restore_prognostic_state!(restored.κc, from.κc)
    restore_prognostic_state!(restored.κe, from.κe)
    return restored
end

Oceananigans.restore_prognostic_state!(::NEMOTKEClosureFields, ::Nothing) = nothing

#####
##### Mixing length (nn_mxl = 2) and surface TKE, one column at a time
#####

# The column kernels cover the same horizontal range as the diffusivity kernel
nemo_column_kernel_parameters(::Symbol) = :xy
nemo_column_kernel_parameters(::KernelParameters{S, O}) where {S, O} = KernelParameters(S[1:2], O[1:2])

function TurbulenceClosures.compute_closure_fields!(closure_fields, closure::NEMOTKEVD, model; parameters = :xyz)
    arch = model.architecture
    grid = model.grid
    velocities = model.velocities
    tracers = buoyancy_tracers(model)
    buoyancy = buoyancy_force(model)
    top_velocity_bcs = (u = velocities.u.boundary_conditions.top,
                        v = velocities.v.boundary_conditions.top)

    launch!(arch, grid, nemo_column_kernel_parameters(parameters),
            compute_nemo_mixing_length!,
            closure_fields.ℓ, closure_fields.e₀, grid, closure,
            tracers, buoyancy, top_velocity_bcs, model.clock, fields(model))

    launch!(arch, grid, parameters,
            compute_CATKE_closure_fields!,
            closure_fields, grid, closure, velocities, tracers, buoyancy)

    return nothing
end

# √(2e)/N on face k; ℓᵐⁱⁿ on boundary faces and below an immersed sea floor
@inline function nemo_stratification_length(i, j, k, grid, ℓᵐⁱⁿ, eᵐⁱⁿ, tracers, buoyancy)
    FT = eltype(grid)
    N² = ∂z_b(i, j, k, grid, buoyancy, tracers)
    N²⁺ = max(N², convert(FT, 1e-20))
    eᶠ = ℑzᵃᵃᶠ(i, j, k, grid, tracers.e)
    e⁺ = max(eᵐⁱⁿ, eᶠ)
    ℓᴺ = max(ℓᵐⁱⁿ, sqrt(2 * e⁺ / N²⁺))
    boundary = peripheral_node(i, j, k, grid, c, c, f) | inactive_node(i, j, k, grid, c, c, f)
    return ifelse(boundary, ℓᵐⁱⁿ, ℓᴺ)
end

@kernel function compute_nemo_mixing_length!(ℓ, e₀, grid, closure, tracers, buoyancy, top_velocity_bcs, clock, model_fields)
    i, j = @index(Global, NTuple)

    closure_ij = getclosure(i, j, closure)
    ℓᵐⁱⁿ = closure_ij.mixing_length.ℓᵐⁱⁿ
    Cᴿ = closure_ij.mixing_length.Cᴿ
    Cᵉᵇ = closure_ij.turbulent_kinetic_energy_equation.Cᵉᵇ
    e₀ᵐⁱⁿ = closure_ij.turbulent_kinetic_energy_equation.e₀ᵐⁱⁿ
    eᵐⁱⁿ = closure_ij.minimum_tke
    Nz = size(grid, 3)

    u★ = friction_velocity(i, j, grid, clock, model_fields, top_velocity_bcs)
    @inbounds e₀[i, j, 1] = max(e₀ᵐⁱⁿ, Cᵉᵇ * u★^2)

    # Up from the sea floor, where ℓ = ℓᵐⁱⁿ: ℓ grows by at most the distance between faces...
    @inbounds ℓ[i, j, 1] = ℓᵐⁱⁿ
    for k in 2:Nz
        ℓᴺ = nemo_stratification_length(i, j, k, grid, ℓᵐⁱⁿ, eᵐⁱⁿ, tracers, buoyancy)
        @inbounds ℓ[i, j, k] = min(ℓᴺ, ℓ[i, j, k-1] + Δzᶜᶜᶜ(i, j, k-1, grid))
    end

    # ... and down from the surface, where ℓ = ℓ₀
    @inbounds ℓ[i, j, Nz+1] = max(ℓᵐⁱⁿ, Cᴿ * u★^2)
    for k in Nz:-1:2
        @inbounds ℓ[i, j, k] = min(ℓ[i, j, k], ℓ[i, j, k+1] + Δzᶜᶜᶜ(i, j, k, grid))
    end
end

#####
##### Diffusivities on faces
#####

@inline function nemo_eddy_diffusivityᶜᶜᶠ(i, j, k, grid, closure, tracers, closure_fields)
    w★ = ℑzᵃᵃᶠ(i, j, k, grid, turbulent_velocityᶜᶜᶜ, closure, tracers.e)
    ℓ = @inbounds closure_fields.ℓ[i, j, k]
    return closure.mixing_length.Cᵏ * ℓ * w★
end

@inline nemo_unstable(i, j, k, grid, buoyancy, tracers) =
    ∂z_b(i, j, k, grid, buoyancy, tracers) < - convert(eltype(grid), 1e-12)

@inline function κuᶜᶜᶠ(i, j, k, grid, closure::NEMOTKEVD, velocities, tracers, buoyancy, closure_fields)
    mixing_length = closure.mixing_length
    K = nemo_eddy_diffusivityᶜᶜᶠ(i, j, k, grid, closure, tracers, closure_fields)
    κu = max(K, mixing_length.νᵇ)
    unstable = nemo_unstable(i, j, k, grid, buoyancy, tracers)
    κu = ifelse(unstable, max(κu, mixing_length.νᵉᵛᵈ), κu)
    κu★ = min(κu, closure.maximum_viscosity)
    FT = eltype(grid)
    return FT(κu★)
end

@inline function κcᶜᶜᶠ(i, j, k, grid, closure::NEMOTKEVD, velocities, tracers, buoyancy, closure_fields)
    mixing_length = closure.mixing_length
    K = nemo_eddy_diffusivityᶜᶜᶠ(i, j, k, grid, closure, tracers, closure_fields)
    κᵇ = mixing_length.κᵇ
    Riᶜ = mixing_length.Riᶜ
    Ri = Riᶜᶜᶠ(i, j, k, grid, velocities, tracers, buoyancy)
    Pr⁻¹ = max(1 / mixing_length.Prᵐᵃˣ, Riᶜ / max(Riᶜ, Ri))
    κc = max(Pr⁻¹ * max(K, κᵇ), κᵇ)
    unstable = nemo_unstable(i, j, k, grid, buoyancy, tracers)
    κc = ifelse(unstable, mixing_length.κᵉᵛᵈ, κc)
    κc★ = min(κc, closure.maximum_tracer_diffusivity)
    FT = eltype(grid)
    return FT(κc★)
end

@inline function κeᶜᶜᶠ(i, j, k, grid, closure::NEMOTKEVD, velocities, tracers, buoyancy, closure_fields)
    K = nemo_eddy_diffusivityᶜᶜᶠ(i, j, k, grid, closure, tracers, closure_fields)
    κe = max(K, closure.mixing_length.νᵇ)
    κe★ = min(κe, closure.maximum_tke_diffusivity)
    FT = eltype(grid)
    return FT(κe★)
end

#####
##### TKE equation
#####

@inline function dissipation_rate(i, j, k, grid, closure::NEMOTKEVD, velocities, tracers, buoyancy, closure_fields)
    Cᴰ = closure.turbulent_kinetic_energy_equation.Cᴰ
    ℓ = ℑzᵃᵃᶜ(i, j, k, grid, closure_fields.ℓ)
    eᵢ = @inbounds tracers.e[i, j, k]
    ω_numerical = 1 / closure.negative_tke_damping_time_scale
    ω_physical = Cᴰ * sqrt(abs(eᵢ)) / ℓ
    return ifelse(eᵢ < 0, ω_numerical, ω_physical)
end

# The surface value e₀ enters as a relaxation of the top cell, implicit in e: ∂t e = τ⁻¹ (e₀ − e), τ⁻¹ = 2κᵉ / Δz²
@inline function fast_tke_tendency(i, j, k, grid, Le, closure::NEMOTKEVD,
                                   next_velocities, previous_velocities,
                                   tracers, buoyancy, closure_fields)

    e = tracers.e
    wb = explicit_buoyancy_flux(i, j, k, grid, closure, next_velocities, tracers, buoyancy, closure_fields)
    wb⁻ = min(zero(grid), wb)
    wb⁺ = max(zero(grid), wb)

    eⁱʲᵏ = @inbounds e[i, j, k]
    eᵐⁱⁿ = closure.minimum_tke
    wb⁻_e = wb⁻ / eⁱʲᵏ * (eⁱʲᵏ > eᵐⁱⁿ)

    at_top = k == size(grid, 3)
    κe = @inbounds closure_fields.κe[i, j, k]
    Δz = Δzᶜᶜᶜ(i, j, k, grid)
    τ⁻¹ = 2 * κe / Δz^2 * at_top
    e₀ = @inbounds closure_fields.e₀[i, j, 1]

    ω = dissipation_rate(i, j, k, grid, closure, next_velocities, tracers, buoyancy, closure_fields)
    active = !inactive_cell(i, j, k, grid)
    @inbounds Le[i, j, k] = (wb⁻_e - ω - τ⁻¹) * active

    u⁺ = next_velocities.u
    v⁺ = next_velocities.v
    uⁿ = previous_velocities.u
    vⁿ = previous_velocities.v
    P = shear_production(i, j, k, grid, closure_fields.κu, uⁿ, u⁺, vⁿ, v⁺)

    return P + wb⁺ + τ⁻¹ * e₀
end

@inline _top_tke_flux(i, j, grid, clock, fields, tke::NEMOTKEEquation, closure, buoyancy,
                      top_tracer_bcs, top_velocity_bcs) = zero(grid)

@kernel function _clip_nemo_tke!(e, grid, eᵐⁱⁿ)
    i, j, k = @index(Global, NTuple)
    active = !inactive_cell(i, j, k, grid)
    @inbounds e[i, j, k] = ifelse(active, max(e[i, j, k], eᵐⁱⁿ), e[i, j, k])
end

function TurbulenceClosures.step_closure_prognostics!(closure_fields, closure::NEMOTKEVD, model, Δt)
    time_step_catke_equation!(model, model.timestepper, Δt)

    # TKE floor (rn_emin)
    launch!(model.architecture, model.grid, :xyz, _clip_nemo_tke!, model.tracers.e, model.grid, closure.minimum_tke)

    u, v, w = model.velocities
    u⁻, v⁻ = closure_fields.previous_velocities
    parent(u⁻) .= parent(u)
    parent(v⁻) .= parent(v)

    return nothing
end

#####
##### Show
#####

function Base.summary(closure::NEMOTKEVD)
    TD = nameof(typeof(TimeSteppers.time_discretization(closure)))
    return string("CATKEVerticalDiffusivity{$TD} with NEMO's TKE scheme")
end

function Base.show(io::IO, closure::NEMOTKEVD)
    ℓ = closure.mixing_length
    tke = closure.turbulent_kinetic_energy_equation
    print(io, summary(closure), '\n',
              "├── minimum_tke: ", prettysummary(closure.minimum_tke), '\n',
              "├── mixing_length: NEMOMixingLength", '\n',
              "│   ├── Cᵏ:    ", prettysummary(ℓ.Cᵏ), '\n',
              "│   ├── Cᴿ:    ", prettysummary(ℓ.Cᴿ), '\n',
              "│   ├── ℓᵐⁱⁿ:  ", prettysummary(ℓ.ℓᵐⁱⁿ), '\n',
              "│   ├── Riᶜ:   ", prettysummary(ℓ.Riᶜ), '\n',
              "│   ├── Prᵐᵃˣ: ", prettysummary(ℓ.Prᵐᵃˣ), '\n',
              "│   ├── νᵇ:    ", prettysummary(ℓ.νᵇ), '\n',
              "│   ├── κᵇ:    ", prettysummary(ℓ.κᵇ), '\n',
              "│   ├── κᵉᵛᵈ:  ", prettysummary(ℓ.κᵉᵛᵈ), '\n',
              "│   └── νᵉᵛᵈ:  ", prettysummary(ℓ.νᵉᵛᵈ), '\n',
              "└── turbulent_kinetic_energy_equation: NEMOTKEEquation", '\n',
              "    ├── Cᴰ:    ", prettysummary(tke.Cᴰ), '\n',
              "    ├── Cᵉᵇ:   ", prettysummary(tke.Cᵉᵇ), '\n',
              "    └── e₀ᵐⁱⁿ: ", prettysummary(tke.e₀ᵐⁱⁿ))
end
