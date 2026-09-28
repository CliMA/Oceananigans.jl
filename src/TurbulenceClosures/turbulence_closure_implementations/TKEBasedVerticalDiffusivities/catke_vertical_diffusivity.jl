using Oceananigans.Fields: Field
using Oceananigans.Units: minute

struct CATKEVerticalDiffusivity{TD, CL, FT, DT, TKE, R} <: AbstractScalarDiffusivity{TD, VerticalFormulation, 2}
    mixing_length :: CL
    turbulent_kinetic_energy_equation :: TKE
    maximum_tracer_diffusivity :: FT
    maximum_tke_diffusivity :: FT
    maximum_viscosity :: FT
    minimum_tke :: FT
    minimum_convective_buoyancy_flux :: FT
    negative_tke_damping_time_scale :: FT
    tke_time_step :: DT
    penetrative_radiation :: R
end

function CATKEVerticalDiffusivity{TD}(mixing_length::CL,
                                      turbulent_kinetic_energy_equation::TKE,
                                      maximum_tracer_diffusivity::FT,
                                      maximum_tke_diffusivity::FT,
                                      maximum_viscosity::FT,
                                      minimum_tke::FT,
                                      minimum_convective_buoyancy_flux::FT,
                                      negative_tke_damping_time_scale::FT,
                                      tke_time_step::DT,
                                      penetrative_radiation::R) where {TD, CL, FT, DT, TKE, R}

    return CATKEVerticalDiffusivity{TD, CL, FT, DT, TKE, R}(mixing_length,
                                                            turbulent_kinetic_energy_equation,
                                                            maximum_tracer_diffusivity,
                                                            maximum_tke_diffusivity,
                                                            maximum_viscosity,
                                                            minimum_tke,
                                                            minimum_convective_buoyancy_flux,
                                                            negative_tke_damping_time_scale,
                                                            tke_time_step,
                                                            penetrative_radiation)
end

function Adapt.adapt_structure(to, closure::CATKEVerticalDiffusivity{TD}) where TD
    return CATKEVerticalDiffusivity{TD}(closure.mixing_length,
                                        closure.turbulent_kinetic_energy_equation,
                                        closure.maximum_tracer_diffusivity,
                                        closure.maximum_tke_diffusivity,
                                        closure.maximum_viscosity,
                                        closure.minimum_tke,
                                        closure.minimum_convective_buoyancy_flux,
                                        closure.negative_tke_damping_time_scale,
                                        closure.tke_time_step,
                                        adapt(to, closure.penetrative_radiation))
end

CATKEVerticalDiffusivity(FT::DataType; kw...) =
    CATKEVerticalDiffusivity(VerticallyImplicitTimeDiscretization(), FT; kw...)

const CATKEVD{TD} = CATKEVerticalDiffusivity{TD} where TD
const CATKEVDArray{TD} = AbstractArray{<:CATKEVD{TD}} where TD
const FlavorOfCATKE{TD} = Union{CATKEVD{TD}, CATKEVDArray{TD}} where TD

"""
    CATKEVerticalDiffusivity([time_discretization = VerticallyImplicitTimeDiscretization(),
                             FT = Float64;]
                             mixing_length = CATKEMixingLength(),
                             turbulent_kinetic_energy_equation = CATKEEquation(),
                             maximum_tracer_diffusivity = Inf,
                             maximum_tke_diffusivity = Inf,
                             maximum_viscosity = Inf,
                             minimum_tke = 1e-9,
                             minimum_convective_buoyancy_flux = 1e-11,
                             negative_tke_damping_time_scale = 1minute,
                             tke_time_step = nothing,
                             penetrative_radiation = nothing)

Return the `CATKEVerticalDiffusivity` turbulence closure for vertical mixing by
small-scale ocean turbulence based on the prognostic evolution of subgrid
Turbulent Kinetic Energy (TKE).

!!! note "CATKE vertical diffusivity"
    `CATKEVerticalDiffusivity` is a new turbulence closure diffusivity. The default
    values for its free parameters are obtained from calibration against large eddy
    simulations. For more details please refer to [Wagner et al. (2025)](@cite Wagner25catke).

    Use with caution and report any issues with the physics at
    [https://github.com/CliMA/Oceananigans.jl/issues](https://github.com/CliMA/Oceananigans.jl/issues).

Arguments
=========

- `time_discretization`: Either `ExplicitTimeDiscretization()` or `VerticallyImplicitTimeDiscretization()`;
                         default `VerticallyImplicitTimeDiscretization()`.

- `FT`: Float type; default `Float64`.

Keyword arguments
=================

- `mixing_length`: The formulation for mixing length; default: `CATKEMixingLength()`.

- `turbulent_kinetic_energy_equation`: The TKE equation; default: `CATKEEquation()`.

- `maximum_tracer_diffusivity`: Maximum value for tracer diffusivity. CATKE-predicted tracer
                                diffusivities that are larger than `maximum_tracer_diffusivity`
                                are clipped. Default: `Inf`.

- `maximum_tke_diffusivity`: Maximum value for TKE diffusivity. CATKE-predicted diffusivities
                             for TKE that are larger than `maximum_tke_diffusivity` are clipped.
                             Default: `Inf`.

- `maximum_viscosity`: Maximum value for momentum diffusivity. CATKE-predicted momentum diffusivities
                       that are larger than `maximum_viscosity` are clipped. Default: `Inf`.

- `minimum_tke`: Minimum value for the turbulent kinetic energy. `minimum_tke` produces
                 a background tracer diffusivity
  ```math
  κ_{bg} ≈ C^{hi}_c \\frac{e^{\\min}}{N}
  ```
  and background viscosity
  ```math
  ν_{bg} ≈ C^{hi}_u \\frac{e^{\\min}}{N}
  ```
  where ``N`` is the buoyancy frequency and by default, ``C^{hi}_c = 0.098`` and ``C^{hi}_u = 0.242``
  are parameters of `CATKEMixingLength`. This feature may be used to model background mixing by
  internal waves [Wagner et al. (2025)](@cite Wagner25catke). Default: 1e-9.

- `minimum_convective_buoyancy_flux` Minimum value for the convective buoyancy flux. Default: 1e-11.

- `negative_tke_damping_time_scale`: Damping time-scale for spurious negative values of TKE,
                                     typically generated by oscillatory errors associated
                                     with the TKE advection. Default: 1 minute.

- `penetrative_radiation`: Radiation absorbed below the surface. A convective layer of depth ``h`` loses buoyancy at
                           the total surface rate ``Jᵇ``, radiation included, while the part ``Jʳ T(h)`` of the
                           radiative flux ``Jʳ ≥ 0`` leaves through its base, so that the convective velocity scale is
                           ``w★³ = h (Jᵇ - Jʳ T(h)) + 2 Jʳ ∫₀ʰ T(d) dd``, with ``T(d)`` the fraction of ``Jʳ``
                           transmitted to depth ``d``. The column convects when ``Jᵇ > 0``. The radiation type
                           extends `surface_radiative_buoyancy_flux`, `transmitted_fraction`,
                           `transmitted_fraction_derivative` and `transmitted_thickness`. Default: `nothing`.

References
==========

Wagner, G. L., Hillier, A., Constantinou, N. C., Silvestri, S., Souza, A., Burns, K., Hill,
    C., Campin, J.-M., Marshall, J., and Ferrari, R. (2025). Formulation and calibration of CATKE,
    a one-equation parameterization for microscale ocean mixing. J. Adv. Model. Earth Sy., 17, e2024MS004522.
"""
function CATKEVerticalDiffusivity(time_discretization::TD = VerticallyImplicitTimeDiscretization(),
                                  FT = Oceananigans.defaults.FloatType;
                                  mixing_length = CATKEMixingLength(),
                                  turbulent_kinetic_energy_equation = CATKEEquation(),
                                  maximum_tracer_diffusivity = Inf,
                                  maximum_tke_diffusivity = Inf,
                                  maximum_viscosity = Inf,
                                  minimum_tke = 1e-9,
                                  minimum_convective_buoyancy_flux = 1e-11,
                                  negative_tke_damping_time_scale = 1minute,
                                  tke_time_step = nothing,
                                  penetrative_radiation = nothing) where TD

    mixing_length = convert_eltype(FT, mixing_length)
    turbulent_kinetic_energy_equation = convert_eltype(FT, turbulent_kinetic_energy_equation)

    return CATKEVerticalDiffusivity{TD}(mixing_length,
                                        turbulent_kinetic_energy_equation,
                                        convert(FT, maximum_tracer_diffusivity),
                                        convert(FT, maximum_tke_diffusivity),
                                        convert(FT, maximum_viscosity),
                                        convert(FT, minimum_tke),
                                        convert(FT, minimum_convective_buoyancy_flux),
                                        convert(FT, negative_tke_damping_time_scale),
                                        tke_time_step,
                                        penetrative_radiation)
end

Base.@constprop :aggressive function Utils.with_tracers(tracer_names, closure::FlavorOfCATKE)
    :e ∈ tracer_names ||
        throw(ArgumentError("Tracers must contain :e to represent turbulent kinetic energy " *
                            "for `CATKEVerticalDiffusivity`."))

    return closure
end

# Required tracer names for CATKE
closure_required_tracers(::FlavorOfCATKE) = tuple(:e)

# For tuples of closures, we need to know _which_ closure is CATKE.
# Here we take a "simple" approach that sorts the tuple so CATKE is first.
# This is not sustainable though if multiple closures require this.
# The two other possibilities are:
# 1. Recursion to find which closure is CATKE in a compiler-inferrable way
# 2. Store the "CATKE index" inside CATKE via validate_closure.
validate_closure(closure_tuple::Tuple) = Tuple(sort(collect(closure_tuple), lt=catke_first))

catke_first(closure1, catke::FlavorOfCATKE) = false
catke_first(catke::FlavorOfCATKE, closure2) = true
catke_first(closure1, closure2) = false
catke_first(catke1::FlavorOfCATKE, catke2::FlavorOfCATKE) = error("Can't have two CATKEs in one closure tuple.")

#####
##### Diffusivities and diffusivity fields utilities
#####

struct CATKEClosureFields{K, L, J, U, KC, LC}
    κu :: K
    κc :: K
    κe :: K
    Le :: L
    Jᵇ :: J
    Jʳ :: J
    previous_velocities :: U
    _tupled_tracer_diffusivities :: KC
    _tupled_implicit_linear_coefficients :: LC
end

Adapt.adapt_structure(to, catke_closure_fields::CATKEClosureFields) =
    CATKEClosureFields(adapt(to, catke_closure_fields.κu),
                           adapt(to, catke_closure_fields.κc),
                           adapt(to, catke_closure_fields.κe),
                           adapt(to, catke_closure_fields.Le),
                           adapt(to, catke_closure_fields.Jᵇ),
                           adapt(to, catke_closure_fields.Jʳ),
                           adapt(to, catke_closure_fields.previous_velocities),
                           adapt(to, catke_closure_fields._tupled_tracer_diffusivities),
                           adapt(to, catke_closure_fields._tupled_implicit_linear_coefficients))

function BoundaryConditions.fill_halo_regions!(catke_closure_fields::CATKEClosureFields, args...; kw...)
    κ = (catke_closure_fields.κu,
         catke_closure_fields.κc,
         catke_closure_fields.κe)
    return fill_halo_regions!(κ, args...; kw...)
end

Base.@constprop :aggressive function build_closure_fields(grid, clock, tracer_names, bcs, closure::FlavorOfCATKE)

    default_diffusivity_bcs = (κu = FieldBoundaryConditions(grid, (Center(), Center(), Face())),
                               κc = FieldBoundaryConditions(grid, (Center(), Center(), Face())),
                               κe = FieldBoundaryConditions(grid, (Center(), Center(), Face())))

    bcs = merge(default_diffusivity_bcs, bcs)

    κu = ZFaceField(grid, boundary_conditions=bcs.κu)
    κc = ZFaceField(grid, boundary_conditions=bcs.κc)
    κe = ZFaceField(grid, boundary_conditions=bcs.κe)
    Le = CenterField(grid)
    Jᵇ = Field{Center, Center, Nothing}(grid)
    Jʳ = Field{Center, Center, Nothing}(grid)

    # Note: we may be able to avoid using the "previous velocities" in favor of a "fully implicit"
    # discretization of shear production
    u⁻ = XFaceField(grid)
    v⁻ = YFaceField(grid)
    previous_velocities = (; u=u⁻, v=v⁻)

    # Secret tuple for getting tracer diffusivities with tuple[tracer_index]
    _tupled_tracer_diffusivities = named_tuple(tracer_names) do name
        Base.@constprop :aggressive
        name === :e ? κe : κc
    end

    _tupled_implicit_linear_coefficients = named_tuple(tracer_names) do name
        Base.@constprop :aggressive
        name === :e ? Le : ZeroField()
    end

    return CATKEClosureFields(κu, κc, κe, Le, Jᵇ, Jʳ,
                                  previous_velocities,
                                  _tupled_tracer_diffusivities,
                                  _tupled_implicit_linear_coefficients)
end

@inline viscosity_location(::FlavorOfCATKE) = (c, c, f)
@inline diffusivity_location(::FlavorOfCATKE) = (c, c, f)

function reset!(closure_fields, ::FlavorOfCATKE)
    fields = (closure_fields.κu, closure_fields.κc, closure_fields.κe, closure_fields.Jᵇ, closure_fields.Jʳ, closure_fields.previous_velocities...)

    for field in fields
        fill!(field, 0)
    end

    return nothing
end

function step_closure_prognostics!(closure_fields, closure::FlavorOfCATKE, model, Δt)
    arch = model.architecture
    grid = model.grid
    velocities = model.velocities
    tracers = buoyancy_tracers(model)
    buoyancy = buoyancy_force(model)
    clock = model.clock
    top_tracer_bcs = get_top_tracer_bcs(buoyancy, tracers)

    # Step TKE equation with the provided timestep
    time_step_catke_equation!(model, model.timestepper, Δt)

    # Update previous velocities and surface buoyancy flux
    u, v, w = model.velocities
    u⁻, v⁻ = closure_fields.previous_velocities
    parent(u⁻) .= parent(u)
    parent(v⁻) .= parent(v)

    active_cells_map = get_active_cells_map(grid, Val(:xy))

    launch!(arch, grid, :xy,
            compute_average_surface_buoyancy_flux!,
            closure_fields, grid, closure, velocities, tracers, buoyancy, top_tracer_bcs, clock, Δt;
            active_cells_map)

    return nothing
end

function compute_closure_fields!(closure_fields, closure::FlavorOfCATKE, model; parameters = :xyz)
    arch = model.architecture
    grid = model.grid
    velocities = model.velocities
    tracers = buoyancy_tracers(model)
    buoyancy = buoyancy_force(model)

    launch!(arch, grid, parameters,
            compute_CATKE_closure_fields!,
            closure_fields, grid, closure, velocities, tracers, buoyancy)

    return nothing
end

# Jᵇ is the total surface buoyancy flux, radiation included, and Jʳ its radiative part, both averaged over the convective time scale.
@kernel function compute_average_surface_buoyancy_flux!(closure_fields, grid, closure, velocities, tracers, buoyancy, top_tracer_bcs, clock, Δt)
    i, j = @index(Global, NTuple)
    k = grid.Nz

    closure = getclosure(i, j, closure)
    radiation = closure.penetrative_radiation

    model_fields = merge(velocities, tracers)
    Jʳ★ = surface_radiative_buoyancy_flux(i, j, grid, radiation, buoyancy, model_fields)
    Jᵇ★ = top_buoyancy_flux(i, j, grid, buoyancy, top_tracer_bcs, clock, model_fields) - Jʳ★
    ℓᴰ = dissipation_length_scaleᶜᶜᶜ(i, j, k, grid, closure, velocities, tracers, buoyancy, closure_fields)

    Jᵇ = closure_fields.Jᵇ
    Jʳ = closure_fields.Jʳ
    Jᵇᵋ = closure.minimum_convective_buoyancy_flux
    Jᵇᵢⱼ = @inbounds Jᵇ[i, j, 1]
    Jʳᵢⱼ = @inbounds Jʳ[i, j, 1]
    Jᵇ⁺ = max(Jᵇᵋ, Jᵇᵢⱼ + Jʳᵢⱼ, Jᵇ★ + Jʳ★) # selects fastest (dominant) time-scale
    t★ = cbrt(ℓᴰ^2 / Jᵇ⁺)
    ϵ = Δt / t★

    @inbounds begin
        Jᵇ[i, j, 1] = (Jᵇᵢⱼ + ϵ * Jᵇ★) / (1 + ϵ)
        Jʳ[i, j, 1] = (Jʳᵢⱼ + ϵ * Jʳ★) / (1 + ϵ)
    end
end

@kernel function compute_CATKE_closure_fields!(closure_fields, grid, closure::FlavorOfCATKE, velocities, tracers, buoyancy)
    i, j, k = @index(Global, NTuple)

    # Ensure this works with "ensembles" of closures, in addition to ordinary single closures
    closure_ij = getclosure(i, j, closure)

    # Note: we also compute the TKE diffusivity here for diagnostic purposes, even though it
    # is recomputed in time_step_turbulent_kinetic_energy.
    κu★ = κuᶜᶜᶠ(i, j, k, grid, closure_ij, velocities, tracers, buoyancy, closure_fields)
    κc★ = κcᶜᶜᶠ(i, j, k, grid, closure_ij, velocities, tracers, buoyancy, closure_fields)
    κe★ = κeᶜᶜᶠ(i, j, k, grid, closure_ij, velocities, tracers, buoyancy, closure_fields)

    κu★ = mask_diffusivity(i, j, k, grid, κu★)
    κc★ = mask_diffusivity(i, j, k, grid, κc★)
    κe★ = mask_diffusivity(i, j, k, grid, κe★)

    @inbounds begin
        closure_fields.κu[i, j, k] = κu★
        closure_fields.κc[i, j, k] = κc★
        closure_fields.κe[i, j, k] = κe★
    end
end

@inline function κuᶜᶜᶠ(i, j, k, grid, closure, velocities, tracers, buoyancy, closure_fields)
    w★ = ℑzᵃᵃᶠ(i, j, k, grid, turbulent_velocityᶜᶜᶜ, closure, tracers.e)
    ℓu = momentum_mixing_lengthᶜᶜᶠ(i, j, k, grid, closure, velocities, tracers, buoyancy, closure_fields)
    κu = ℓu * w★
    κu_max = closure.maximum_viscosity
    κu★ = min(κu, κu_max)
    FT = eltype(grid)
    return FT(κu★)
end

@inline function κcᶜᶜᶠ(i, j, k, grid, closure, velocities, tracers, buoyancy, closure_fields)
    w★ = ℑzᵃᵃᶠ(i, j, k, grid, turbulent_velocityᶜᶜᶜ, closure, tracers.e)
    ℓc = tracer_mixing_lengthᶜᶜᶠ(i, j, k, grid, closure, velocities, tracers, buoyancy, closure_fields)
    κc = ℓc * w★
    κc_max = closure.maximum_tracer_diffusivity
    κc★ = min(κc, κc_max)
    FT = eltype(grid)
    return FT(κc★)
end

@inline function κeᶜᶜᶠ(i, j, k, grid, closure, velocities, tracers, buoyancy, closure_fields)
    w★ = ℑzᵃᵃᶠ(i, j, k, grid, turbulent_velocityᶜᶜᶜ, closure, tracers.e)
    ℓe = TKE_mixing_lengthᶜᶜᶠ(i, j, k, grid, closure, velocities, tracers, buoyancy, closure_fields)
    κe = ℓe * w★
    κe_max = closure.maximum_tke_diffusivity
    κe★ = min(κe, κe_max)
    FT = eltype(grid)
    return FT(κe★)
end

@inline viscosity(::FlavorOfCATKE, closure_fields) = closure_fields.κu
@inline diffusivity(::FlavorOfCATKE, closure_fields, ::Val{id}) where id = closure_fields._tupled_tracer_diffusivities[id]

#####
##### Show
#####

function Base.summary(closure::CATKEVD)
    TD = nameof(typeof(TimeSteppers.time_discretization(closure)))
    return string("CATKEVerticalDiffusivity{$TD}")
end

function Base.show(io::IO, clo::CATKEVD)
    print(io, summary(clo))
    print(io, '\n')
    print(io, "├── maximum_tracer_diffusivity: ", prettysummary(clo.maximum_tracer_diffusivity), '\n',
              "├── maximum_tke_diffusivity: ", prettysummary(clo.maximum_tke_diffusivity), '\n',
              "├── maximum_viscosity: ", prettysummary(clo.maximum_viscosity), '\n',
              "├── minimum_tke: ", prettysummary(clo.minimum_tke), '\n',
              "├── negative_tke_time_scale: ", prettysummary(clo.negative_tke_damping_time_scale), '\n',
              "├── minimum_convective_buoyancy_flux: ", prettysummary(clo.minimum_convective_buoyancy_flux), '\n',
              "├── tke_time_step: ", prettysummary(clo.tke_time_step), '\n',
              "├── penetrative_radiation: ", prettysummary(clo.penetrative_radiation), '\n',
              "├── mixing_length: ", prettysummary(clo.mixing_length), '\n',
              "│   ├── Cˢ:   ", prettysummary(clo.mixing_length.Cˢ), '\n',
              "│   ├── Cᵇ:   ", prettysummary(clo.mixing_length.Cᵇ), '\n',
              "│   ├── Cʰⁱu: ", prettysummary(clo.mixing_length.Cʰⁱu), '\n',
              "│   ├── Cʰⁱc: ", prettysummary(clo.mixing_length.Cʰⁱc), '\n',
              "│   ├── Cʰⁱe: ", prettysummary(clo.mixing_length.Cʰⁱe), '\n',
              "│   ├── Cˡᵒu: ", prettysummary(clo.mixing_length.Cˡᵒu), '\n',
              "│   ├── Cˡᵒc: ", prettysummary(clo.mixing_length.Cˡᵒc), '\n',
              "│   ├── Cˡᵒe: ", prettysummary(clo.mixing_length.Cˡᵒe), '\n',
              "│   ├── Cᵘⁿu: ", prettysummary(clo.mixing_length.Cᵘⁿu), '\n',
              "│   ├── Cᵘⁿc: ", prettysummary(clo.mixing_length.Cᵘⁿc), '\n',
              "│   ├── Cᵘⁿe: ", prettysummary(clo.mixing_length.Cᵘⁿe), '\n',
              "│   ├── Cᶜu:  ", prettysummary(clo.mixing_length.Cᶜu), '\n',
              "│   ├── Cᶜc:  ", prettysummary(clo.mixing_length.Cᶜc), '\n',
              "│   ├── Cᶜe:  ", prettysummary(clo.mixing_length.Cᶜe), '\n',
              "│   ├── Cᵉc:  ", prettysummary(clo.mixing_length.Cᵉc), '\n',
              "│   ├── Cᵉe:  ", prettysummary(clo.mixing_length.Cᵉe), '\n',
              "│   ├── Cˢᵖ:  ", prettysummary(clo.mixing_length.Cˢᵖ), '\n',
              "│   ├── CRiᵟ: ", prettysummary(clo.mixing_length.CRiᵟ), '\n',
              "│   └── CRi⁰: ", prettysummary(clo.mixing_length.CRi⁰), '\n',
              "└── turbulent_kinetic_energy_equation: ", prettysummary(clo.turbulent_kinetic_energy_equation), '\n',
              "    ├── CʰⁱD: ", prettysummary(clo.turbulent_kinetic_energy_equation.CʰⁱD),  '\n',
              "    ├── CˡᵒD: ", prettysummary(clo.turbulent_kinetic_energy_equation.CˡᵒD),  '\n',
              "    ├── CᵘⁿD: ", prettysummary(clo.turbulent_kinetic_energy_equation.CᵘⁿD),  '\n',
              "    ├── CᶜD:  ", prettysummary(clo.turbulent_kinetic_energy_equation.CᶜD),  '\n',
              "    ├── CᵉD:  ", prettysummary(clo.turbulent_kinetic_energy_equation.CᵉD),  '\n',
              "    ├── Cᵂu★: ", prettysummary(clo.turbulent_kinetic_energy_equation.Cᵂu★), '\n',
              "    ├── CᵂwΔ: ", prettysummary(clo.turbulent_kinetic_energy_equation.CᵂwΔ), '\n',
              "    └── Cᵂϵ:  ", prettysummary(clo.turbulent_kinetic_energy_equation.Cᵂϵ))
end

#####
##### Checkpointing
#####

function prognostic_state(cf::CATKEClosureFields)
    return (previous_velocities = prognostic_state(cf.previous_velocities),
            Jᵇ = prognostic_state(cf.Jᵇ),
            Jʳ = prognostic_state(cf.Jʳ),
            κu = prognostic_state(cf.κu),
            κc = prognostic_state(cf.κc),
            κe = prognostic_state(cf.κe))
end

function restore_prognostic_state!(restored::CATKEClosureFields, from)
    restore_prognostic_state!(restored.previous_velocities, from.previous_velocities)
    restore_prognostic_state!(restored.Jᵇ, from.Jᵇ)
    restore_prognostic_state!(restored.Jʳ, from.Jʳ)
    restore_prognostic_state!(restored.κu, from.κu)
    restore_prognostic_state!(restored.κc, from.κc)
    restore_prognostic_state!(restored.κe, from.κe)
    return restored
end

restore_prognostic_state!(::CATKEClosureFields, ::Nothing) = nothing
