struct TKEDissipationVerticalDiffusivity{TD, KE, ST, LMIN, FT, DT} <: AbstractScalarDiffusivity{TD, VerticalFormulation, 2}
    tke_dissipation_equations :: KE
    stability_functions :: ST
    minimum_length_scale :: LMIN
    maximum_tracer_diffusivity :: FT
    maximum_tke_diffusivity :: FT
    maximum_dissipation_diffusivity :: FT
    maximum_viscosity :: FT
    minimum_tke :: FT
    minimum_stratification_number_safety_factor :: FT
    negative_tke_damping_time_scale :: FT
    tke_dissipation_time_step :: DT
end

function TKEDissipationVerticalDiffusivity{TD}(tke_dissipation_equations::KE,
                                               stability_functions::ST,
                                               minimum_length_scale :: LMIN,
                                               maximum_tracer_diffusivity::FT,
                                               maximum_tke_diffusivity::FT,
                                               maximum_dissipation_diffusivity::FT,
                                               maximum_viscosity::FT,
                                               minimum_tke::FT,
                                               minimum_stratification_number_safety_factor::FT,
                                               negative_tke_damping_time_scale::FT, 
                                               tke_dissipation_time_step::DT) where {TD, KE, ST, LMIN, FT, DT}

    return TKEDissipationVerticalDiffusivity{TD, KE, ST, LMIN, FT, DT}(tke_dissipation_equations,
                                                                       stability_functions,
                                                                       minimum_length_scale,
                                                                       maximum_tracer_diffusivity,
                                                                       maximum_tke_diffusivity,
                                                                       maximum_dissipation_diffusivity,
                                                                       maximum_viscosity,
                                                                       minimum_tke,
                                                                       minimum_stratification_number_safety_factor,
                                                                       negative_tke_damping_time_scale,
                                                                       tke_dissipation_time_step)
end

TKEDissipationVerticalDiffusivity(FT::DataType; kw...) =
    TKEDissipationVerticalDiffusivity(VerticallyImplicitTimeDiscretization(), FT; kw...)

const TDVD{TD} = TKEDissipationVerticalDiffusivity{TD} where TD
const TDVDArray{TD} = AbstractArray{<:TDVD{TD}} where TD
const FlavorOfTD{TD} = Union{TDVD{TD}, TDVDArray{TD}} where TD

@inline Base.eltype(::TKEDissipationVerticalDiffusivity{<:Any, <:Any, <:Any, <:Any, FT}) where FT = FT

"""
    TKEDissipationVerticalDiffusivity([time_discretization = VerticallyImplicitTimeDiscretization(),
                                      FT = Float64;]
                                      tke_dissipation_equations = TKEDissipationEquations(),
                                      stability_functions = VariableStabilityFunctions(),
                                      minimum_length_scale = StratifiedDisplacementScale(),
                                      maximum_tracer_diffusivity = Inf,
                                      maximum_tke_diffusivity = Inf,
                                      maximum_dissipation_diffusivity = Inf,
                                      maximum_viscosity = Inf,
                                      minimum_tke = 1e-6,
                                      minimum_stratification_number_safety_factor = 0.73,
                                      negative_tke_damping_time_scale = 1minute,
                                      tke_dissipation_time_step = nothing)

Return the `TKEDissipationVerticalDiffusivity` turbulence closure for vertical mixing by
small-scale ocean turbulence based on the prognostic evolution of subgrid
Turbulent Kinetic Energy (TKE).

!!! note "TKEDissipationVerticalDiffusivity"
    `TKEDissipationVerticalDiffusivity` is a new turbulence closure diffusivity. The default
    values for its free parameters are obtained from calibration against large eddy
    simulations. For more details please refer to [Wagner23catke](@cite).

    Use with caution and report any issues with the physics at https://github.com/CliMA/Oceananigans.jl/issues.

Arguments
=========

- `time_discretization`: Either `ExplicitTimeDiscretization()` or `VerticallyImplicitTimeDiscretization()`;
                         default `VerticallyImplicitTimeDiscretization()`.

- `FT`: Float type; default `Float64`.


Keyword arguments
=================

- `maximum_diffusivity`: Maximum value for tracer, momentum, and TKE diffusivities.
                         Used to clip the diffusivity when/if CATKE predicts
                         diffusivities that are too large.
                         Default: `Inf`.

- `minimum_tke`: Minimum value for the turbulent kinetic energy.
                 Can be used to model the presence "background" TKE
                 levels due to, for example, mixing by breaking internal waves.
                 Default: 1e-9.

- `negative_tke_damping_time_scale`: Damping time-scale for spurious negative values of TKE,
                                     typically generated by oscillatory errors associated
                                     with TKE advection.
                                     Default: 1 minute.

Note that for numerical stability, it is recommended to either have a relative short
`negative_turbulent_kinetic_energy_damping_time_scale` or a reasonable
`minimum_turbulent_kinetic_energy`, or both.
"""
function TKEDissipationVerticalDiffusivity(time_discretization::TD = VerticallyImplicitTimeDiscretization(),
                                           FT = Float64;
                                           tke_dissipation_equations = TKEDissipationEquations(),
                                           stability_functions = VariableStabilityFunctions(),
                                           minimum_length_scale = StratifiedDisplacementScale(),
                                           maximum_tracer_diffusivity = Inf,
                                           maximum_tke_diffusivity = Inf,
                                           maximum_dissipation_diffusivity = Inf,
                                           maximum_viscosity = Inf,
                                           minimum_tke = 1e-6,
                                           minimum_stratification_number_safety_factor = 0.73,
                                           negative_tke_damping_time_scale = 1minute,
                                           tke_dissipation_time_step = nothing) where TD

    stability_functions = convert_eltype(FT, stability_functions)

    return TKEDissipationVerticalDiffusivity{TD}(tke_dissipation_equations,
                                                 stability_functions,
                                                 minimum_length_scale,
                                                 convert(FT, maximum_tracer_diffusivity),
                                                 convert(FT, maximum_tke_diffusivity),
                                                 convert(FT, maximum_dissipation_diffusivity),
                                                 convert(FT, maximum_viscosity),
                                                 convert(FT, minimum_tke),
                                                 convert(FT, minimum_stratification_number_safety_factor),
                                                 convert(FT, negative_tke_damping_time_scale),
                                                 tke_dissipation_time_step)
end

function with_tracers(tracer_names, closure::FlavorOfTD)
    :e ∈ tracer_names && :ϵ ∈ tracer_names ||
        throw(ArgumentError("Tracers must contain :e and :ϵ to represent turbulent kinetic energy " *
                            "for `TKEDissipationVerticalDiffusivity`."))

    return closure
end

#####
##### Stratified displacement length scale limiter
#####

Base.@kwdef struct StratifiedDisplacementScale{FT}
    Cᴺ :: FT = 0.75
    minimum_buoyancy_frequency :: FT = 1e-14
end

#####
##### Diffusivities and diffusivity fields utilities
#####

function DiffusivityFields(grid, tracer_names, bcs, closure::FlavorOfTD)

    default_diffusivity_bcs = (κu = FieldBoundaryConditions(grid, (Center, Center, Face)),
                               κc = FieldBoundaryConditions(grid, (Center, Center, Face)),
                               κe = FieldBoundaryConditions(grid, (Center, Center, Face)),
                               κϵ = FieldBoundaryConditions(grid, (Center, Center, Face)))

    bcs = merge(default_diffusivity_bcs, bcs)

    κu = ZFaceField(grid, boundary_conditions=bcs.κu)
    κc = ZFaceField(grid, boundary_conditions=bcs.κc)
    κe = ZFaceField(grid, boundary_conditions=bcs.κe)
    κϵ = ZFaceField(grid, boundary_conditions=bcs.κϵ)
    Le = CenterField(grid)
    Lϵ = CenterField(grid)

    # Note: we may be able to avoid using the "previous velocities" in favor of a "fully implicit"
    # discretization of shear production
    u⁻ = XFaceField(grid)
    v⁻ = YFaceField(grid)
    previous_velocities = (; u=u⁻, v=v⁻)

    # Secret tuple for getting tracer diffusivities with tuple[tracer_index]
    _tupled_tracer_diffusivities = Dict{Symbol, Any}(name => κc for name in tracer_names)
    _tupled_tracer_diffusivities[:e] = κe
    _tupled_tracer_diffusivities[:ϵ] = κϵ
    _tupled_tracer_diffusivities = NamedTuple(name => _tupled_tracer_diffusivities[name]
                                              for name in tracer_names)

    _tupled_implicit_linear_coefficients = Dict{Symbol, Any}(name => ZeroField() for name in tracer_names)
    _tupled_implicit_linear_coefficients[:e] = Le
    _tupled_implicit_linear_coefficients[:ϵ] = Lϵ
    _tupled_implicit_linear_coefficients = NamedTuple(name => _tupled_implicit_linear_coefficients[name]
                                                      for name in tracer_names)

    return (; κu, κc, κe, κϵ, Le, Lϵ, previous_velocities,
            _tupled_tracer_diffusivities, _tupled_implicit_linear_coefficients)
end        

@inline viscosity_location(::FlavorOfTD) = (c, c, f)
@inline diffusivity_location(::FlavorOfTD) = (c, c, f)

function compute_diffusivities!(diffusivities, closure::FlavorOfTD, model; parameters = :xyz)

    arch = model.architecture
    grid = model.grid
    velocities = model.velocities
    tracers = model.tracers
    buoyancy = model.buoyancy
    clock = model.clock
    top_tracer_bcs = NamedTuple(c => tracers[c].boundary_conditions.top for c in propertynames(tracers))

    if isfinite(model.clock.last_Δt) # Check that we have taken a valid time-step first.
        # Compute e at the current time:
        #   * update tendency Gⁿ using current and previous velocity field
        #   * use tridiagonal solve to take an implicit step
        time_step_tke_dissipation_equations!(model)
    end

    # Update "previous velocities"
    u, v, w = model.velocities
    u⁻, v⁻ = diffusivities.previous_velocities
    parent(u⁻) .= parent(u)
    parent(v⁻) .= parent(v)

    launch!(arch, grid, parameters,
            compute_TKEDissipation_diffusivities!,
            diffusivities, grid, closure, velocities, tracers, buoyancy)

    return nothing
end

@kernel function compute_TKEDissipation_diffusivities!(diffusivities, grid, closure::FlavorOfTD,
                                                       velocities, tracers, buoyancy)
    i, j, k = @index(Global, NTuple)

    # Ensure this works with "ensembles" of closures, in addition to ordinary single closures
    closure_ij = getclosure(i, j, closure)

    # Note: we also compute the TKE diffusivity here for diagnostic purposes, even though it
    # is recomputed in time_step_turbulent_kinetic_energy.
    κu★ = κuᶜᶜᶠ(i, j, k, grid, closure_ij, velocities, tracers, buoyancy)
    κc★ = κcᶜᶜᶠ(i, j, k, grid, closure_ij, velocities, tracers, buoyancy)
    κe★ = κeᶜᶜᶠ(i, j, k, grid, closure_ij, velocities, tracers, buoyancy)
    κϵ★ = κϵᶜᶜᶠ(i, j, k, grid, closure_ij, velocities, tracers, buoyancy)

    κu★ = mask_diffusivity(i, j, k, grid, κu★)
    κc★ = mask_diffusivity(i, j, k, grid, κc★)
    κe★ = mask_diffusivity(i, j, k, grid, κe★)
    κϵ★ = mask_diffusivity(i, j, k, grid, κϵ★)

    @inbounds begin
        diffusivities.κu[i, j, k] = κu★
        diffusivities.κc[i, j, k] = κc★
        diffusivities.κe[i, j, k] = κe★
        diffusivities.κϵ[i, j, k] = κϵ★
    end
end

@inline function turbulent_kinetic_energyᶜᶜᶜ(i, j, k, grid, closure, tracers)
    eᵐⁱⁿ = closure.minimum_tke
    eⁱʲᵏ = @inbounds tracers.e[i, j, k]
    return max(eᵐⁱⁿ, eⁱʲᵏ)
end

@inline max_a_b(i, j, k, grid, a::Number, b, args...) = max(a, b(i, j, k, grid, args...))

@inline maximum_dissipation(i, j, k, grid, closure, tracers, buoyancy) = convert(eltype(grid), Inf)

@inline function minimum_dissipation(i, j, k, grid, closure, tracers, buoyancy)
    FT = eltype(grid)

    N²min = closure.minimum_length_scale.minimum_buoyancy_frequency
    N²⁺ = ℑbzᵃᵃᶜ(i, j, k, grid, max_a_b, N²min, ∂z_b, buoyancy, tracers)

    Cᴺ = closure.minimum_length_scale.Cᴺ
    e★ = turbulent_kinetic_energyᶜᶜᶜ(i, j, k, grid, closure, tracers)
    ℓst = Cᴺ * sqrt(e★ / N²⁺)

    𝕊u₀ = closure.stability_functions.𝕊u₀
    ℓmin = min(grid.Lz, ℓst)
    ϵmin = 𝕊u₀^3 * sqrt(e★)^3 / ℓmin

    another_ϵmin = convert(FT, 1e-12)
    return max(another_ϵmin, ϵmin)
end

@inline function dissipationᶜᶜᶜ(i, j, k, grid, closure, tracers, buoyancy)
    ϵᵐⁱⁿ = minimum_dissipation(i, j, k, grid, closure, tracers, buoyancy)
    ϵᵐᵃˣ = maximum_dissipation(i, j, k, grid, closure, tracers, buoyancy)
    ϵⁱʲᵏ = @inbounds tracers.ϵ[i, j, k]
    return clamp(ϵⁱʲᵏ, ϵᵐⁱⁿ, ϵᵐᵃˣ)
end

@inline function κuᶜᶜᶠ(i, j, k, grid, closure::TDVD, velocities, tracers, buoyancy)
    e² = ℑzᵃᵃᶠ(i, j, k, grid, ϕ², turbulent_kinetic_energyᶜᶜᶜ, closure, tracers)
    ϵ  = ℑzᵃᵃᶠ(i, j, k, grid, dissipationᶜᶜᶜ, closure, tracers, buoyancy)
    𝕊u = momentum_stability_functionᶜᶜᶠ(i, j, k, grid, closure, velocities, tracers, buoyancy)
    κu = 𝕊u * e² / ϵ
    κu_max = closure.maximum_viscosity
    return min(κu, κu_max)
end

@inline function κcᶜᶜᶠ(i, j, k, grid, closure::TDVD, velocities, tracers, buoyancy)
    e² = ℑzᵃᵃᶠ(i, j, k, grid, ϕ², turbulent_kinetic_energyᶜᶜᶜ, closure, tracers)
    ϵ  = ℑzᵃᵃᶠ(i, j, k, grid, dissipationᶜᶜᶜ, closure, tracers, buoyancy)
    𝕊c = tracer_stability_functionᶜᶜᶠ(i, j, k, grid, closure, velocities, tracers, buoyancy)
    κc = 𝕊c * e² / ϵ
    κc_max = closure.maximum_tracer_diffusivity
    return min(κc, κc_max)
end

@inline function κeᶜᶜᶠ(i, j, k, grid, closure::TDVD, velocities, tracers, buoyancy)
    e² = ℑzᵃᵃᶠ(i, j, k, grid, ϕ², turbulent_kinetic_energyᶜᶜᶜ, closure, tracers)
    ϵ  = ℑzᵃᵃᶠ(i, j, k, grid, dissipationᶜᶜᶜ, closure, tracers, buoyancy)
    𝕊e = tke_stability_functionᶜᶜᶠ(i, j, k, grid, closure, velocities, tracers, buoyancy)
    κe = 𝕊e * e² / ϵ
    κe_max = closure.maximum_tke_diffusivity
    return min(κe, κe_max)
end

@inline function κϵᶜᶜᶠ(i, j, k, grid, closure::TDVD, velocities, tracers, buoyancy)
    e² = ℑzᵃᵃᶠ(i, j, k, grid, ϕ², turbulent_kinetic_energyᶜᶜᶜ, closure, tracers)
    ϵ  = ℑzᵃᵃᶠ(i, j, k, grid, dissipationᶜᶜᶜ, closure, tracers, buoyancy)
    𝕊ϵ = dissipation_stability_functionᶜᶜᶠ(i, j, k, grid, closure, velocities, tracers, buoyancy)
    κϵ = 𝕊ϵ * e² / ϵ
    κϵ_max = closure.maximum_dissipation_diffusivity
    return min(κϵ, κϵ_max)
end

@inline viscosity(::FlavorOfTD, diffusivities) = diffusivities.κu
@inline diffusivity(::FlavorOfTD, diffusivities, ::Val{id}) where id = diffusivities._tupled_tracer_diffusivities[id]
    
#####
##### Show
#####

function Base.summary(closure::TDVD)
    TD = nameof(typeof(time_discretization(closure)))
    return string("TKEDissipationVerticalDiffusivity{$TD}")
end

function Base.show(io::IO, clo::TDVD)
    print(io, summary(clo))
    print(io, '\n')
    print(io, "├── maximum_tracer_diffusivity: ", prettysummary(clo.maximum_tracer_diffusivity), '\n',
              "├── maximum_tke_diffusivity: ", prettysummary(clo.maximum_tke_diffusivity), '\n',
              "├── maximum_dissipation_diffusivity: ", prettysummary(clo.maximum_dissipation_diffusivity), '\n',
              "├── maximum_viscosity: ", prettysummary(clo.maximum_viscosity), '\n',
              "├── minimum_tke: ", prettysummary(clo.minimum_tke), '\n',
              "├── negative_tke_damping_time_scale: ", prettysummary(clo.negative_tke_damping_time_scale), '\n',
              "├── tke_dissipation_time_step: ", prettysummary(clo.tke_dissipation_time_step), '\n',
              "├── tke_dissipation_equations: ", prettysummary(clo.tke_dissipation_equations), '\n',
              "│   ├── Cᵋϵ: ", prettysummary(clo.tke_dissipation_equations.Cᵋϵ),  '\n',
              "│   ├── Cᴾϵ: ", prettysummary(clo.tke_dissipation_equations.Cᴾϵ),  '\n',
              "│   ├── Cᵇϵ: ", prettysummary(clo.tke_dissipation_equations.Cᵇϵ),  '\n',
              "│   ├── Cᵂu★: ", prettysummary(clo.tke_dissipation_equations.Cᵂu★), '\n',
              "│   └── CᵂwΔ: ", prettysummary(clo.tke_dissipation_equations.CᵂwΔ), '\n')
    print(io, "└── ", summarize_stability_functions(clo.stability_functions), "", "    ")
end

