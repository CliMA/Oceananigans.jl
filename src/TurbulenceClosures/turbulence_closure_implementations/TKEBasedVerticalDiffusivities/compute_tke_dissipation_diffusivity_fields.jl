#####
##### Diffusivities and diffusivity fields utilities
#####

function DiffusivityFields(grid, tracer_names, bcs, closure::FlavorOfKEpsilon)

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

    # Secret tuple for getting tracer diffusivity_fields with tuple[tracer_index]
    tracer_diffusivity_fields = Dict(name => κc for name in tracer_names)
    tracer_diffusivity_fields[:e] = κe
    tracer_diffusivity_fields[:ϵ] = κϵ

    implicit_linear_coefficients = Dict{Symbol, Any}(name => ZeroField() for name in tracer_names)
    implicit_linear_coefficients[:e] = Le
    implicit_linear_coefficients[:ϵ] = Lϵ

    _tupled_tracer_diffusivity_fields = NamedTuple(name => tracer_diffusivity_fields[name] for name in tracer_names)
    _tupled_implicit_linear_coefficients = NamedTuple(name => implicit_linear_coefficients[name] for name in tracer_names)

    return (; κu, κc, κe, κϵ, Le, Lϵ,
            _tupled_tracer_diffusivity_fields,
            _tupled_implicit_linear_coefficients)
end        

@inline viscosity(::FlavorOfKEpsilon, diffusivity_fields) = diffusivity_fields.κu
@inline diffusivity(::FlavorOfKEpsilon, diffusivity_fields, ::Val{id}) where id =
    diffusivity_fields._tupled_tracer_diffusivity_fields[id]

function compute_diffusivities!(diffusivity_fields, closure::FlavorOfKEpsilon, model; parameters = :xyz)

    arch = model.architecture
    grid = model.grid
    velocities = model.velocities
    tracers = model.tracers
    buoyancy = model.buoyancy

    launch!(arch, grid, parameters,
            compute_tke_dissipation_diffusivity_fields!,
            diffusivity_fields, grid, closure, velocities, tracers, buoyancy)

    return nothing
end

@kernel function compute_tke_dissipation_diffusivity_fields!(diffusivity_fields, grid, closure::FlavorOfKEpsilon,
                                                             velocities, tracers, buoyancy)
    i, j, k = @index(Global, NTuple)

    # Ensure this works with "ensembles" of closures, in addition to ordinary single closures
    closure_ij = getclosure(i, j, closure)

    @inbounds begin
        κu★ = κuᶜᶜᶠ(i, j, k, grid, closure_ij, tracers)
        κc★ = κcᶜᶜᶠ(i, j, k, grid, closure_ij, tracers)
        κe★ = κeᶜᶜᶠ(i, j, k, grid, closure_ij, tracers)
        κϵ★ = κeᶜᶜᶠ(i, j, k, grid, closure_ij, tracers)

        on_periphery = peripheral_node(i, j, k, grid, c, c, f)
        within_inactive = inactive_node(i, j, k, grid, c, c, f)
        nan = convert(eltype(grid), NaN)
        κu★ = ifelse(on_periphery, zero(grid), ifelse(within_inactive, nan, κu★))
        κc★ = ifelse(on_periphery, zero(grid), ifelse(within_inactive, nan, κc★))
        κe★ = ifelse(on_periphery, zero(grid), ifelse(within_inactive, nan, κe★))
        κϵ★ = ifelse(on_periphery, zero(grid), ifelse(within_inactive, nan, κϵ★))

        diffusivity_fields.κu[i, j, k] = κu★
        diffusivity_fields.κc[i, j, k] = κc★
        diffusivity_fields.κe[i, j, k] = κe★
        diffusivity_fields.κϵ[i, j, k] = κϵ★

        # "Patankar trick" for buoyancy production (cf Patankar 1980 or Burchard et al. 2003)
        # If buoyancy flux is a _sink_ of TKE, we treat it implicitly.
        wb = explicit_buoyancy_flux(i, j, k, grid, tracers, buoyancy, diffusivity_fields.κc)
        ϵⁱʲᵏ = @inbounds tracers.ϵ[i, j, k]
        eⁱʲᵏ = @inbounds tracers.e[i, j, k]
        eᵐⁱⁿ = closure_ij.minimum_turbulent_kinetic_energy

        e⁺ = clip(eⁱʲᵏ)
        ϵ⁺ = clip(ϵⁱʲᵏ)

        # See `buoyancy_flux`
        dissipative_buoyancy_flux = (sign(wb) * sign(eⁱʲᵏ) < 0) & (eⁱʲᵏ > eᵐⁱⁿ)
        wb_e = ifelse(dissipative_buoyancy_flux, wb / eⁱʲᵏ, zero(grid))

        # Treat the divergence of TKE flux at solid bottoms implicitly.
        # This will damp TKE near boundaries. The bottom-localized TKE flux may be written
        #
        #       ∂t e = - δ(z + h) ∇ ⋅ Jᵉ + ⋯
        #       ∂t e = + δ(z + h) Jᵉ / Δz + ⋯
        #
        # where δ(z + h) is a δ-function that is 0 everywhere except adjacent to the bottom boundary
        # at $z = -h$ and Δz is the grid spacing at the bottom
        #
        # Thus if
        #
        #       Jᵉ ≡ - Cᵂϵ * √e³
        #          = - (Cᵂϵ * √e) e
        #
        # Then the contribution of Jᵉ to the implicit flux is
        #
        #       Lᵂ = - Cᵂϵ * √e / Δz.
        #
        # on_bottom = !inactive_cell(i, j, k, grid) & inactive_cell(i, j, k-1, grid)
        # Δz = Δzᶜᶜᶜ(i, j, k, grid)
        # Cᵂϵ = closure_ij.tke_dissipation_equation.Cᵂϵ
        # w★ = sqrt(e⁺)
        # div_Jᵏ_e = - on_bottom * Cᵂϵ * w★ / Δz

        div_Jᵏ_e = zero(grid)

        # Implicit TKE dissipation
        ω = ϵ⁺ / max(eⁱʲᵏ, eᵐⁱⁿ)
        
        # The interior contributions to the linear implicit term `L` are defined via
        #
        #       ∂t e = Lᵉ e + ⋯,
        #
        # So
        #
        #       Lᵉ e = wb - ϵ
        #            = (wb / e - ω) e,
        #               ↖--------↗
        #                  = Lᵉ
        #
        # where ω = ϵ / e ∼ √e / ℓ.
        
        diffusivity_fields.Le[i, j, k] = wb_e - ω + div_Jᵏ_e

        # The interior contributions to the linear implicit term `L` are defined via
        #
        #       ∂t ϵ = Lᵋ ϵ + ⋯,
        #
        # So
        #
        #       Lᵋ ϵ = Cᴮ * wb * ϵ / k - Cᵋ * ϵ^2
        #            = (wb / k - Cᵋ * ϵ) ϵ,
        #               ↖------------↗
        #                    = Lᵋ
        #
        # where ω = ϵ / e ∼ √e / ℓ.

        ωᴮ = explicit_dissipation_buoyancy_transformation_rate(closure_ij, wb, eⁱʲᵏ)
        ωᴮ = min(zero(ωᴮ), ωᴮ) # implicit contribution

        ωᴰ = explicit_dissipation_destruction_rate(closure_ij, ϵⁱʲᵏ, eⁱʲᵏ)

        diffusivity_fields.Lϵ[i, j, k] = ωᴮ - ωᴰ
    end
end

@inline function implicit_linear_coefficient(i, j, k, grid, closure::FlavorOfKEpsilon{<:VITD},
                                             K, ::Val{id}, args...) where id

    L = K._tupled_implicit_linear_coefficients[id]
    return @inbounds L[i, j, k]
end

@inline function κuᶜᶜᶠ(i, j, k, grid, closure::FlavorOfKEpsilon, tracers)
    ϵ = @inbounds tracers.ϵ[i, j, k]
    w★ = ℑzᵃᵃᶠ(i, j, k, grid, turbulent_velocityᶜᶜᶜ, closure, tracers.e)
    ℓu = w★^3 / ϵ
    κu = ℓu * w★
    κu_max = closure.maximum_viscosity
    return min(κu, κu_max)
end

@inline function κcᶜᶜᶠ(i, j, k, grid, closure::FlavorOfKEpsilon, tracers)
    ϵ = @inbounds tracers.ϵ[i, j, k]
    w★ = ℑzᵃᵃᶠ(i, j, k, grid, turbulent_velocityᶜᶜᶜ, closure, tracers.e)
    ℓc = w★^3 / ϵ
    κc = ℓc * w★
    κc_max = closure.maximum_tracer_diffusivity
    return min(κc, κc_max)
end

@inline function κeᶜᶜᶠ(i, j, k, grid, closure::FlavorOfKEpsilon, tracers)
    ϵ = @inbounds tracers.ϵ[i, j, k]
    w★ = ℑzᵃᵃᶠ(i, j, k, grid, turbulent_velocityᶜᶜᶜ, closure, tracers.e)
    ℓe = w★^3 / ϵ
    κe = ℓe * w★
    κe_max = closure.maximum_tke_diffusivity
    return min(κe, κe_max)
end
    
@inline function κϵᶜᶜᶠ(i, j, k, grid, closure::FlavorOfKEpsilon, tracers)
    ϵ = @inbounds tracers.ϵ[i, j, k]
    w★ = ℑzᵃᵃᶠ(i, j, k, grid, turbulent_velocityᶜᶜᶜ, closure, tracers.e)
    ℓϵ = w★^3 / ϵ
    κϵ = ℓϵ * w★
    κϵ_max = closure.maximum_dissipation_diffusivity
    return min(κϵ, κϵ_max)
end
    
