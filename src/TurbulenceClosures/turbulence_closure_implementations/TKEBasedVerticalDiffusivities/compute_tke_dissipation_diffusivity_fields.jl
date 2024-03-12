#####
##### Diffusivities and diffusivity fields utilities
#####

function DiffusivityFields(grid, tracer_names, bcs, closure::FlavorOfKEpsilon)

    default_diffusivity_bcs = (κᵘ = FieldBoundaryConditions(grid, (Center, Center, Face)),
                               κᶜ = FieldBoundaryConditions(grid, (Center, Center, Face)),
                               κᵏ = FieldBoundaryConditions(grid, (Center, Center, Face)),
                               κᵋ = FieldBoundaryConditions(grid, (Center, Center, Face)))

    bcs = merge(default_diffusivity_bcs, bcs)

    κᵘ = ZFaceField(grid, boundary_conditions=bcs.κᵘ)
    κᶜ = ZFaceField(grid, boundary_conditions=bcs.κᶜ)
    κᵏ = ZFaceField(grid, boundary_conditions=bcs.κᵏ)
    κᵋ = ZFaceField(grid, boundary_conditions=bcs.κᵋ)
    Lᵏ = CenterField(grid)
    Lᵋ = CenterField(grid)

    # Secret tuple for getting tracer diffusivity_fields with tuple[tracer_index]
    tracer_diffusivity_fields = Dict(name => κᶜ for name in tracer_names)
    tracer_diffusivity_fields[:k] = κᵏ
    tracer_diffusivity_fields[:ϵ] = κᵋ

    implicit_linear_coefficients = Dict{Symbol, Any}(name => ZeroField() for name in tracer_names)
    implicit_linear_coefficients[:k] = Lᵏ
    implicit_linear_coefficients[:ϵ] = Lᵋ

    _tupled_tracer_diffusivity_fields   = NamedTuple(name => tracer_diffusivity_fields[name] for name in tracer_names)
    _tupled_implicit_linear_coefficients = NamedTuple(name => implicit_linear_coefficients[name] for name in tracer_names)


    return (; κᵘ, κᶜ, κᵏ, κᵋ, Lᵏ, Lᵋ,
            _tupled_tracer_diffusivity_fields,
            _tupled_implicit_linear_coefficients)
end        

function compute_diffusivity_fields!(diffusivity_fields, closure::FlavorOfKEpsilon, model; parameters = :xyz)

    arch = model.architecture
    grid = model.grid
    velocities = model.velocities
    tracers = model.tracers
    buoyancy = model.buoyancy
    clock = model.clock
    top_tracer_bcs = NamedTuple(c => tracers[c].boundary_conditions.top for c in propertynames(tracers))
    Δt = model.clock.time - diffusivity_fields.previous_compute_time[]
    diffusivity_fields.previous_compute_time[] = model.clock.time

    launch!(arch, grid, parameters,
            compute_k_epsilon_diffusivity_fields!,
            diffusivity_fields, grid, closure, velocities, tracers, buoyancy)

    return nothing
end

@kernel function compute_k_epsilon_diffusivity_fields!(diffusivity_fields, grid, closure::FlavorOfKEpsilon, velocities, tracers, buoyancy)
    i, j, k = @index(Global, NTuple)

    # Ensure this works with "ensembles" of closures, in addition to ordinary single closures
    closure_ij = getclosure(i, j, closure)

    @inbounds begin
        κu★ = κuᶜᶜᶠ(i, j, k, grid, closure_ij, tracers)
        κc★ = κcᶜᶜᶠ(i, j, k, grid, closure_ij, tracers)
        κk★ = κeᶜᶜᶠ(i, j, k, grid, closure_ij, tracers)
        κϵ★ = κeᶜᶜᶠ(i, j, k, grid, closure_ij, tracers)

        on_periphery = peripheral_node(i, j, k, grid, c, c, f)
        within_inactive = inactive_node(i, j, k, grid, c, c, f)
        nan = convert(eltype(grid), NaN)
        κu★ = ifelse(on_periphery, zero(grid), ifelse(within_inactive, nan, κu★))
        κc★ = ifelse(on_periphery, zero(grid), ifelse(within_inactive, nan, κc★))
        κk★ = ifelse(on_periphery, zero(grid), ifelse(within_inactive, nan, κk★))
        κϵ★ = ifelse(on_periphery, zero(grid), ifelse(within_inactive, nan, κϵ★))

        diffusivity_fields.κᵘ[i, j, k] = κu★
        diffusivity_fields.κᶜ[i, j, k] = κc★
        diffusivity_fields.κᵏ[i, j, k] = κk★
        diffusivity_fields.κᵋ[i, j, k] = κϵ★

        # "Patankar trick" for buoyancy production (cf Patankar 1980 or Burchard et al. 2003)
        # If buoyancy flux is a _sink_ of TKE, we treat it implicitly.
        wb = explicit_buoyancy_flux(i, j, k, grid, tracers, buoyancy, diffusivity_fields.κᶜ)
        kⁱʲᵏ = @inbounds tracers.k[i, j, k]
        ϵⁱʲᵏ = @inbounds tracers.ϵ[i, j, k]
        kᵐⁱⁿ = closure_ij.minimum_turbulent_kinetic_energy

        k⁺ = clip(kⁱʲᵏ)
        ϵ⁺ = clip(ϵⁱʲᵏ)

        # See `buoyancy_flux`
        dissipative_buoyancy_flux = (sign(wb) * sign(kⁱʲᵏ) < 0) & (kⁱʲᵏ > kᵐⁱⁿ)
        wb_k = ifelse(dissipative_buoyancy_flux, wb / kⁱʲᵏ, zero(grid))

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
        on_bottom = !inactive_cell(i, j, k, grid) & inactive_cell(i, j, k-1, grid)
        Δz = Δzᶜᶜᶜ(i, j, k, grid)
        Cᵂϵ = closure_ij.turbulent_kinetic_energy_equation.Cᵂϵ
        w★ = sqrt(k⁺)
        div_Jᵏ_k = - on_bottom * Cᵂϵ * w★ / Δz

        # Implicit TKE dissipation
        ω = ϵ⁺ / max(kⁱʲᵏ, kᵐⁱⁿ)
        
        # The interior contributions to the linear implicit term `L` are defined via
        #
        #       ∂t e = Lⁱ e + ⋯,
        #
        # So
        #
        #       Lⁱ e = wb - ϵ
        #            = (wb / e - ω) e,
        #               ↖--------↗
        #                  = Lⁱ
        #
        # where ω = ϵ / e ∼ √e / ℓ.
        
        diffusivity_fields.Lᵏ[i, j, k] = wb_k - ω + div_Jᵏ_k
    end
end

@inline function implicit_linear_coefficient(i, j, k, grid, closure::FlavorOfKEpsilon{<:VITD},
                                             K, ::Val{id}, args...) where id

    L = K._tupled_implicit_linear_coefficients[id]
    return @inbounds L[i, j, k]
end

@inline function κuᶜᶜᶠ(i, j, k, grid, closure::FlavorOfKEpsilon, tracers)
    w★ = ℑzᵃᵃᶠ(i, j, k, grid, turbulent_velocityᶜᶜᶜ, closure, tracers.k)
    e = @inbounds tracers.k[i, j, k]
    ϵ = @inbounds tracers.ϵ[i, j, k]
    ℓu = sqrt(e^3) / ϵ
    κu = ℓu * w★
    κu_max = closure.maximum_viscosity
    return min(κu, κu_max)
end

@inline function κcᶜᶜᶠ(i, j, k, grid, closure::FlavorOfKEpsilon, tracers)
    w★ = ℑzᵃᵃᶠ(i, j, k, grid, turbulent_velocityᶜᶜᶜ, closure, tracers.k)
    e = @inbounds tracers.k[i, j, k]
    ϵ = @inbounds tracers.ϵ[i, j, k]
    ℓc = sqrt(e^3) / ϵ
    κc = ℓc * w★
    κc_max = closure.maximum_tracer_diffusivity
    return min(κc, κc_max)
end

@inline function κkᶜᶜᶠ(i, j, k, grid, closure::FlavorOfKEpsilon, tracers)

    w★ = ℑzᵃᵃᶠ(i, j, k, grid, turbulent_velocityᶜᶜᶜ, closure, tracers.k)
    e = @inbounds tracers.k[i, j, k]
    ϵ = @inbounds tracers.ϵ[i, j, k]
    ℓe = sqrt(e^3) / ϵ
    κe = ℓe * w★
    κe_max = closure.maximum_tke_diffusivity
    return min(κe, κe_max)
end
    
@inline function κϵᶜᶜᶠ(i, j, k, grid, closure::FlavorOfKEpsilon, tracers)

    w★ = ℑzᵃᵃᶠ(i, j, k, grid, turbulent_velocityᶜᶜᶜ, closure, tracers.k)
    e = @inbounds tracers.k[i, j, k]
    ϵ = @inbounds tracers.ϵ[i, j, k]
    ℓϵ = sqrt(k^3) / ϵ
    κϵ = ℓϵ * w★
    κϵ_max = closure.maximum_dissipation_diffusivity
    return min(κϵ, κϵ_max)
end
    
