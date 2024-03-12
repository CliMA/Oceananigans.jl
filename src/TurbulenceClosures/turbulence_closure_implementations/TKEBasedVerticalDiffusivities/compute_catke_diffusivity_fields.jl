#####
##### Diffusivities and diffusivity fields utilities
#####

function DiffusivityFields(grid, tracer_names, bcs, closure::FlavorOfCATKE)

    default_diffusivity_bcs = (κᵘ = FieldBoundaryConditions(grid, (Center, Center, Face)),
                               κᶜ = FieldBoundaryConditions(grid, (Center, Center, Face)),
                               κᵉ = FieldBoundaryConditions(grid, (Center, Center, Face)))

    bcs = merge(default_diffusivity_bcs, bcs)

    κᵘ = ZFaceField(grid, boundary_conditions=bcs.κᵘ)
    κᶜ = ZFaceField(grid, boundary_conditions=bcs.κᶜ)
    κᵉ = ZFaceField(grid, boundary_conditions=bcs.κᵉ)
    Lᵉ = CenterField(grid)
    Jᵇ = Field{Center, Center, Nothing}(grid)
    previous_compute_time = Ref(zero(grid))

    # Secret tuple for getting tracer diffusivity_fields with tuple[tracer_index]
    _tupled_tracer_diffusivity_fields    = NamedTuple(name => name === :e ? κᵉ : κᶜ          for name in tracer_names)
    _tupled_implicit_linear_coefficients = NamedTuple(name => name === :e ? Lᵉ : ZeroField() for name in tracer_names)

    return (; κᵘ, κᶜ, κᵉ, Lᵉ, Jᵇ,
            previous_compute_time,
            _tupled_tracer_diffusivity_fields,
            _tupled_implicit_linear_coefficients)
end        

function compute_diffusivities!(diffusivity_fields, closure::FlavorOfCATKE, model; parameters = :xyz)

    arch = model.architecture
    grid = model.grid
    velocities = model.velocities
    tracers = model.tracers
    buoyancy = model.buoyancy
    clock = model.clock
    top_tracer_bcs = NamedTuple(c => tracers[c].boundary_conditions.top for c in propertynames(tracers))
    Δt = model.clock.time - diffusivity_fields.previous_compute_time[]
    diffusivity_fields.previous_compute_time[] = model.clock.time

    launch!(arch, grid, :xy,
            compute_average_surface_buoyancy_flux!,
            diffusivity_fields.Jᵇ, grid, closure, velocities, tracers, buoyancy, top_tracer_bcs, clock, Δt)

    launch!(arch, grid, parameters,
            compute_CATKE_diffusivity_fields!,
            diffusivity_fields, grid, closure, velocities, tracers, buoyancy)

    return nothing
end

@kernel function compute_average_surface_buoyancy_flux!(Jᵇ, grid, closure, velocities, tracers, buoyancy, top_tracer_bcs, clock, Δt)
    i, j = @index(Global, NTuple)
    k = grid.Nz

    closure = getclosure(i, j, closure)
    Jᵇ★ = top_buoyancy_flux(i, j, grid, buoyancy, top_tracer_bcs, clock, merge(velocities, tracers))
    ℓᴰ = dissipation_length_scaleᶜᶜᶜ(i, j, k, grid, closure, velocities, tracers, buoyancy, Jᵇ)

    Jᵇᵋ = closure.minimum_convective_buoyancy_flux
    Jᵇᵢⱼ = @inbounds Jᵇ[i, j, 1]
    Jᵇ⁺ = max(Jᵇᵋ, Jᵇᵢⱼ, Jᵇ★) # selects fastest (dominant) time-scale
    t★ = (ℓᴰ^2 / Jᵇ⁺)^(1/3)
    τ = Δt / t★

    @inbounds Jᵇ[i, j, 1] = (Jᵇᵢⱼ + τ * Jᵇ★) / (1 + τ)
end

@kernel function compute_CATKE_diffusivity_fields!(diffusivity_fields, grid, closure::FlavorOfCATKE, velocities, tracers, buoyancy)
    i, j, k = @index(Global, NTuple)

    # Ensure this works with "ensembles" of closures, in addition to ordinary single closures
    closure_ij = getclosure(i, j, closure)
    Jᵇ = diffusivity_fields.Jᵇ

    @inbounds begin
        κᵘ★ = κuᶜᶜᶠ(i, j, k, grid, closure_ij, velocities, tracers, buoyancy, Jᵇ)
        κᶜ★ = κcᶜᶜᶠ(i, j, k, grid, closure_ij, velocities, tracers, buoyancy, Jᵇ)
        κᵉ★ = κeᶜᶜᶠ(i, j, k, grid, closure_ij, velocities, tracers, buoyancy, Jᵇ)

        on_periphery = peripheral_node(i, j, k, grid, c, c, f)
        within_inactive = inactive_node(i, j, k, grid, c, c, f)
        nan = convert(eltype(grid), NaN)
        κᵘ★ = ifelse(on_periphery, zero(grid), ifelse(within_inactive, nan, κᵘ★))
        κᶜ★ = ifelse(on_periphery, zero(grid), ifelse(within_inactive, nan, κᶜ★))
        κᵉ★ = ifelse(on_periphery, zero(grid), ifelse(within_inactive, nan, κᵉ★))

        diffusivity_fields.κᵘ[i, j, k] = κᵘ★
        diffusivity_fields.κᶜ[i, j, k] = κᶜ★
        diffusivity_fields.κᵉ[i, j, k] = κᵉ★

        # "Patankar trick" for buoyancy production (cf Patankar 1980 or Burchard et al. 2003)
        # If buoyancy flux is a _sink_ of TKE, we treat it implicitly.
        wb = explicit_buoyancy_flux(i, j, k, grid, tracers, buoyancy, diffusivity_fields.κᶜ)
        eⁱʲᵏ = @inbounds tracers.e[i, j, k]
        eᵐⁱⁿ = closure_ij.minimum_turbulent_kinetic_energy

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
        on_bottom = !inactive_cell(i, j, k, grid) & inactive_cell(i, j, k-1, grid)
        Δz = Δzᶜᶜᶜ(i, j, k, grid)
        Cᵂϵ = closure_ij.turbulent_kinetic_energy_equation.Cᵂϵ
        e⁺ = clip(eⁱʲᵏ)
        w★ = sqrt(e⁺)
        div_Jᵉ_e = - on_bottom * Cᵂϵ * w★ / Δz

        # Implicit TKE dissipation
        ω = dissipation_rate(i, j, k, grid, closure_ij, velocities, tracers, buoyancy, diffusivity_fields)
        
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
        
        diffusivity_fields.Lᵉ[i, j, k] = wb_e - ω + div_Jᵉ_e
    end
end

@inline function implicit_linear_coefficient(i, j, k, grid, closure::FlavorOfCATKE{<:VITD}, K, ::Val{id}, args...) where id
    L = K._tupled_implicit_linear_coefficients[id]
    return @inbounds L[i, j, k]
end

@inline function κuᶜᶜᶠ(i, j, k, grid, closure::FlavorOfCATKE, velocities, tracers, buoyancy, surface_buoyancy_flux)
    w★ = ℑzᵃᵃᶠ(i, j, k, grid, turbulent_velocityᶜᶜᶜ, closure, tracers.e)
    ℓᵘ = momentum_mixing_lengthᶜᶜᶠ(i, j, k, grid, closure, velocities, tracers, buoyancy, surface_buoyancy_flux)
    κᵘ = ℓᵘ * w★
    κᵘ_max = closure.maximum_viscosity
    return min(κᵘ, κᵘ_max)
end

@inline function κuᶜᶜᶜ(i, j, k, grid, closure::FlavorOfCATKE, velocities, tracers, buoyancy, surface_buoyancy_flux)
    w★ = turbulent_velocityᶜᶜᶜ(i, j, k, grid, closure, tracers.e)
    ℓᵘ = momentum_mixing_lengthᶜᶜᶜ(i, j, k, grid, closure, velocities, tracers, buoyancy, surface_buoyancy_flux)
    κᵘ = ℓᵘ * w★
    κᵘ_max = closure.maximum_viscosity
    return min(κᵘ, κᵘ_max)
end

@inline function κcᶜᶜᶠ(i, j, k, grid, closure::FlavorOfCATKE, velocities, tracers, buoyancy, surface_buoyancy_flux)
    w★ = ℑzᵃᵃᶠ(i, j, k, grid, turbulent_velocityᶜᶜᶜ, closure, tracers.e)
    ℓᶜ = tracer_mixing_lengthᶜᶜᶠ(i, j, k, grid, closure, velocities, tracers, buoyancy, surface_buoyancy_flux)
    κᶜ = ℓᶜ * w★
    κᶜ_max = closure.maximum_tracer_diffusivity
    return min(κᶜ, κᶜ_max)
end

@inline function κcᶜᶜᶜ(i, j, k, grid, closure::FlavorOfCATKE, velocities, tracers, buoyancy, surface_buoyancy_flux)
    w★ = turbulent_velocityᶜᶜᶜ(i, j, k, grid, closure, tracers.e)
    ℓᶜ = tracer_mixing_lengthᶜᶜᶜ(i, j, k, grid, closure, velocities, tracers, buoyancy, surface_buoyancy_flux)
    κᶜ = ℓᶜ * w★
    κᶜ_max = closure.maximum_tracer_diffusivity
    return min(κᶜ, κᶜ_max)
end

@inline function κeᶜᶜᶠ(i, j, k, grid, closure::FlavorOfCATKE, velocities, tracers, buoyancy, surface_buoyancy_flux)
    w★ = ℑzᵃᵃᶠ(i, j, k, grid, turbulent_velocityᶜᶜᶜ, closure, tracers.e)
    ℓᵉ = TKE_mixing_lengthᶜᶜᶠ(i, j, k, grid, closure, velocities, tracers, buoyancy, surface_buoyancy_flux)
    κᵉ = ℓᵉ * w★
    κᵉ_max = closure.maximum_tke_diffusivity
    return min(κᵉ, κᵉ_max)
end
    
