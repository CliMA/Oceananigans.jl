using Oceananigans.TimeSteppers: _rk3_substep_field!, stage_Δt
import Oceananigans.TimeSteppers: lm_rk3_substep!, seed_lm_pressure!

function seed_lm_pressure!(model::NonhydrostaticModel, Δt)
    compute_pressure_correction!(model, Δt)
    make_pressure_correction!(model, Δt)
    return nothing
end

function lm_rk3_substep!(model::NonhydrostaticModel, Δt, γⁿ, ζⁿ, callbacks,
                         ::Union{Val{1}, Val{2}})
    Δτ   = stage_Δt(Δt, γⁿ, ζⁿ)
    grid = model.grid

    compute_flux_bc_tendencies!(model)
    model_fields = prognostic_fields(model)

    for (i, name) in enumerate(keys(model_fields))
        field = model_fields[name]
        exclude_periphery = i < 4 # We assume that the first 3 fields are velocity / momentum variables
        kernel_args = (field, Δt, γⁿ, ζⁿ, model.timestepper.Gⁿ[name], model.timestepper.G⁻[name])
        launch!(architecture(grid), grid, :xyz, _rk3_substep_field!, kernel_args...; exclude_periphery)

        implicit_step!(field,
                       model.timestepper.implicit_solver,
                       model.closure,
                       model.closure_fields,
                       Val(i-3),
                       model.clock,
                       fields(model),
                       Δτ)
    end

    apply_stale_pressure_correction!(model, Δτ)

    return nothing
end

function lm_rk3_substep!(model::NonhydrostaticModel, Δt, γⁿ, ζⁿ, callbacks, ::Val{3})
    Δτ   = stage_Δt(Δt, γⁿ, ζⁿ)
    grid = model.grid

    compute_flux_bc_tendencies!(model)
    model_fields = prognostic_fields(model)

    for (i, name) in enumerate(keys(model_fields))
        field = model_fields[name]
        exclude_periphery = i < 4
        kernel_args = (field, Δt, γⁿ, ζⁿ, model.timestepper.Gⁿ[name], model.timestepper.G⁻[name])
        launch!(architecture(grid), grid, :xyz, _rk3_substep_field!, kernel_args...; exclude_periphery)

        implicit_step!(field,
                       model.timestepper.implicit_solver,
                       model.closure,
                       model.closure_fields,
                       Val(i-3),
                       model.clock,
                       fields(model),
                       Δτ)
    end

    # Pressure-increment formulation. After the stale projection the velocity is u*³ ≈
    # û³ − Δt ∇φⁿ; the fresh Poisson then yields the increment δp such that the projected
    # velocity satisfies u^{n+1} = û³ − Δt ∇φ^{n+1} with φ^{n+1} = φⁿ + (Δτ/Δt) δp.
    apply_stale_pressure_correction!(model, Δτ)

    parent(model.timestepper.pⁿ) .= parent(model.pressures.pNHS)

    compute_pressure_correction!(model, Δτ)
    make_pressure_correction!(model, Δτ)

    γ = convert(eltype(model.pressures.pNHS), Δτ / Δt)
    parent(model.pressures.pNHS) .= parent(model.timestepper.pⁿ) .+ γ .* parent(model.pressures.pNHS)
    fill_halo_regions!(model.pressures.pNHS)

    return nothing
end

function apply_stale_pressure_correction!(model::NonhydrostaticModel, Δτ)
    foreach(mask_immersed_field!, model.velocities)
    fill_halo_regions!(model.velocities, model.clock, fields(model))

    ϵ = eps(eltype(model.pressures.pNHS))
    Δτ⁺ = max(ϵ, Δτ)
    model.pressures.pNHS .*= Δτ⁺
    fill_halo_regions!(model.pressures.pNHS)
    make_pressure_correction!(model, Δτ)
    return nothing
end
