#####
##### Pipelined RK3: start halo communication early, overlap with pressure solver
#####
##### The key idea: instead of waiting until update_state! to start halo comms,
##### begin them right after the field data is finalized:
#####   - Tracer halos: start after _rk3_substep_field! (no pressure correction)
#####   - Velocity halos: start after make_pressure_correction!
#####
##### This overlaps NCCL transfers with:
#####   - Pressure solver FFTs and tridiag (during tracer halos)
#####   - cache_previous_tendencies! and update_state! setup (during velocity halos)
#####   - Interior tendency computation (both)
#####
##### Timeline comparison (one RK3 substage):
#####
##### Current:
#####   [rk3_substep: advance + pressure solve] → [update_state: fill_halo → tendencies]
#####   |← no overlap with NCCL →|                 |← overlap →|
#####
##### Pipelined:
#####   [advance tracers] → [fill_halo tracers async] → [advance vel + pressure solve] → [fill_halo vel async] → [tendencies]
#####                        |← tracer NCCL overlaps with pressure solve →|               |← vel NCCL overlaps →|
#####

using Oceananigans.TimeSteppers: _rk3_substep_field!, stage_Δt, implicit_step!
using Oceananigans.Models.NonhydrostaticModels: compute_pressure_correction!,
                                                 make_pressure_correction!,
                                                 compute_flux_bc_tendencies!
using Oceananigans.Models: prognostic_fields
import Oceananigans.TimeSteppers: rk3_substep!

const NCCLNonhydrostaticModel = Oceananigans.Models.NonhydrostaticModels.NonhydrostaticModel{
    <:Any, <:Any, <:NCCLDistributedArchitecture, <:NCCLDistributedGrid}

function rk3_substep!(model::NCCLNonhydrostaticModel, Δt, γⁿ, ζⁿ, callbacks)
    Δτ = stage_Δt(Δt, γⁿ, ζⁿ)
    grid = model.grid
    arch = DC.architecture(grid)

    compute_flux_bc_tendencies!(model)
    model_fields = prognostic_fields(model)
    field_names = keys(model_fields)

    #=
    ──── Phase 1: Advance tracers and start their halo communication ────
    Tracers don't need pressure correction, so they're final after _rk3_substep_field!.
    Starting their halos here lets them overlap with the velocity step + pressure solve.
    =#
    for (i, name) in enumerate(field_names)
        i < 4 && continue  # skip velocities
        field = model_fields[name]
        launch!(arch, grid, :xyz, _rk3_substep_field!,
                field, Δt, γⁿ, ζⁿ, model.timestepper.Gⁿ[name], model.timestepper.G⁻[name])

        implicit_step!(field, model.timestepper.implicit_solver,
                       model.closure, model.closure_fields,
                       Val(i-3), model.clock, Oceananigans.fields(model), Δτ)
    end

    # Start tracer halos on comm_stream (non-blocking)
    for (i, name) in enumerate(field_names)
        i < 4 && continue
        Oceananigans.BoundaryConditions.fill_halo_regions!(
            model_fields[name], model.clock, Oceananigans.fields(model); async=true)
    end

    #=
    ──── Phase 2: Advance velocities + pressure correction ────
    While tracer halos transfer on comm_stream, we advance velocities,
    solve the pressure Poisson equation, and apply pressure correction.
    =#
    for (i, name) in enumerate(field_names)
        i >= 4 && continue  # skip tracers
        field = model_fields[name]
        launch!(arch, grid, :xyz, _rk3_substep_field!,
                field, Δt, γⁿ, ζⁿ, model.timestepper.Gⁿ[name], model.timestepper.G⁻[name];
                exclude_periphery=true)

        implicit_step!(field, model.timestepper.implicit_solver,
                       model.closure, model.closure_fields,
                       Val(i-3), model.clock, Oceananigans.fields(model), Δτ)
    end

    compute_pressure_correction!(model, Δτ)
    make_pressure_correction!(model, Δτ)

    #=
    ──── Phase 3: Start velocity halo communication ────
    Velocities are now finalized (pressure-corrected). Start their halos
    on comm_stream, overlapping with cache_previous_tendencies! and
    the beginning of update_state!.
    =#
    for (i, name) in enumerate(field_names)
        i >= 4 && continue
        Oceananigans.BoundaryConditions.fill_halo_regions!(
            model_fields[name], model.clock, Oceananigans.fields(model); async=true)
    end

    return nothing
end

#=
──── Modified update_state! that skips fill_halo_regions! when halos are pending ────

When the pipelined rk3_substep! has already started halo communication via async
fill_halo_regions!, the update_state! should NOT re-start them. We detect this
by checking if pending_unpacks is non-empty.
=#

using Oceananigans.ImmersedBoundaries: mask_immersed_field!
using Oceananigans.Models: update_model_field_time_series!
using Oceananigans.BoundaryConditions: update_boundary_conditions!, fill_halo_regions!
using Oceananigans.Models.NonhydrostaticModels: compute_auxiliaries!

import Oceananigans.Models.NonhydrostaticModels: update_state!

function update_state!(model::NCCLNonhydrostaticModel, callbacks=[])
    foreach(model.tracers) do tracer
        mask_immersed_field!(tracer)
    end

    update_model_field_time_series!(model, model.clock)
    update_boundary_conditions!(Oceananigans.fields(model), model)

    # Skip fill_halo_regions! if halos were already started by pipelined rk3_substep!
    if isempty(pending_unpacks)
        fill_halo_regions!(merge(model.velocities, model.tracers), model.clock,
                           Oceananigans.fields(model); fill_open_bcs=false, async=true)
    end

    for aux_field in model.auxiliary_fields
        compute!(aux_field)
    end

    compute_auxiliaries!(model)

    fill_halo_regions!(model.closure_fields; only_local_halos=true)
    fill_halo_regions!(model.pressures.pHY′; only_local_halos=true)

    for callback in callbacks
        callback.callsite isa Oceananigans.UpdateStateCallsite && callback(model)
    end

    Oceananigans.Models.NonhydrostaticModels.compute_tendencies!(model, callbacks)
    Oceananigans.Biogeochemistry.update_biogeochemical_state!(model.biogeochemistry, model)

    return nothing
end
