"""
    AdamsBashforthTimeStepper(float_type, arch, grid, tracers, χ=0.125;
                              Gⁿ = TendencyFields(arch, grid, tracers),
                              G⁻ = TendencyFields(arch, grid, tracers))

Return an AdamsBashforthTimeStepper object with tendency fields on `arch` and
`grid` with AB2 parameter `χ`. The tendency fields can be specified via optional
kwargs.
"""
struct AdamsBashforthTimeStepper{T, TG, P} <: AbstractTimeStepper
     χ :: T
    Gⁿ :: TG
    G⁻ :: TG
    predictor_velocities :: P
end

function AdamsBashforthTimeStepper(float_type, arch, grid, velocities, tracers, χ=0.1;
                                   Gⁿ = TendencyFields(arch, grid, tracers),
                                   G⁻ = TendencyFields(arch, grid, tracers))

    u★ = Field{Face, Cell, Cell}(data(velocities.u), grid, velocities.u.boundary_conditions)
    v★ = Field{Cell, Face, Cell}(data(velocities.v), grid, velocities.v.boundary_conditions)
    w★ = Field{Cell, Cell, Face}(data(velocities.w), grid, velocities.w.boundary_conditions)

    U★ = (u=u★, v=v★, w=w★)

    return AdamsBashforthTimeStepper{float_type, typeof(Gⁿ), typeof(U★)}(χ, Gⁿ, G⁻, U★)
end

#####
##### Time steppping
#####

"""
    time_step!(model::IncompressibleModel{<:AdamsBashforthTimeStepper}, Δt; euler=false)

Step forward `model` one time step `Δt` with a 2nd-order Adams-Bashforth method and
pressure-correction substep. Setting `euler=true` will take a forward Euler time step.
"""
function time_step!(model::IncompressibleModel{<:AdamsBashforthTimeStepper}, Δt; euler=false)
    χ = ifelse(euler, convert(eltype(model.grid), -0.5), model.timestepper.χ)

    # Convert NamedTuples of Fields to NamedTuples of OffsetArrays
    velocities, tracers, pressures, diffusivities, Gⁿ, G⁻, predictor_velocities =
        datatuples(model.velocities, model.tracers, model.pressures, model.diffusivities,
                   model.timestepper.Gⁿ, model.timestepper.G⁻, model.timestepper.predictor_velocities)

    time_step_precomputations!(diffusivities, pressures, velocities, tracers, model)
    
    calculate_tendencies!(Gⁿ, velocities, tracers, pressures, diffusivities, model)

    ab2_time_step_tracers!(tracers, model.architecture, model.grid, Δt, χ, Gⁿ, G⁻)

    # Fractional step. Note that predictor velocities share memory space with velocities.
    ab2_update_predictor_velocities!(predictor_velocities, model.architecture, model.grid, Δt, χ, Gⁿ, G⁻)

    calculate_pressure_correction!(pressures.pNHS, Δt, predictor_velocities, model)

    fractional_step_velocities!(velocities, tracers, model.architecture,
                                model.grid, Δt, pressures.pNHS)

    ab2_store_source_terms!(G⁻, model.architecture, model.grid, χ, Gⁿ)

    # Compute w from recontinuity
    fill_halo_regions!(model.velocities, model.architecture, model.clock, state(model))

    compute_w_from_continuity!(model)

    tick!(model.clock, Δt)

    return nothing
end

#####
##### Source term storage
#####

""" Store source terms for `u`, `v`, and `w`. """
@kernel function ab2_store_velocity_source_terms!(G⁻, grid::AbstractGrid{FT}, χ, Gⁿ) where FT
    i, j, k = @index(Global, NTuple)
    @inbounds G⁻.u[i, j, k] = Gⁿ.u[i, j, k]
    @inbounds G⁻.v[i, j, k] = Gⁿ.v[i, j, k]
    @inbounds G⁻.w[i, j, k] = Gⁿ.w[i, j, k]
end

""" Store previous source terms for a tracer before updating them. """
@kernel function ab2_store_tracer_source_term!(Gc⁻, grid::AbstractGrid{FT}, χ, Gcⁿ) where FT
    i, j, k = @index(Global, NTuple)
    @inbounds Gc⁻[i, j, k] = Gcⁿ[i, j, k]
end

""" Store previous source terms before updating them. """
function ab2_store_source_terms!(G⁻, arch, grid, χ, Gⁿ)
    workgroup, worksize = work_layout(grid, :xyz)
    store_velocity_source_terms_kernel! = ab2_store_velocity_source_terms!(device(arch), workgroup, worksize)
    store_tracer_source_term_kernel! = ab2_store_tracer_source_term!(device(arch), workgroup, worksize)

    event = store_velocity_source_terms_kernel!(G⁻, grid, χ, Gⁿ)
    event = nothing

    # Tracer fields
    for i in 4:length(G⁻)
        @inbounds Gc⁻ = G⁻[i]
        @inbounds Gcⁿ = Gⁿ[i]
        event = store_tracer_source_term_kernel!(Gc⁻, grid, χ, Gcⁿ)
    end

    wait(event)

    return nothing
end

#####
##### Tracer time stepping and predictor velocity updating
#####

"""
Time step tracers via

    `c^{n+1} = c^n + Δt ( (3/2 + χ) * Gc^{n} - (1/2 + χ) G^{n-1} )`

"""
@kernel function ab2_time_step_tracer!(c, grid::AbstractGrid{FT}, Δt, χ, Gcⁿ, Gc⁻) where FT
    i, j, k = @index(Global, NTuple)

    @inbounds c[i, j, k] += Δt * ((FT(1.5) + χ) * Gcⁿ[i, j, k] - (FT(0.5) + χ) * Gc⁻[i, j, k])
end

function ab2_time_step_tracers!(C, arch, grid, Δt, χ, Gⁿ, G⁻)
    workgroup, worksize = work_layout(grid, :xyz)
    time_step_tracer_kernel! = ab2_time_step_tracer!(device(arch), workgroup, worksize)
    event = nothing

    for i in 1:length(C)
        @inbounds c = C[i]
        @inbounds Gcⁿ = Gⁿ[i+3]
        @inbounds Gc⁻ = G⁻[i+3]
        event = time_step_tracer_kernel!(c, grid, Δt, χ, Gcⁿ, Gc⁻)
    end

    wait(event)

    return nothing
end

""" Update predictor velocity field. """
@kernel function _ab2_update_predictor_velocities!(U★, grid::AbstractGrid{FT}, Δt, χ, Gⁿ, G⁻) where FT
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        U★.u[i, j, k] += Δt * (   (FT(1.5) + χ) * Gⁿ.u[i, j, k]
                                - (FT(0.5) + χ) * G⁻.u[i, j, k] )

        U★.v[i, j, k] += Δt * (   (FT(1.5) + χ) * Gⁿ.v[i, j, k]
                                - (FT(0.5) + χ) * G⁻.v[i, j, k] )

        U★.w[i, j, k] += Δt * (   (FT(1.5) + χ) * Gⁿ.w[i, j, k]
                                - (FT(0.5) + χ) * G⁻.w[i, j, k] )
    end
end

ab2_update_predictor_velocities!(U★, arch, grid, Δt, χ, Gⁿ, G⁻) =
    launch!(arch, grid, :xyz,  _ab2_update_predictor_velocities!, U★, grid, Δt, χ, Gⁿ, G⁻)
