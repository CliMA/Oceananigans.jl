"""
    QuasiAdamsBashforth2TimeStepper{T, TG} <: AbstractTimeStepper

Holds tendency fields and the parameter `χ` for a modified second-order
Adams-Bashforth timestepping method.
"""
struct QuasiAdamsBashforth2TimeStepper{T, TG} <: AbstractTimeStepper
     χ :: T
    Gⁿ :: TG
    G⁻ :: TG
end

"""
    QuasiAdamsBashforth2TimeStepper(float_type, arch, grid, tracers, χ=0.125;
                              Gⁿ = TendencyFields(arch, grid, tracers),
                              G⁻ = TendencyFields(arch, grid, tracers))

Return an QuasiAdamsBashforth2TimeStepper object with tendency fields on `arch` and
`grid` with AB2 parameter `χ`. The tendency fields can be specified via optional
kwargs.
"""
function QuasiAdamsBashforth2TimeStepper(float_type, arch, grid, velocities, tracers, χ=0.1;
                                         Gⁿ = TendencyFields(arch, grid, tracers),
                                         G⁻ = TendencyFields(arch, grid, tracers))

    return QuasiAdamsBashforth2TimeStepper{float_type, typeof(Gⁿ)}(χ, Gⁿ, G⁻)
end

#####
##### Time steppping
#####

"""
    time_step!(model::IncompressibleModel{<:QuasiAdamsBashforth2TimeStepper}, Δt; euler=false)

Step forward `model` one time step `Δt` with a 2nd-order Adams-Bashforth method and
pressure-correction substep. Setting `euler=true` will take a forward Euler time step.
"""
function time_step!(model::IncompressibleModel{<:QuasiAdamsBashforth2TimeStepper}, Δt; euler=false)
    χ = ifelse(euler, convert(eltype(model.grid), -0.5), model.timestepper.χ)

    # Convert NamedTuples of Fields to NamedTuples of OffsetArrays
    velocities, tracers, pressures, diffusivities, Gⁿ, G⁻ =
        datatuples(model.velocities, model.tracers, model.pressures, model.diffusivities,
                   model.timestepper.Gⁿ, model.timestepper.G⁻)

    precomputations!(diffusivities, pressures, velocities, tracers, model)
    
    calculate_tendencies!(Gⁿ, velocities, tracers, pressures, diffusivities, model)

    # Full step for tracers, fractional step for velocities.
    ab2_step!(velocities, tracers, model.architecture, model.grid, Δt, χ, Gⁿ, G⁻)

    calculate_pressure_correction!(pressures.pNHS, Δt, velocities, model)
    pressure_correct_velocities!(velocities, model.architecture, model.grid, Δt, pressures.pNHS)

    store_tendencies!(G⁻, model.architecture, model.grid, Gⁿ)

    tick!(model.clock, Δt)

    return nothing
end

#####
##### Tracer time stepping and predictor velocity updating
#####

function ab2_step!(U, C, arch, grid, Δt, χ, Gⁿ, G⁻)

    workgroup, worksize = work_layout(grid, :xyz)

    barrier = Event(device(arch))

    step_velocities_kernel! = ab2_step_velocities!(device(arch), workgroup, worksize)
    step_tracer_kernel! = ab2_step_tracer!(device(arch), workgroup, worksize)

    velocities_event = step_velocities_kernel!(U, Δt, χ, Gⁿ, G⁻, dependencies=Event(device(arch)))

    events = [velocities_event]

    for i in 1:length(C)
        @inbounds c = C[i]
        @inbounds Gcⁿ = Gⁿ[i+3]
        @inbounds Gc⁻ = G⁻[i+3]
        event = step_tracer_kernel!(c, Δt, χ, Gcⁿ, Gc⁻, dependencies=barrier)
        push!(events, event)
    end

    wait(device(arch), MultiEvent(Tuple(events)))

    return nothing
end

"""
Time step tracers via

    `c^{n+1} = c^n + Δt ( (3/2 + χ) * Gc^{n} - (1/2 + χ) G^{n-1} )`

"""
@kernel function ab2_step_tracer!(c, Δt, χ::FT, Gcⁿ, Gc⁻) where FT
    i, j, k = @index(Global, NTuple)

    @inbounds c[i, j, k] += Δt * ((FT(1.5) + χ) * Gcⁿ[i, j, k] - (FT(0.5) + χ) * Gc⁻[i, j, k])
end

""" Update predictor velocity field. """
@kernel function ab2_step_velocities!(U, Δt, χ::FT, Gⁿ, G⁻) where FT
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        U.u[i, j, k] += Δt * (   (FT(1.5) + χ) * Gⁿ.u[i, j, k]
                               - (FT(0.5) + χ) * G⁻.u[i, j, k] )

        U.v[i, j, k] += Δt * (   (FT(1.5) + χ) * Gⁿ.v[i, j, k]
                               - (FT(0.5) + χ) * G⁻.v[i, j, k] )

        U.w[i, j, k] += Δt * (   (FT(1.5) + χ) * Gⁿ.w[i, j, k]
                               - (FT(0.5) + χ) * G⁻.w[i, j, k] )
    end
end
