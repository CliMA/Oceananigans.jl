using Oceananigans.Advection: FluxFormSemiLagrangian, flux_form_semi_lagrangian_step!, maximum_courant_number,
                              with_workspace, static_volume, initial_stretching
using Oceananigans.Fields: CenterField, XFaceField, YFaceField, Field
using Oceananigans.Grids: halo_size, topology, Flat, Center
using Oceananigans.Operators: flux_div_xyᶜᶜᶜ
using Oceananigans.TurbulenceClosures: closure_required_tracers

#####
##### Horizontal flux-form semi-Lagrangian advection of the tracers that use `FluxFormSemiLagrangian`
#####
##### The horizontal step is taken once per time step, on the final Runge-Kutta stage, right after the
##### tracer tendencies are computed. At that point the transport velocities and the face areas are
##### the ones that produced `w` and `∂t_σ` for the final stage, so the volume swept horizontally
##### plus the vertical transport reproduce the change in cell thickness from σ⁰ to σⁿ⁺¹ exactly.
#####
##### Writing σ* for the thickness after the horizontal step, the tracer is advanced as
#####
#####     σ* c* = σ⁰ c⁰ - [δx(F q̂) + δy(G q̂)] / Vₛ          (horizontal FFSL step)
#####     σⁿ⁺¹ cⁿ⁺¹ = σ* c* + Δt Gᶜ                         (final Runge-Kutta stage)
#####
##### where Gᶜ holds the vertical advection and all other tendency terms.
#####

struct FluxFormSemiLagrangianWorkspace{names, X, Y, S, C}
    sˣ :: X
    sʸ :: Y
    σ⁰ :: S
    q̂ˣ :: X
    q̂ʸ :: Y
    qˣ :: C
    qʸ :: C
end

function FluxFormSemiLagrangianWorkspace{names}(grid) where names
    sˣ = XFaceField(grid)
    sʸ = YFaceField(grid)
    X, Y = typeof(sˣ), typeof(sʸ)
    σ⁰ = initial_stretching_field(grid)
    qˣ = CenterField(grid)
    return FluxFormSemiLagrangianWorkspace{names, X, Y, typeof(σ⁰), typeof(qˣ)}(sˣ, sʸ, σ⁰, XFaceField(grid), YFaceField(grid), qˣ, CenterField(grid))
end

ffsl_tracer_names(::FluxFormSemiLagrangianWorkspace{names}) where names = names

fields_tuple(w::FluxFormSemiLagrangianWorkspace) = (; w.sˣ, w.sʸ, w.σ⁰, w.q̂ˣ, w.q̂ʸ, w.qˣ, w.qʸ)

initial_stretching_field(grid) = nothing
initial_stretching_field(grid::MutableGridOfSomeKind) = Field{Center, Center, Nothing}(grid)

const FFSLScheme = FluxFormSemiLagrangian

function validate_flux_form_semi_lagrangian_scheme(name, scheme, reference, grid, closure)
    if name ∈ closure_required_tracers(closure)
        throw(ArgumentError("FluxFormSemiLagrangian cannot advect the closure tracer $name. " *
                            "Specify the advection of each tracer with a NamedTuple, for example " *
                            "`tracer_advection = (; c = FluxFormSemiLagrangian(), $name = WENO())`."))
    end

    if maximum_courant_number(scheme) != maximum_courant_number(reference) || scheme.limiter != reference.limiter
        throw(ArgumentError("All tracers advected with FluxFormSemiLagrangian must share the same " *
                            "maximum_courant_number and limiter."))
    end

    Hx, Hy, _ = halo_size(grid)
    Hʳ = maximum_courant_number(scheme) + 3
    TX, TY, _ = topology(grid)
    too_small_x = (TX !== Flat) && (Hx < Hʳ)
    too_small_y = (TY !== Flat) && (Hy < Hʳ)

    if too_small_x || too_small_y
        throw(ArgumentError("FluxFormSemiLagrangian with maximum_courant_number = $(maximum_courant_number(scheme)) " *
                            "requires a horizontal halo of at least $Hʳ, but the grid halo is $(halo_size(grid)). " *
                            "Build the grid with, for example, `halo = ($Hʳ, $Hʳ, $(halo_size(grid, 3)))`."))
    end

    return nothing
end

"""
$(TYPEDSIGNATURES)

Attach the workspace used by the horizontal step to every `FluxFormSemiLagrangian` scheme in `advection`.
The workspace (swept Courant numbers, initial grid stretching and scratch fields) is shared by all tracers.
"""
function materialize_flux_form_semi_lagrangian(advection, grid, timestepper, closure)
    tracer_advection = Base.structdiff(advection, NamedTuple{(:momentum,)})
    names = Tuple(name for name in keys(tracer_advection) if tracer_advection[name] isa FFSLScheme)
    isempty(names) && return advection

    if !(timestepper isa SplitRungeKuttaTimeStepper)
        throw(ArgumentError("FluxFormSemiLagrangian tracer advection requires a SplitRungeKuttaTimeStepper, " *
                            "for example `timestepper = :SplitRungeKutta3`."))
    end

    if last(timestepper.β) != 1
        throw(ArgumentError("FluxFormSemiLagrangian tracer advection requires a SplitRungeKuttaTimeStepper " *
                            "whose last stage spans the full time step (β[end] == 1)."))
    end

    reference = tracer_advection[first(names)]

    for name in names
        validate_flux_form_semi_lagrangian_scheme(name, tracer_advection[name], reference, grid, closure)
    end

    workspace = FluxFormSemiLagrangianWorkspace{names}(grid)
    schemes = map(name -> with_workspace(tracer_advection[name], workspace), names)

    return merge(advection, NamedTuple{names}(schemes))
end

@inline flux_form_semi_lagrangian_workspace(advection::NamedTuple) = flux_form_semi_lagrangian_workspace(values(advection))
@inline flux_form_semi_lagrangian_workspace(::Tuple{}) = nothing
@inline flux_form_semi_lagrangian_workspace(schemes::Tuple) = flux_form_semi_lagrangian_workspace(first(schemes), Base.tail(schemes))
@inline flux_form_semi_lagrangian_workspace(scheme, others) = flux_form_semi_lagrangian_workspace(others)
@inline flux_form_semi_lagrangian_workspace(scheme::FluxFormSemiLagrangian{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:FluxFormSemiLagrangianWorkspace}, others) = scheme.workspace

"""
$(TYPEDSIGNATURES)

Advect the tracers that use `FluxFormSemiLagrangian` horizontally. Called after the tracer tendencies are computed:
on the first stage the grid stretching at the beginning of the step is stored, and on the final stage the
horizontal step is taken with the stage's transport velocities.

With a [`TracerTimeStepSplitting`](@ref), only the tracers stepped every time step are advected here; the slow
tracers are advected by `slow_flux_form_semi_lagrangian_advection!` during the long step.
"""
flux_form_semi_lagrangian_advection!(model) =
    flux_form_semi_lagrangian_advection!(model, model.timestepper, flux_form_semi_lagrangian_workspace(model.advection))

flux_form_semi_lagrangian_advection!(model, timestepper, ::Nothing) = nothing
flux_form_semi_lagrangian_advection!(model, ::SplitRungeKuttaTimeStepper, ::Nothing) = nothing

function flux_form_semi_lagrangian_advection!(model, timestepper::SplitRungeKuttaTimeStepper, workspace)
    names = fast_flux_form_semi_lagrangian_names(ffsl_tracer_names(workspace), model.tracer_time_step_splitting)
    isempty(names) && return nothing

    stage = model.clock.stage

    if stage == 1
        store_initial_stretching!(workspace.σ⁰, model.grid)
    end

    if stage == timestepper.Nstages
        u, v, _ = model.transport_velocities
        σc = NamedTuple{names}(model.timestepper.Ψ⁻)
        horizontal_flux_form_semi_lagrangian_step!(model, workspace, Val(names), σc, (; u, v),
                                                   model.clock.last_stage_Δt, model.clock)
    end

    return nothing
end

"""
$(TYPEDSIGNATURES)

Advect the slow tracers of `splitting` that use `FluxFormSemiLagrangian` horizontally during `stage` of the long step.
On the first stage the grid stretching at the beginning of the long step is stored; on the final stage (whose
sub-step `Δτ` spans the whole long step) the horizontal step is taken with the long-step transport velocities,
the same velocities that set the long-step vertical velocity through continuity.
"""
function slow_flux_form_semi_lagrangian_advection!(splitting, model, stage, Nstages, Δτ)
    workspace = flux_form_semi_lagrangian_workspace(model.advection)
    slow_flux_form_semi_lagrangian_advection!(splitting, model, workspace, stage, Nstages, Δτ)
    return nothing
end

slow_flux_form_semi_lagrangian_advection!(splitting, model, ::Nothing, stage, Nstages, Δτ) = nothing

function slow_flux_form_semi_lagrangian_advection!(splitting, model, workspace, stage, Nstages, Δτ)
    names = slow_flux_form_semi_lagrangian_names(ffsl_tracer_names(workspace), splitting)
    isempty(names) && return nothing

    if stage == 1
        store_initial_stretching!(workspace.σ⁰, model.grid)
    end

    if stage == Nstages
        ū, v̄, _ = splitting.velocities
        σc = NamedTuple{names}(splitting.previous_tracers)
        horizontal_flux_form_semi_lagrangian_step!(model, workspace, Val(names), σc, (u = ū, v = v̄), Δτ, splitting.clock)
    end

    return nothing
end

fast_flux_form_semi_lagrangian_names(names, splitting) = filter(name -> !is_slow_tracer(splitting, Val(name)), names)
slow_flux_form_semi_lagrangian_names(names, splitting) = filter(name -> is_slow_tracer(splitting, Val(name)), names)

store_initial_stretching!(::Nothing, grid) = nothing
store_initial_stretching!(σ⁰, grid) = parent(σ⁰) .= parent(grid.z.σᶜᶜⁿ)

# Store c⁰ = σ⁰c⁰ / σ⁰ in the tracer field (its halos are filled next) and remove from σ⁰c⁰ the
# term `c ∇ₕ⋅(Aₕ uₕ)` that the final-stage tendency holds, see `div_Uc(..., ::FluxFormSemiLagrangian, ...)`.
@kernel function _prepare_flux_form_semi_lagrangian_step!(σc, c, grid, u, v, Δt, σ⁰)
    i, j, k = @index(Global, NTuple)
    σ = initial_stretching(i, j, σ⁰)
    Vₛ = static_volume(i, j, k, grid)
    δF = Δt * flux_div_xyᶜᶜᶜ(i, j, k, grid, u, v)
    σc⁰ = @inbounds σc[i, j, k]
    @inbounds σc[i, j, k] = σc⁰ + c[i, j, k] * δF / Vₛ
    @inbounds c[i, j, k] = σc⁰ / σ
end

# The workspace is shared by the tracers stepped every time step and the slow tracers of a `TracerTimeStepSplitting`:
# its fields only carry information from the first to the final stage of a single (short or long) step, and the
# long step is taken after the short step that completes it.
function horizontal_flux_form_semi_lagrangian_step!(model, workspace, ::Val{names}, σc, U, Δt, clock) where names
    grid = model.grid
    arch = architecture(grid)

    tracers = NamedTuple{names}(model.tracers)
    scheme = model.advection[first(names)]
    Cmax = Val(maximum_courant_number(scheme))

    Δt = convert(eltype(grid), Δt)

    map(values(σc), values(tracers)) do σcₙ, cₙ
        launch!(arch, grid, :xyz, _prepare_flux_form_semi_lagrangian_step!, σcₙ, cₙ, grid, U.u, U.v, Δt, workspace.σ⁰)
    end

    fill_halo_regions!(tracers, clock, fields(model))

    flux_form_semi_lagrangian_step!(values(σc), values(tracers), fields_tuple(workspace), grid, U, Δt, scheme.limiter, Cmax)

    return nothing
end
