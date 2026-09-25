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

struct FluxFormSemiLagrangianGeometry{names, X, Y, S}
    sˣ :: X
    sʸ :: Y
    σ⁰ :: S
end

FluxFormSemiLagrangianGeometry{names}(sˣ::X, sʸ::Y, σ⁰::S) where {names, X, Y, S} =
    FluxFormSemiLagrangianGeometry{names, X, Y, S}(sˣ, sʸ, σ⁰)

ffsl_tracer_names(::FluxFormSemiLagrangianGeometry{names}) where names = names

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

Attach the scratch fields used by the horizontal step to every `FluxFormSemiLagrangian` scheme in `advection`.
The swept Courant numbers and the initial grid stretching are shared by all tracers.
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

    geometry = FluxFormSemiLagrangianGeometry{names}(XFaceField(grid), YFaceField(grid), initial_stretching_field(grid))

    schemes = map(names) do name
        workspace = (; qˣ = CenterField(grid), qʸ = CenterField(grid), geometry)
        with_workspace(tracer_advection[name], workspace)
    end

    return merge(advection, NamedTuple{names}(schemes))
end

@inline flux_form_semi_lagrangian_workspace(advection::NamedTuple) = flux_form_semi_lagrangian_workspace(values(advection))
@inline flux_form_semi_lagrangian_workspace(::Tuple{}) = nothing
@inline flux_form_semi_lagrangian_workspace(schemes::Tuple) = flux_form_semi_lagrangian_workspace(first(schemes), Base.tail(schemes))
@inline flux_form_semi_lagrangian_workspace(scheme, others) = flux_form_semi_lagrangian_workspace(others)
@inline flux_form_semi_lagrangian_workspace(scheme::FluxFormSemiLagrangian{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:NamedTuple}, others) = scheme.workspace

"""
$(TYPEDSIGNATURES)

Advect the tracers that use `FluxFormSemiLagrangian` horizontally. Called after the tracer tendencies are computed:
on the first stage the grid stretching at the beginning of the step is stored, and on the final stage the
horizontal step is taken with the stage's transport velocities.
"""
flux_form_semi_lagrangian_advection!(model) =
    flux_form_semi_lagrangian_advection!(model, model.timestepper, flux_form_semi_lagrangian_workspace(model.advection))

flux_form_semi_lagrangian_advection!(model, timestepper, ::Nothing) = nothing

function flux_form_semi_lagrangian_advection!(model, timestepper::SplitRungeKuttaTimeStepper, workspace)
    geometry = workspace.geometry
    stage = model.clock.stage

    if stage == 1
        store_initial_stretching!(geometry.σ⁰, model.grid)
    end

    if stage == timestepper.Nstages
        horizontal_flux_form_semi_lagrangian_step!(model, geometry, model.clock.last_stage_Δt)
    end

    return nothing
end

store_initial_stretching!(::Nothing, grid) = nothing
store_initial_stretching!(σ⁰, grid) = parent(σ⁰) .= parent(grid.z.σᶜᶜⁿ)

# Store c⁰ = σ⁰c⁰ / σ⁰ in the tracer fields (their halos are filled next) and remove from σ⁰c⁰
# the term `c ∇ₕ⋅(Aₕ uₕ)` that the final-stage tendency holds, see `div_Uc(..., ::FluxFormSemiLagrangian, ...)`.
@kernel function _prepare_flux_form_semi_lagrangian_step!(σc, tracers, grid, u, v, Δt, σ⁰)
    i, j, k = @index(Global, NTuple)

    σ = initial_stretching(i, j, σ⁰)
    Vₛ = static_volume(i, j, k, grid)
    δF = Δt * flux_div_xyᶜᶜᶜ(i, j, k, grid, u, v)

    for n in 1:length(tracers)
        c = tracers[n]
        σc⁰ = @inbounds σc[n][i, j, k]
        @inbounds σc[n][i, j, k] = σc⁰ + c[i, j, k] * δF / Vₛ
        @inbounds c[i, j, k] = σc⁰ / σ
    end
end

function horizontal_flux_form_semi_lagrangian_step!(model, geometry, Δt)
    grid = model.grid
    arch = architecture(grid)
    names = ffsl_tracer_names(geometry)

    tracers = NamedTuple{names}(model.tracers)
    schemes = NamedTuple{names}(model.advection)
    σc = values(NamedTuple{names}(model.timestepper.Ψ⁻))
    qˣ = map(scheme -> scheme.workspace.qˣ, values(schemes))
    qʸ = map(scheme -> scheme.workspace.qʸ, values(schemes))

    scheme = first(schemes)
    Cmax = Val(maximum_courant_number(scheme))

    u, v, _ = model.transport_velocities
    Δt = convert(eltype(grid), Δt)

    launch!(arch, grid, :xyz, _prepare_flux_form_semi_lagrangian_step!, σc, values(tracers), grid, u, v, Δt, geometry.σ⁰)
    fill_halo_regions!(tracers, model.clock, fields(model))

    shared_geometry = (; sˣ = geometry.sˣ, sʸ = geometry.sʸ, σ⁰ = geometry.σ⁰)
    flux_form_semi_lagrangian_step!(σc, values(tracers), qˣ, qʸ, shared_geometry, grid, (; u, v), Δt, scheme.limiter, Cmax)

    return nothing
end
