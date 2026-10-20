using Oceananigans: fields, TracerTendencyCallsite
using Oceananigans.Advection: _advective_tracer_flux_x, _advective_tracer_flux_y
using Oceananigans.Fields: OneField, tracernames
using Oceananigans.Grids: immersed_inactive_node, static_column_depthᶜᶜᵃ
using Oceananigans.Models.HydrostaticFreeSurfaceModels: ZStarCoordinate, tracer_advecting_velocities
using Oceananigans.Operators: Azᶜᶜᶜ, Δrᶜᶜᶜ, Ax_qᶠᶜᶜ, Ay_qᶜᶠᶜ
using Oceananigans.Simulations: Simulations, Callback, add_callback!
using Oceananigans.TimeSteppers: time_discretization
using Oceananigans.TurbulenceClosures: _diffusive_flux_x, _diffusive_flux_y, closure_required_tracers

"""
    YinYangConservationFixer()

Remove, in every substep of a `HydrostaticFreeSurfaceModel` on a `YinYangGrid`, the volume and the tracer
content that the mismatch of the two panels' fluxes across their seam adds. Add it with
[`add_conservation_fixer!`](@ref).

Under a `ZStarCoordinate` the fixer conserves the volume and the tracer contents to rounding error, apart from
their true sources, and keeps uniform tracers uniform. Under a static vertical coordinate it conserves the
volume, while the tracer contents change as on a single grid. Closure tracers are not corrected. With
`QuasiAdamsBashforth2`, a step whose Δt differs from the previous one corrects the free surface with the
previous Δt: this happens after a change of Δt, which includes the steps that `Simulation` shortens to align
with `TimeInterval` schedules or `stop_time`, and the steps right after them.

```jldoctest
julia> using Oceananigans

julia> YinYangConservationFixer()
YinYangConservationFixer
```
"""
struct YinYangConservationFixer end

Base.summary(::YinYangConservationFixer) = "YinYangConservationFixer"
Base.show(io::IO, fixer::YinYangConservationFixer) = print(io, summary(fixer))

# The skeleton is materialized by `add_conservation_fixer!`, and cannot act as a callback itself
(::YinYangConservationFixer)(args...) =
    throw(ArgumentError("YinYangConservationFixer() is not a callback; add it with add_conservation_fixer!(simulation)."))

struct HydrostaticYinYangConservationFixer{N, F, W, B, P, S}
    tracers :: N # (Val(name), Val(index)) of every corrected tracer
    faces :: F   # (; x, y), each (; i, j, Δw): the faces where the owned weight jumps by Δw, shared by both panels
    weights :: W # owned_weights(grid)
    budget :: B  # (row, quantity, density, panel), for the volume followed by the tracers; see `compute_budget!`
    panels :: P  # the panels of `budget`
    sums :: S    # (1, quantity, density, 1), the sums of `budget` over its rows and panels
end

tracer_names(fixer::HydrostaticYinYangConservationFixer) = map(name_and_index -> tracer_name(name_and_index...), fixer.tracers)
tracer_name(::Val{name}, index) where name = name

Base.summary(fixer::HydrostaticYinYangConservationFixer) = string("YinYangConservationFixer of the volume and the tracers ", tracer_names(fixer))
Base.show(io::IO, fixer::HydrostaticYinYangConservationFixer) = print(io, summary(fixer))

# The fixer keeps no state between substeps
Oceananigans.prognostic_state(::HydrostaticYinYangConservationFixer) = nothing

# The correction needs the tracer tendencies and the free surface of each substep, which the
# `TracerTendencyCallsite` of `HydrostaticFreeSurfaceModel` provides. Every substep runs the callback whatever
# its schedule. A fixed name makes a second call replace the first fixer, so that the budget is removed once.
function Simulations.add_conservation_fixer!(simulation, ::YinYangGridOfSomeKind)
    fixer = materialize_conservation_fixer(YinYangConservationFixer(), simulation.model)
    callback = Callback(fixer; callsite=TracerTendencyCallsite())
    return add_callback!(simulation, callback; name=:conservation_fixer)
end

materialize_conservation_fixer(::YinYangConservationFixer, model) =
    throw(ArgumentError("YinYangConservationFixer requires a HydrostaticFreeSurfaceModel."))

materialize_conservation_fixer(fixer::YinYangConservationFixer, model::HydrostaticFreeSurfaceModel) =
    materialize_conservation_fixer(fixer, model.grid, tracernames(model.tracers), model.closure)

materialize_conservation_fixer(::YinYangConservationFixer, grid, tracer_names, closure) =
    throw(ArgumentError("YinYangConservationFixer requires a YinYangGrid."))

function materialize_conservation_fixer(::YinYangConservationFixer, grid::YinYangGridOfSomeKind, tracer_names, closure)
    # The tracers that a closure evolves itself are left alone
    closure_tracers = closure_required_tracers(closure)
    tracers = Tuple((Val(name), Val(n)) for (n, name) in enumerate(tracer_names) if name ∉ closure_tracers)

    arch = architecture(grid)
    FT = eltype(grid)
    weights = owned_weights(grid)
    faces = seam_faces(arch, FT, on_architecture(CPU(), interior(getregion(weights, 1), :, :, 1)))

    Nx, Ny, _ = size(getregion(grid, 1))
    Nrows = max(Nx * Ny, length(faces.x.i) + length(faces.y.i))
    Nd = 1 + length(tracers)
    budget = zeros(arch, FT, Nrows, 3, Nd, 2)
    panels = MultiRegionObject(Tuple(view(budget, :, :, :, r) for r in 1:2))
    sums = zeros(arch, FT, 1, 3, Nd, 1)

    return HydrostaticYinYangConservationFixer(tracers, faces, weights, budget, panels, sums)
end

#####
##### The algorithm
#####
##### Each panel steps all of its cells with its own fluxes and interpolates its halos from the other panel, so
##### the panels disagree on the flux through the curve that divides the sphere between them. The owned content
##### of a quantity therefore changes by the seam budget E = Σ F Δw of `owned_weights`, besides its true sources.
##### Every substep of length Δt, right after the tracer tendencies, the fixer removes Δt E:
#####
##### * the free surface of every column of reference depth h drops by Δt ν h, with ν = E / Σ w V₀ for the volume
#####   and the reference cell volumes V₀, so that a `ZStarCoordinate` shrinks every cell by the same factor;
##### * under a `ZStarCoordinate`, the tracers leave with the removed volume, which subtracts ν c from the
#####   tendency of every tracer c and keeps uniform tracers uniform;
##### * the rest of each tracer's seam budget, E - ν Σ w V₀ c, is removed in proportion to |c|.
#####

# The x-faces (i, j) and y-faces (i, j) across which the owned weight `w` of the interior cells jumps,
# with the jump Δw = w[i, j] - w[i-1, j] or w[i, j] - w[i, j-1], taking w = 0 outside the interior
function seam_faces(arch, FT, w)
    Nx, Ny = size(w)
    weight(i, j) = 1 ≤ i ≤ Nx && 1 ≤ j ≤ Ny ? w[i, j] : zero(eltype(w))
    x = [(i, j, weight(i, j) - weight(i-1, j)) for j in 1:Ny for i in 1:Nx+1 if weight(i, j) != weight(i-1, j)]
    y = [(i, j, weight(i, j) - weight(i, j-1)) for j in 1:Ny+1 for i in 1:Nx if weight(i, j) != weight(i, j-1)]
    table(faces) = on_architecture(arch, (; i = [f[1] for f in faces], j = [f[2] for f in faces], Δw = FT[f[3] for f in faces]))
    return (; x = table(x), y = table(y))
end

#####
##### The budget of each panel
#####
##### Quantity 1 holds F Δw summed over the column of each seam face, for the volume flux and the tracer fluxes
##### of the tendencies. Quantities 2 and 3 hold Σₖ w V₀ c and Σₖ w V₀ |c| over each column (i + (j - 1) Nx),
##### with c = 1 for the volume. The rows past those a quantity uses stay zero.
#####

@inline seam_flux_x(i, j, k, grid, U) = Ax_qᶠᶜᶜ(i, j, k, grid, U.u)
@inline seam_flux_y(i, j, k, grid, U) = Ay_qᶜᶠᶜ(i, j, k, grid, U.v)

@inline seam_flux_x(i, j, k, grid, advection, U, c, closure, K, id, clock, model_fields, buoyancy) =
    _advective_tracer_flux_x(i, j, k, grid, advection, U.u, c) +
    diffusive_seam_flux_x(i, j, k, grid, closure, K, id, c, clock, model_fields, buoyancy)

@inline seam_flux_y(i, j, k, grid, advection, U, c, closure, K, id, clock, model_fields, buoyancy) =
    _advective_tracer_flux_y(i, j, k, grid, advection, U.v, c) +
    diffusive_seam_flux_y(i, j, k, grid, closure, K, id, c, clock, model_fields, buoyancy)

@inline diffusive_seam_flux_x(i, j, k, grid, ::Nothing, K, args...) = zero(grid)
@inline diffusive_seam_flux_y(i, j, k, grid, ::Nothing, K, args...) = zero(grid)

@inline diffusive_seam_flux_x(i, j, k, grid, closure, K, args...) =
    Ax_qᶠᶜᶜ(i, j, k, grid, _diffusive_flux_x, time_discretization(closure), closure, K, args...)

@inline diffusive_seam_flux_y(i, j, k, grid, closure, K, args...) =
    Ay_qᶜᶠᶜ(i, j, k, grid, _diffusive_flux_y, time_discretization(closure), closure, K, args...)

@inline diffusive_seam_flux_x(i, j, k, grid, ::Tuple{}, ::Tuple{}, args...) = zero(grid)
@inline diffusive_seam_flux_y(i, j, k, grid, ::Tuple{}, ::Tuple{}, args...) = zero(grid)

@inline diffusive_seam_flux_x(i, j, k, grid, closures::Tuple, Ks::Tuple, args...) =
    diffusive_seam_flux_x(i, j, k, grid, first(closures), first(Ks), args...) +
    diffusive_seam_flux_x(i, j, k, grid, Base.tail(closures), Base.tail(Ks), args...)

@inline diffusive_seam_flux_y(i, j, k, grid, closures::Tuple, Ks::Tuple, args...) =
    diffusive_seam_flux_y(i, j, k, grid, first(closures), first(Ks), args...) +
    diffusive_seam_flux_y(i, j, k, grid, Base.tail(closures), Base.tail(Ks), args...)

@inline store_seam_fluxes!(budget, row, d, i, j, Δw, grid, flux, ::Tuple{}) = nothing

@inline function store_seam_fluxes!(budget, row, d, i, j, Δw, grid, flux, densities)
    F = zero(grid)
    for k in 1:size(grid, 3)
        F += flux(i, j, k, grid, first(densities)...)
    end
    @inbounds budget[row, 1, d] = F * Δw
    return store_seam_fluxes!(budget, row, d + 1, i, j, Δw, grid, flux, Base.tail(densities))
end

@kernel function _seam_fluxes!(budget, n₀, grid, faces, flux, densities)
    n = @index(Global, Linear)
    @inbounds i, j, Δw = faces.i[n], faces.j[n], faces.Δw[n]
    store_seam_fluxes!(budget, n₀ + n, 1, i, j, Δw, grid, flux, densities)
end

@inline function wet_reference_volume(i, j, k, grid)
    active = !immersed_inactive_node(i, j, k, grid, Center(), Center(), Center())
    return ifelse(active, Azᶜᶜᶜ(i, j, k, grid) * Δrᶜᶜᶜ(i, j, k, grid), zero(grid))
end

@inline store_column_contents!(budget, row, d, i, j, grid, w, ::Tuple{}) = nothing

@inline function store_column_contents!(budget, row, d, i, j, grid, w, contents)
    c = first(contents)
    C = zero(grid)
    A = zero(grid)
    for k in 1:size(grid, 3)
        V₀ = wet_reference_volume(i, j, k, grid)
        @inbounds cᵏ = c[i, j, k]
        C += V₀ * cᵏ
        A += V₀ * abs(cᵏ)
    end
    @inbounds budget[row, 2, d] = w * C
    @inbounds budget[row, 3, d] = w * A
    return store_column_contents!(budget, row, d + 1, i, j, grid, w, Base.tail(contents))
end

@kernel function _column_contents!(budget, grid, weights, contents)
    i, j = @index(Global, NTuple)
    @inbounds w = weights[i, j, 1]
    store_column_contents!(budget, i + (j - 1) * size(grid, 1), 1, i, j, grid, w, contents)
end

@inline tracer_field(fields, ::Val{name}, index) where name = fields[name]

@inline function seam_flux_arguments(model, ::Val{name}, index) where name
    velocities = tracer_advecting_velocities(model.transport_velocities, model.biogeochemistry, model.closure,
                                             model.closure_fields, model.forcing[name], Val(name))
    return (model.advection[name], velocities, model.tracers[name], model.closure, model.closure_fields,
            index, model.clock, fields(model), model.buoyancy)
end

# The fluxes of the tendencies just computed, from the same state, in one panel
function compute_budget!(budget, faces, weights, model, tracers)
    grid = model.grid
    arch = architecture(grid)

    densities = ((model.transport_velocities,), map(t -> seam_flux_arguments(model, t...), tracers)...)
    Nx = length(faces.x.i)
    Ny = length(faces.y.i)
    launch!(arch, grid, (Nx,), _seam_fluxes!, budget, 0, grid, faces.x, seam_flux_x, densities)
    launch!(arch, grid, (Ny,), _seam_fluxes!, budget, Nx, grid, faces.y, seam_flux_y, densities)

    contents = (OneField(), map(t -> tracer_field(model.tracers, t...), tracers)...)
    launch!(arch, grid, :xy, _column_contents!, budget, grid, weights, contents)

    return nothing
end

#####
##### The correction
#####

# The rate ν = E / Σ w V₀ at which the volume is removed
@inline volume_removal_rate(sums) = @inbounds sums[1, 1, 1, 1] / sums[1, 2, 1, 1]

# The rate r that removes the rest of the seam budget of density d in proportion to |c|
@inline function residual_removal_rate(sums, d, ν)
    @inbounds E, C, A = sums[1, 1, d, 1], sums[1, 2, d, 1], sums[1, 3, d, 1]
    return ifelse(A > 0, (E - ν * C) / ifelse(A > 0, A, one(A)), zero(A))
end

# The fraction of the removed volume that leaves the cells, as opposed to the free surface alone
stretching(::ZStarCoordinate) = 1
stretching(vertical_coordinate) = 0

@inline correct_tendencies!(::Tuple{}, ::Tuple{}, i, j, k, d, wet, sums, νσ, ν) = nothing

@inline function correct_tendencies!(Gs, cs, i, j, k, d, wet, sums, νσ, ν)
    r = residual_removal_rate(sums, d, ν)
    @inbounds c = first(cs)[i, j, k]
    @inbounds first(Gs)[i, j, k] -= ifelse(wet, νσ * c + r * abs(c), zero(c))
    return correct_tendencies!(Base.tail(Gs), Base.tail(cs), i, j, k, d + 1, wet, sums, νσ, ν)
end

# The tendencies evolve σc, so that the correction changes the content Σ w V₀ σc by Δt Σ w V₀ δG
@kernel function _correct_tracer_tendencies!(Gs, cs, grid, sums, s)
    i, j, k = @index(Global, NTuple)
    ν = volume_removal_rate(sums)
    wet = !immersed_inactive_node(i, j, k, grid, Center(), Center(), Center())
    correct_tendencies!(Gs, cs, i, j, k, 2, wet, sums, s * ν, ν)
end

correct_tracer_tendencies!(model, sums, ::Tuple{}) = nothing

function correct_tracer_tendencies!(model, sums, tracers)
    grid = model.grid
    Gs = map(t -> tracer_field(model.timestepper.Gⁿ, t...), tracers)
    cs = map(t -> tracer_field(model.tracers, t...), tracers)
    launch!(architecture(grid), grid, :xyz, _correct_tracer_tendencies!, Gs, cs, grid, sums, stretching(model.vertical_coordinate))
    return nothing
end

@kernel function _correct_free_surface!(η, grid, sums, Δt)
    i, j = @index(Global, NTuple)
    ν = volume_removal_rate(sums)
    @inbounds η[i, j, grid.Nz+1] -= Δt * ν * static_column_depthᶜᶜᵃ(i, j, grid)
end

# The explicit free surface of `QuasiAdamsBashforth2TimeStepper` takes the current correction with the weight
# 3/2 + χ now and with -(1/2 + χ) at the next step, through its tendency, as the tracers do
@kernel function _correct_explicit_free_surface!(η, Gη, grid, sums, αΔt)
    i, j = @index(Global, NTuple)
    ν = volume_removal_rate(sums)
    νh = ν * static_column_depthᶜᶜᵃ(i, j, grid)
    @inbounds η[i, j, grid.Nz+1] -= αΔt * νh
    @inbounds Gη[i, j, grid.Nz+1] -= νh
end

correct_displacement!(free_surface, timestepper, grid, sums, Δt) =
    launch!(architecture(grid), grid, :xy, _correct_free_surface!, free_surface.displacement, grid, sums, convert(eltype(grid), Δt))

function correct_displacement!(free_surface::ExplicitFreeSurface, timestepper::QuasiAdamsBashforth2TimeStepper, grid, sums, Δt)
    χ = timestepper.χ
    αΔt = convert(eltype(grid), (3 * one(χ) / 2 + χ) * Δt)
    launch!(architecture(grid), grid, :xy, _correct_explicit_free_surface!, free_surface.displacement, timestepper.Gⁿ.η, grid, sums, αΔt)
    return nothing
end

correct_free_surface!(::Nothing, model, sums, Δt) = nothing

function correct_free_surface!(free_surface, model, sums, Δt)
    @apply_regionally correct_displacement!(free_surface, model.timestepper, model.grid, sums, Δt)
    fill_halo_regions!(free_surface.displacement, model.clock, fields(model))
    return nothing
end

# The time step of the substep in progress. The split Runge-Kutta steppers record Δt / β of the stage on the clock
# before the stage; `QuasiAdamsBashforth2` records its step after it, so `last_Δt` is the step before
substep_time_step(::SplitRungeKuttaTimeStepper, clock) = clock.last_stage_Δt
substep_time_step(::QuasiAdamsBashforth2TimeStepper, clock) = clock.last_Δt

# Executed at the `TracerTendencyCallsite` of every substep
(fixer::HydrostaticYinYangConservationFixer)(model) = remove_seam_imbalance!(fixer, model, substep_time_step(model.timestepper, model.clock))

function remove_seam_imbalance!(fixer::HydrostaticYinYangConservationFixer, model, Δt)
    @apply_regionally compute_budget!(fixer.panels, fixer.faces, fixer.weights, model, fixer.tracers)
    sum!(fixer.sums, fixer.budget)
    @apply_regionally correct_tracer_tendencies!(model, fixer.sums, fixer.tracers)
    correct_free_surface!(model.free_surface, model, fixer.sums, Δt)
    return nothing
end
