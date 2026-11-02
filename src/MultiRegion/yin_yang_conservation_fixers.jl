using Oceananigans: fields, TracerTendencyCallsite
using Oceananigans.Advection: Centered, _advective_tracer_flux_x, _advective_tracer_flux_y
using Oceananigans.Fields: OneField
using Oceananigans.Grids: immersed_inactive_node, static_column_depthᶜᶜᵃ
using Oceananigans.Models.HydrostaticFreeSurfaceModels: ZStarCoordinate, tracer_advecting_velocities
using Oceananigans.Operators: Azᶜᶜᶜ, Δrᶜᶜᶜ, Ax_qᶠᶜᶜ, Ay_qᶜᶠᶜ
using Oceananigans.Simulations: Simulations, Callback, add_callback!
using Oceananigans.TurbulenceClosures: _diffusive_flux_x, _diffusive_flux_y, closure_required_tracers

"""
    YinYangConservationFixer(model::HydrostaticFreeSurfaceModel)

The callback that removes, in every substep of `model` on a `YinYangGrid`, the volume and the tracer content
that the mismatch of the two panels' fluxes across their seam adds. It acts at the `TracerTendencyCallsite`,
where [`add_conservation_fixer!`](@ref) adds it.

Under a `ZStarCoordinate` the fixer conserves the volume and the tracer contents to rounding error, apart from
their true sources, and keeps uniform tracers uniform. Under a static vertical coordinate it conserves the
volume, while the tracer contents change as on a single grid. Closure tracers are not corrected. With
`QuasiAdamsBashforth2`, a step whose Δt differs from the previous one corrects the free surface with the previous Δt.
"""
struct YinYangConservationFixer{N, W, F, B, P, S}
    tracers :: N # name => index in `model.tracers` of every corrected tracer
    weights :: W # owned_weights(grid)
    faces :: F   # (; x, y), each (; i, j, Δw): the faces where the owned weight jumps by Δw, shared by both panels
    budget :: B  # (row, quantity, density, panel), for the volume followed by the tracers; see `compute_budget!`
    panels :: P  # the panels of `budget`
    sums :: S    # (1, quantity, density, 1), the sums of `budget` over its rows and panels
end

Base.summary(fixer::YinYangConservationFixer) =
    string("YinYangConservationFixer of the volume and the tracers ", keys(fixer.tracers))

Base.show(io::IO, fixer::YinYangConservationFixer) = print(io, summary(fixer))

# The fixer keeps no state between substeps
Oceananigans.prognostic_state(::YinYangConservationFixer) = nothing

# Every substep runs the callback at the `TracerTendencyCallsite`, whatever its schedule. The fixed name makes
# a second call replace the first fixer, so that the budget is removed once.
Simulations.add_conservation_fixer!(simulation, ::YinYangGridOfSomeKind) =
    add_callback!(simulation, Callback(YinYangConservationFixer(simulation.model); callsite=TracerTendencyCallsite());
                  name=:conservation_fixer)

function YinYangConservationFixer(model::HydrostaticFreeSurfaceModel)
    grid = model.grid
    closure_tracers = closure_required_tracers(model.closure) # evolved by the closure itself, so left alone
    tracers = NamedTuple(name => n for (n, name) in enumerate(keys(model.tracers)) if name ∉ closure_tracers)

    arch = architecture(grid)
    FT = eltype(grid)
    weights = owned_weights(grid)
    faces = seam_faces(arch, FT, on_architecture(CPU(), interior(getregion(weights, 1), :, :, 1)))

    Nx, Ny, _ = size(grid)
    Nrows = max(Nx * Ny, length(faces.x.i) + length(faces.y.i))
    budget = zeros(arch, FT, Nrows, 3, 1 + length(tracers), 2)
    panels = MultiRegionObject(Tuple(view(budget, :, :, :, r) for r in 1:2))
    sums = zeros(arch, FT, 1, 3, 1 + length(tracers), 1)

    return YinYangConservationFixer(tracers, weights, faces, budget, panels, sums)
end

#####
##### The algorithm
#####
##### Each panel steps all of its cells with its own fluxes, so the panels disagree on the flux through the curve
##### that divides the sphere between them, and the owned content of a quantity changes by the seam budget
##### E = Σ F Δw of `owned_weights` besides its true sources. Every substep of length Δt, right after the tracer
##### tendencies, the fixer removes Δt E: the free surface of every column of reference depth h drops by Δt ν h,
##### with ν = E / Σ w V₀ for the volume and the reference cell volumes V₀, so that a `ZStarCoordinate` shrinks
##### every cell by the same factor and the tracers leave with the removed volume (which subtracts ν c from the
##### tendency of every tracer c); the rest of each tracer's budget, E - ν Σ w V₀ c, is removed in proportion to |c|.
#####

# The advective and diffusive fluxes of c through x- and y-faces, as in the tracer tendencies
@inline tracer_flux_x(i, j, k, grid, advection, U, c, closure, K, id, clock, model_fields, buoyancy) =
    _advective_tracer_flux_x(i, j, k, grid, advection, U.u, c) +
    Ax_qᶠᶜᶜ(i, j, k, grid, _diffusive_flux_x, closure, K, id, c, clock, model_fields, buoyancy)

@inline tracer_flux_y(i, j, k, grid, advection, U, c, closure, K, id, clock, model_fields, buoyancy) =
    _advective_tracer_flux_y(i, j, k, grid, advection, U.v, c) +
    Ay_qᶜᶠᶜ(i, j, k, grid, _diffusive_flux_y, closure, K, id, c, clock, model_fields, buoyancy)

@inline function active_reference_volume(i, j, k, grid)
    active = !immersed_inactive_node(i, j, k, grid, Center(), Center(), Center())
    return ifelse(active, Azᶜᶜᶜ(i, j, k, grid) * Δrᶜᶜᶜ(i, j, k, grid), zero(grid))
end

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

# Quantity 1 of density d holds F Δw summed over the column of each seam face, the x-faces first
@kernel function _seam_fluxes!(budget, d, n₀, grid, faces, flux, flux_arguments)
    n = @index(Global, Linear)
    @inbounds i, j, Δw = faces.i[n], faces.j[n], faces.Δw[n]
    F = zero(grid)
    for k in 1:size(grid, 3)
        F += flux(i, j, k, grid, flux_arguments...)
    end
    @inbounds budget[n₀ + n, 1, d] = F * Δw
end

# Quantities 2 and 3 hold Σₖ w V₀ c and Σₖ w V₀ |c| over each column (row i + (j - 1) Nx), with c = 1 for the volume
@kernel function _column_contents!(budget, d, grid, weights, c)
    i, j = @index(Global, NTuple)
    @inbounds w = weights[i, j, 1]
    C = zero(grid)
    A = zero(grid)
    for k in 1:size(grid, 3)
        V₀ = active_reference_volume(i, j, k, grid)
        @inbounds cᵏ = c[i, j, k]
        C += V₀ * cᵏ
        A += V₀ * abs(cᵏ)
    end
    row = i + (j - 1) * size(grid, 1)
    @inbounds budget[row, 2, d] = w * C
    @inbounds budget[row, 3, d] = w * A
end

function launch_budget!(budget, d, grid, faces, weights, c, flux_arguments)
    arch = architecture(grid)
    Nx = length(faces.x.i)
    launch!(arch, grid, (Nx,), _seam_fluxes!, budget, d, 0, grid, faces.x, tracer_flux_x, flux_arguments)
    launch!(arch, grid, (length(faces.y.i),), _seam_fluxes!, budget, d, Nx, grid, faces.y, tracer_flux_y, flux_arguments)
    launch!(arch, grid, :xy, _column_contents!, budget, d, grid, weights, c)
    return nothing
end

# The fluxes of the tendencies just computed, from the same state, in one panel. Second-order centered
# fluxes of c = 1 are the volume fluxes, with no diffusion.
function compute_budget!(budget, faces, weights, model, tracers)
    grid = model.grid
    U = model.transport_velocities
    model_fields = fields(model)
    volume_flux_arguments = (Centered(eltype(grid); order=2), U, OneField(), nothing, nothing, Val(0),
                             model.clock, model_fields, model.buoyancy)
    launch_budget!(budget, 1, grid, faces, weights, OneField(), volume_flux_arguments)

    for (d, (name, index)) in enumerate(pairs(tracers))
        c = model.tracers[name]
        velocities = tracer_advecting_velocities(U, model.biogeochemistry, model.closure, model.closure_fields,
                                                 model.forcing[name], Val(name))
        flux_arguments = (model.advection[name], velocities, c, model.closure, model.closure_fields, Val(index),
                          model.clock, model_fields, model.buoyancy)
        launch_budget!(budget, d + 1, grid, faces, weights, c, flux_arguments)
    end

    return nothing
end

#####
##### The correction
#####

@inline budget_sum(sums, quantity, d) = @inbounds sums[1, quantity, d, 1]

# The rate ν = E / Σ w V₀ at which the volume is removed
@inline volume_removal_rate(sums) = budget_sum(sums, 1, 1) / budget_sum(sums, 2, 1)

# The rate r that removes the rest of the seam budget of density d in proportion to |c|
@inline function residual_removal_rate(sums, d, ν)
    E, C, A = budget_sum(sums, 1, d), budget_sum(sums, 2, d), budget_sum(sums, 3, d)
    return ifelse(A > 0, (E - ν * C) / ifelse(A > 0, A, one(A)), zero(A))
end

# The fraction of the removed volume that leaves the cells, as opposed to the free surface alone
stretching(::ZStarCoordinate) = 1
stretching(vertical_coordinate) = 0

# The tendencies evolve σc, so that the correction changes the content Σ w V₀ σc by Δt Σ w V₀ δG
@kernel function _correct_tracer_tendency!(G, c, grid, sums, d, s)
    i, j, k = @index(Global, NTuple)
    ν = volume_removal_rate(sums)
    r = residual_removal_rate(sums, d, ν)
    active = !immersed_inactive_node(i, j, k, grid, Center(), Center(), Center())
    @inbounds cᵢ = c[i, j, k]
    @inbounds G[i, j, k] -= ifelse(active, s * ν * cᵢ + r * abs(cᵢ), zero(cᵢ))
end

function correct_tracer_tendencies!(model, sums, tracers)
    grid = model.grid
    s = stretching(model.vertical_coordinate)
    for (d, name) in enumerate(keys(tracers))
        G = model.timestepper.Gⁿ[name]
        c = model.tracers[name]
        launch!(architecture(grid), grid, :xyz, _correct_tracer_tendency!, G, c, grid, sums, d + 1, s)
    end
    return nothing
end

@kernel function _correct_free_surface!(η, grid, sums, factor)
    i, j = @index(Global, NTuple)
    ν = volume_removal_rate(sums)
    @inbounds η[i, j, grid.Nz+1] -= factor * ν * static_column_depthᶜᶜᵃ(i, j, grid)
end

# The displacement drops by Δt ν h. The explicit free surface of `QuasiAdamsBashforth2TimeStepper` takes the
# current correction with the weight 3/2 + χ now and with -(1/2 + χ) at the next step, through its tendency,
# as the tracers do.
free_surface_corrections(free_surface, timestepper, Δt) = ((free_surface.displacement, Δt),)

function free_surface_corrections(free_surface::ExplicitFreeSurface, timestepper::QuasiAdamsBashforth2TimeStepper, Δt)
    χ = timestepper.χ
    return ((free_surface.displacement, (3 * one(χ) / 2 + χ) * Δt), (timestepper.Gⁿ.η, 1))
end

correct_free_surface!(::Nothing, model, sums, Δt) = nothing

function correct_free_surface!(free_surface, model, sums, Δt)
    grid = model.grid
    for (η, factor) in free_surface_corrections(free_surface, model.timestepper, Δt)
        factor = convert(eltype(grid), factor)
        @apply_regionally launch!(architecture(grid), grid, :xy, _correct_free_surface!, η, grid, sums, factor)
    end
    fill_halo_regions!(free_surface.displacement, model.clock, fields(model))
    return nothing
end

# The time step of the substep in progress. The split Runge-Kutta steppers record Δt / β of the stage on the clock
# before the stage; `QuasiAdamsBashforth2` records its step after it, so `last_Δt` is the step before
substep_time_step(::SplitRungeKuttaTimeStepper, clock) = clock.last_stage_Δt
substep_time_step(::QuasiAdamsBashforth2TimeStepper, clock) = clock.last_Δt

# Executed at the `TracerTendencyCallsite` of every substep
function (fixer::YinYangConservationFixer)(model)
    @apply_regionally compute_budget!(fixer.panels, fixer.faces, fixer.weights, model, fixer.tracers)
    sum!(fixer.sums, fixer.budget)
    @apply_regionally correct_tracer_tendencies!(model, fixer.sums, fixer.tracers)
    correct_free_surface!(model.free_surface, model, fixer.sums, substep_time_step(model.timestepper, model.clock))
    return nothing
end
