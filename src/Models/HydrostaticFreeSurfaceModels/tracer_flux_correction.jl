using Oceananigans.Advection: WENO, UpwindBiased, stencil_node, _advective_tracer_flux_x, _advective_tracer_flux_y, _advective_tracer_flux_z
using Oceananigans.Fields: XFaceField, YFaceField
using Oceananigans.Grids: Flat, inactive_cell, inactive_node, topology, column_depthᶜᶜᵃ, static_column_depthᶜᶜᵃ
using Oceananigans.Operators: Vᶜᶜᶜ, σⁿ
using Oceananigans.TimeSteppers: time_discretization
using Oceananigans.Utils: ExplicitTimeDiscretization, AdaptiveVerticallyImplicitDiscretization, KernelParameters

#####
##### Flux-corrected transport (Zalesak, 1979) of tracers through the faces next to walls, in a split Runge-Kutta substep
#####
##### The correction acts on the tendencies of the stage state before the grid advances, with the free surface of the substep.
#####

# with an adaptive vertically implicit discretization, the correction limits the explicit part of the vertical flux
const FluxCorrectedWENO = WENO{<:Any, <:Any, <:Any, <:Union{ExplicitTimeDiscretization, AdaptiveVerticallyImplicitDiscretization}}

# The antidiffusive fluxes and limiters of split Runge-Kutta steps of tracers advected with `FluxCorrectedWENO`, in single precision,
# shared by all tracers; a keyword argument for `TimeStepper`
flux_correction_fields(timestepper, advection, tracers, grid) = NamedTuple()

function flux_correction_fields(::SplitRungeKutta, advection, tracers, grid)
    any(name -> advection[name] isa FluxCorrectedWENO, keys(tracers)) || return NamedTuple()
    TX, TY, TZ = topology(grid)
    Aˣ = TX === Flat ? ZeroField(Float32) : XFaceField(grid, Float32)
    Aʸ = TY === Flat ? ZeroField(Float32) : YFaceField(grid, Float32)
    Aᶻ = TZ === Flat ? ZeroField(Float32) : ZFaceField(grid, Float32)
    flux_correction = (; Aˣ, Aʸ, Aᶻ, R⁺ = CenterField(grid, Float32), R⁻ = CenterField(grid, Float32))
    return (; flux_correction)
end

# A = Fᴴ - Fᴸ of the stage state `c` through the faces next to walls, Fᴴ the flux of `advection` and Fᴸ the donor flux, and the
# limiters R± that keep the substep from Ψ⁻ with Gⁿ, to the free surface η, within the extremes of the stage state within two
# indices and of its own donor update
function compute_antidiffusive_limiters!(flux_correction, Gⁿ, c, Ψ⁻, η, grid, Δt, advection::FluxCorrectedWENO, U)
    (; Aˣ, Aʸ, Aᶻ, R⁺, R⁻) = flux_correction
    arch  = architecture(grid)
    donor = UpwindBiased{1, eltype(grid)}(nothing, nothing, time_discretization(advection))
    Nx, Ny, Nz = size(grid)
    TX, TY, TZ = topology(grid)

    x_faces = KernelParameters((Nx + (TX !== Flat), Ny, Nz), (0, 0, 0))
    y_faces = KernelParameters((Nx, Ny + (TY !== Flat), Nz), (0, 0, 0))
    z_faces = KernelParameters((Nx, Ny, Nz + (TZ !== Flat)), (0, 0, 0))

    launch_antidiffusive_flux!(Aˣ, arch, grid, x_faces, Val(1), _advective_tracer_flux_x, advection, donor, U.u, c)
    launch_antidiffusive_flux!(Aʸ, arch, grid, y_faces, Val(2), _advective_tracer_flux_y, advection, donor, U.v, c)
    launch_antidiffusive_flux!(Aᶻ, arch, grid, z_faces, Val(3), _advective_tracer_flux_z, advection, donor, U.w, c)

    launch!(arch, grid, :xyz, _compute_antidiffusive_limiters!, R⁺, R⁻, grid, Δt, c, Gⁿ, Ψ⁻, η, Aˣ, Aʸ, Aᶻ)
    fill_halo_regions!((R⁺, R⁻))

    return nothing
end

# the rejected part of A moves into Gⁿ, which keeps the tracer content
limit_antidiffusive_tendency!(flux_correction, Gⁿ, c, Ψ⁻, η, grid, Δt, advection, U) = nothing

function limit_antidiffusive_tendency!(flux_correction::NamedTuple, Gⁿ, c, Ψ⁻, η, grid, Δt, advection::FluxCorrectedWENO, U)
    compute_antidiffusive_limiters!(flux_correction, Gⁿ, c, Ψ⁻, η, grid, Δt, advection, U)
    (; Aˣ, Aʸ, Aᶻ, R⁺, R⁻) = flux_correction
    launch!(architecture(grid), grid, :xyz, _reject_antidiffusive_tendency!, Gⁿ, grid, R⁺, R⁻, Aˣ, Aʸ, Aᶻ)
    return nothing
end

# the tracer `name` that the substep advances, with the velocities of its explicit fluxes
function limit_antidiffusive_tendency!(model, η, Δt, ::Val{name}) where name
    (hasclosure(model.closure, FlavorOfCATKE) && name == :e) && return nothing
    U = tracer_advecting_velocities(model.transport_velocities, model.biogeochemistry, model.closure,
                                    model.closure_fields, model.forcing[name], Val(name))
    limit_antidiffusive_tendency!(model.timestepper.flux_correction, model.timestepper.Gⁿ[name], model.tracers[name],
                                  model.timestepper.Ψ⁻[name], η, model.grid, Δt, model.advection[name], U)
    return nothing
end

function limit_antidiffusive_tendencies!(model, Δt)
    η = isnothing(model.free_surface) ? nothing : model.free_surface.displacement
    foreach_name(model.tracers) do _, val_tracer_name
        limit_antidiffusive_tendency!(model, η, Δt, val_tracer_name)
    end
    return nothing
end

launch_antidiffusive_flux!(::ZeroField, args...) = nothing

launch_antidiffusive_flux!(A, arch, grid, parameters, dir, flux::F, advection, donor, U, c) where F =
    launch!(arch, grid, parameters, _compute_antidiffusive_flux!, A, grid, dir, flux, advection, donor, U, c)

# whether the 2N cells about the face (i, j, k) along `dir` reach a cell without data
@inline function wall_face(i, j, k, grid, dir, ::WENO{N}) where N
    no_data(s) = inactive_node(stencil_node(i, j, k, dir, s)..., grid, Center(), Center(), Center())
    return mapreduce(no_data, |, ntuple(n -> n - N - 1, Val(2N)))
end

@kernel function _compute_antidiffusive_flux!(A, grid, dir, flux, advection, donor, U, c)
    i, j, k = @index(Global, NTuple)
    Fᴴ = flux(i, j, k, grid, advection, U, c)
    Fᴸ = flux(i, j, k, grid, donor,     U, c)
    @inbounds A[i, j, k] = ifelse(wall_face(i, j, k, grid, dir, advection), Fᴴ - Fᴸ, zero(Fᴴ))
end

# Δt / V, zero in cells without volume
@inline function substep_reciprocal_volume(i, j, k, grid, Δt)
    V = Vᶜᶜᶜ(i, j, k, grid)
    return ifelse(inactive_cell(i, j, k, grid), zero(V), Δt / ifelse(V > 0, V, one(V)))
end

# the net, incoming and outgoing antidiffusive fluxes of the cell (i, j, k)
@inline function antidiffusive_budget(i, j, k, Aˣ, Aʸ, Aᶻ)
    low  = @inbounds (Aˣ[i, j, k],     Aʸ[i, j, k],     Aᶻ[i, j, k])
    high = @inbounds (Aˣ[i + 1, j, k], Aʸ[i, j + 1, k], Aᶻ[i, j, k + 1])
    a₀   = zero(first(low))

    net      = foldl(+, map((l, h) -> h - l, low, high); init = a₀)
    incoming = foldl(+, map((l, h) -> max(l, a₀) + max(-h, a₀), low, high); init = a₀)
    outgoing = foldl(+, map((l, h) -> max(-l, a₀) + max(h, a₀), low, high); init = a₀)

    return net, incoming, outgoing
end

# The donor update cᴸ is the substep's own update with the donor flux through the faces next to walls, on the grid that the
# free surface η of the substep stretches by σ⁺, as the grid update does
@kernel function _compute_antidiffusive_limiters!(R⁺, R⁻, grid, Δt, c, Gⁿ, Ψ⁻, η, Aˣ, Aʸ, Aᶻ)
    i, j, k = @index(Global, NTuple)
    h  = static_column_depthᶜᶜᵃ(i, j, grid)
    σ⁺ = ifelse(h == 0, one(grid), column_depthᶜᶜᵃ(i, j, grid.Nz + 1, grid, η) / h)
    σ⁻¹ = 1 / σ⁺
    κ  = substep_reciprocal_volume(i, j, k, grid, Δt) * σⁿ(i, j, k, grid, Center(), Center(), Center()) * σ⁻¹
    net, incoming, outgoing = antidiffusive_budget(i, j, k, Aˣ, Aʸ, Aᶻ)
    cᴴ = @inbounds (Ψ⁻[i, j, k] + Δt * Gⁿ[i, j, k]) * σ⁻¹
    cᴸ = cᴴ + κ * net
    P⁺ = κ * incoming
    P⁻ = κ * outgoing

    TX, TY, TZ = topology(grid)
    sx, sy, sz = TX !== Flat, TY !== Flat, TZ !== Flat

    upper, lower = cᴸ, cᴸ
    for q in -2:2, p in -2:2, o in -2:2
        i′, j′, k′ = i + sx * o, j + sy * p, k + sz * q
        active = !inactive_cell(i′, j′, k′, grid)
        cₙ = @inbounds c[i′, j′, k′]
        upper = ifelse(active, max(upper, cₙ), upper)
        lower = ifelse(active, min(lower, cₙ), lower)
    end

    z = zero(cᴸ)
    @inbounds R⁺[i, j, k] = ifelse(P⁺ > 0, min(one(z), max(upper - cᴸ, z) / ifelse(P⁺ > 0, P⁺, one(z))), one(z))
    @inbounds R⁻[i, j, k] = ifelse(P⁻ > 0, min(one(z), max(cᴸ - lower, z) / ifelse(P⁻ > 0, P⁻, one(z))), one(z))
end

# the part of the antidiffusive flux A through the face between the cells `low` and `high` that the limiters reject
@inline function rejected_antidiffusion(A, R⁺, R⁻, low, high)
    C = @inbounds ifelse(A ≥ 0, min(R⁺[high...], R⁻[low...]), min(R⁺[low...], R⁻[high...]))
    return (1 - C) * A
end

# The tendency per reference volume Vᵣ = V / σ takes back the rejected fluxes, so that the substep advances with Fᴸ + C A.
@kernel function _reject_antidiffusive_tendency!(Gⁿ, grid, R⁺, R⁻, Aˣ, Aʸ, Aᶻ)
    i, j, k = @index(Global, NTuple)
    V = Vᶜᶜᶜ(i, j, k, grid)
    σ = σⁿ(i, j, k, grid, Center(), Center(), Center())
    Vᵣ⁻¹ = ifelse(inactive_cell(i, j, k, grid), zero(V), σ / ifelse(V > 0, V, one(V)))

    # neighbours along flat directions collapse onto the cell, where the antidiffusive fluxes vanish
    TX, TY, TZ = topology(grid)
    sx, sy, sz = TX !== Flat, TY !== Flat, TZ !== Flat

    @inbounds begin
        δr = zero(eltype(Gⁿ))
        δr += rejected_antidiffusion(Aˣ[i + 1, j, k], R⁺, R⁻, (i, j, k), (i + sx, j, k))
        δr -= rejected_antidiffusion(Aˣ[i, j, k],     R⁺, R⁻, (i - sx, j, k), (i, j, k))
        δr += rejected_antidiffusion(Aʸ[i, j + 1, k], R⁺, R⁻, (i, j, k), (i, j + sy, k))
        δr -= rejected_antidiffusion(Aʸ[i, j, k],     R⁺, R⁻, (i, j - sy, k), (i, j, k))
        δr += rejected_antidiffusion(Aᶻ[i, j, k + 1], R⁺, R⁻, (i, j, k), (i, j, k + sz))
        δr -= rejected_antidiffusion(Aᶻ[i, j, k],     R⁺, R⁻, (i, j, k - sz), (i, j, k))

        Gⁿ[i, j, k] += Vᵣ⁻¹ * δr
    end
end
