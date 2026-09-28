# Kinetic energy and enstrophy budgets of the hydrostatic momentum equations, evaluated after every time step.
#
#   K = ½ Σ Δx Δy Δz (u² + v²),     Z = ½ Σ Az Δz ζ²,
#   dK/dt = Σ_terms Σ Δx Δy Δz (u G_u + v G_v),     dZ/dt = Σ_terms Σ Az Δz ζ curl(G),
#
# with the terms Coriolis, advection, baroclinic pressure, barotropic pressure, boundary fluxes (wind and bottom
# drag), implicit vertical viscosity, and the remainder of the interior tendency (immersed drag, forcing, explicit
# closure). Each term's rate is integrated in time with the trapezoidal rule; the difference between the change
# of K (or Z) and the sum of the integrated terms is the time-stepping residual.

using Oceananigans.Advection: U_dot_∇u, U_dot_∇v, U_dot_∇u_hydrostatic_metric, U_dot_∇v_hydrostatic_metric
using Oceananigans.Coriolis: x_f_cross_U, y_f_cross_U
using Oceananigans.Operators: ∂xᶠᶜᶜ, ∂yᶜᶠᶜ, ζ₃ᶠᶠᶜ, Azᶠᶠᶜ, Δzᶠᶠᶜ, Δxᶠᶜᶜ, Δyᶠᶜᶜ, Δzᶠᶜᶜ, Δxᶜᶠᶜ, Δyᶜᶠᶜ, Δzᶜᶠᶜ
using Oceananigans.TurbulenceClosures: ∂ⱼ_τ₁ⱼ, ∂ⱼ_τ₂ⱼ
using Oceananigans.BoundaryConditions: compute_x_bcs!, compute_y_bcs!, compute_z_bcs!, fill_halo_regions!
using Oceananigans.Grids: peripheral_node, inactive_node, topology, RightCenterFolded
using Oceananigans.Models.HydrostaticFreeSurfaceModels: ExplicitFreeSurface
using Oceananigans.Architectures: architecture

@inline coriolis_u(i, j, k, grid, coriolis, U) = - x_f_cross_U(i, j, k, grid, coriolis, U)
@inline coriolis_v(i, j, k, grid, coriolis, U) = - y_f_cross_U(i, j, k, grid, coriolis, U)

@inline advection_u(i, j, k, grid, advection, U) = - U_dot_∇u(i, j, k, grid, advection, U) - U_dot_∇u_hydrostatic_metric(i, j, k, grid, advection, U, U)
@inline advection_v(i, j, k, grid, advection, U) = - U_dot_∇v(i, j, k, grid, advection, U) - U_dot_∇v_hydrostatic_metric(i, j, k, grid, advection, U, U)

@inline baroclinic_pressure_u(i, j, k, grid, p) = - ∂xᶠᶜᶜ(i, j, k, grid, p)
@inline baroclinic_pressure_v(i, j, k, grid, p) = - ∂yᶜᶠᶜ(i, j, k, grid, p)

@inline barotropic_pressure_u(i, j, k, grid, g, η) = @inbounds - g * (η[i, j, grid.Nz+1] - η[i-1, j, grid.Nz+1]) / Δxᶠᶜᶜ(i, j, k, grid)
@inline barotropic_pressure_v(i, j, k, grid, g, η) = @inbounds - g * (η[i, j, grid.Nz+1] - η[i, j-1, grid.Nz+1]) / Δyᶜᶠᶜ(i, j, k, grid)

@inline viscosity_u(i, j, k, grid, closure, closure_fields, clock, fields, buoyancy) = - ∂ⱼ_τ₁ⱼ(i, j, k, grid, closure, closure_fields, clock, fields, buoyancy)
@inline viscosity_v(i, j, k, grid, closure, closure_fields, clock, fields, buoyancy) = - ∂ⱼ_τ₂ⱼ(i, j, k, grid, closure, closure_fields, clock, fields, buoyancy)

@inline energy_weightᶠᶜᶜ(i, j, k, grid) = Δxᶠᶜᶜ(i, j, k, grid) * Δyᶠᶜᶜ(i, j, k, grid) * Δzᶠᶜᶜ(i, j, k, grid)
@inline energy_weightᶜᶠᶜ(i, j, k, grid) = Δxᶜᶠᶜ(i, j, k, grid) * Δyᶜᶠᶜ(i, j, k, grid) * Δzᶜᶠᶜ(i, j, k, grid)
@inline enstrophy_weightᶠᶠᶜ(i, j, k, grid) = Azᶠᶠᶜ(i, j, k, grid) * Δzᶠᶠᶜ(i, j, k, grid)

@inline active_faceᶠᶜᶜ(i, j, k, grid) = !(peripheral_node(i, j, k, grid, Face(), Center(), Center()) | inactive_node(i, j, k, grid, Face(), Center(), Center()))
@inline active_faceᶜᶠᶜ(i, j, k, grid) = !(peripheral_node(i, j, k, grid, Center(), Face(), Center()) | inactive_node(i, j, k, grid, Center(), Face(), Center()))

face_fields(grid, u_function, v_function, args...) =
    (Field(KernelFunctionOperation{Face, Center, Center}(u_function, grid, args...)),
     Field(KernelFunctionOperation{Center, Face, Center}(v_function, grid, args...)))

"""
    MomentumBudget(model; implicit_closure=nothing, surface_flux=nothing)

`implicit_closure` is the explicit counterpart of a vertically implicit closure, used to evaluate the viscous
term that the implicit solver applies. `surface_flux = (u_function, v_function, args)` evaluates the surface
momentum flux (the wind) so that it can be separated from the other boundary fluxes.
"""
function MomentumBudget(model; implicit_closure=nothing, surface_flux=nothing)
    grid = model.grid
    U = model.velocities
    free_surface = model.free_surface
    g = free_surface.gravitational_acceleration
    η = free_surface.displacement.data

    terms = (coriolis = face_fields(grid, coriolis_u, coriolis_v, model.coriolis, U),
             advection = face_fields(grid, advection_u, advection_v, model.advection.momentum, U),
             baroclinic_pressure = face_fields(grid, baroclinic_pressure_u, baroclinic_pressure_v, model.pressure.pHY′),
             barotropic_pressure = face_fields(grid, barotropic_pressure_u, barotropic_pressure_v, g, η))

    if !isnothing(implicit_closure)
        terms = merge(terms, (; implicit_viscosity = face_fields(grid, viscosity_u, viscosity_v, implicit_closure,
                                                                model.closure_fields, model.clock, fields(model), model.buoyancy)))
    end

    if !isnothing(surface_flux)
        terms = merge(terms, (; wind = face_fields(grid, surface_flux...)))
    end

    boundary_fluxes = (XFaceField(grid), YFaceField(grid))
    remainder = (XFaceField(grid), YFaceField(grid))

    ζ = Field(KernelFunctionOperation{Face, Face, Center}(ζ₃ᶠᶠᶜ, grid, U.u, U.v))
    curl_buffer = XFaceField(grid), YFaceField(grid)
    curl = Field(KernelFunctionOperation{Face, Face, Center}(ζ₃ᶠᶠᶜ, grid, curl_buffer...))

    weights = (u = Field(KernelFunctionOperation{Face, Center, Center}(energy_weightᶠᶜᶜ, grid)),
               v = Field(KernelFunctionOperation{Center, Face, Center}(energy_weightᶜᶠᶜ, grid)),
               ζ = Field(KernelFunctionOperation{Face, Face, Center}(enstrophy_weightᶠᶠᶜ, grid)))

    active = (u = Array(interior(Field(KernelFunctionOperation{Face, Center, Center}(active_faceᶠᶜᶜ, grid)))) .== 1,
              v = Array(interior(Field(KernelFunctionOperation{Center, Face, Center}(active_faceᶜᶠᶜ, grid)))) .== 1)

    # The U-pivot fold duplicates the last row of u, which is counted once by halving its weight
    fold_u = ones(size(active.u))
    topology(grid, 2) <: RightCenterFolded && (fold_u[:, end, :] .= 1 / 2)

    names = (keys(terms)..., :boundary_fluxes, :remainder)
    history = (time = Float64[], kinetic_energy = Float64[], enstrophy = Float64[],
               energy_rates = Dict(name => Float64[] for name in names),
               enstrophy_rates = Dict(name => Float64[] for name in names))

    return (; model, terms, boundary_fluxes, remainder, ζ, curl_buffer, curl, weights, active, fold_u, history,
              explicit_free_surface = free_surface isa ExplicitFreeSurface)
end

# Degenerate faces at the tripolar pivots have zero metrics and non-finite tendencies
masked(field, active) = map((q, a) -> ifelse(a & isfinite(q), q, zero(q)), Array(interior(field)), active)

function budget_rates(budget, Tu, Tv)
    u = masked(budget.model.velocities.u, budget.active.u)
    v = masked(budget.model.velocities.v, budget.active.v)
    wu = masked(budget.weights.u, budget.active.u) .* budget.fold_u
    wv = masked(budget.weights.v, budget.active.v)
    energy_rate = sum(wu .* u .* Tu) + sum(wv .* v .* Tv)

    set!(budget.curl_buffer[1], Tu)
    set!(budget.curl_buffer[2], Tv)
    fill_halo_regions!(budget.curl_buffer)
    compute!(budget.curl)
    enstrophy_rate = sum(Array(interior(budget.weights.ζ)) .* Array(interior(budget.ζ)) .* Array(interior(budget.curl)))

    return energy_rate, enstrophy_rate
end

function record_budget!(budget)
    model = budget.model
    arch = architecture(model.grid)
    foreach(compute!, (budget.weights..., budget.ζ))
    for pair in budget.terms, field in pair
        compute!(field)
    end

    for field in budget.boundary_fluxes
        fill!(parent(field), 0)
    end
    args = (model.clock, fields(model), model.closure, model.buoyancy)
    for (G, velocity) in zip(budget.boundary_fluxes, (model.velocities.u, model.velocities.v))
        compute_x_bcs!(G, velocity, arch, args...)
        compute_y_bcs!(G, velocity, arch, args...)
        compute_z_bcs!(G, velocity, arch, args...)
    end

    Gⁿ = model.timestepper.Gⁿ
    included = (:coriolis, :advection, :baroclinic_pressure)
    budget.explicit_free_surface && (included = (included..., :barotropic_pressure))
    for (n, G) in enumerate((Gⁿ.u, Gⁿ.v))
        interior(budget.remainder[n]) .= interior(G)
        for name in included
            interior(budget.remainder[n]) .-= interior(budget.terms[name][n])
        end
    end

    u_mask, v_mask = budget.active.u, budget.active.v
    kinetic_energy = (sum(masked(budget.weights.u, u_mask) .* budget.fold_u .* masked(model.velocities.u, u_mask) .^ 2) +
                      sum(masked(budget.weights.v, v_mask) .* masked(model.velocities.v, v_mask) .^ 2)) / 2
    enstrophy = sum(Array(interior(budget.weights.ζ)) .* Array(interior(budget.ζ)) .^ 2) / 2

    push!(budget.history.time, model.clock.time)
    push!(budget.history.kinetic_energy, kinetic_energy)
    push!(budget.history.enstrophy, enstrophy)

    all_terms = merge(budget.terms, (boundary_fluxes = budget.boundary_fluxes, remainder = budget.remainder))
    for (name, (Gu, Gv)) in pairs(all_terms)
        energy_rate, enstrophy_rate = budget_rates(budget, masked(Gu, u_mask), masked(Gv, v_mask))
        push!(budget.history.energy_rates[name], energy_rate)
        push!(budget.history.enstrophy_rates[name], enstrophy_rate)
    end

    return nothing
end

trapezoid(t, rate) = vcat(0.0, cumsum([(rate[n+1] + rate[n]) / 2 * (t[n+1] - t[n]) for n in 1:length(t)-1]))

# Cumulative integrals of each term and the residual; the wind is part of the boundary fluxes
function integrated_budget(history)
    t = history.time
    energy = Dict(name => trapezoid(t, rate) for (name, rate) in history.energy_rates)
    enstrophy = Dict(name => trapezoid(t, rate) for (name, rate) in history.enstrophy_rates)
    tendency_names = filter(!=(:wind), collect(keys(energy)))
    energy_residual = (history.kinetic_energy .- history.kinetic_energy[1]) .- sum(energy[name] for name in tendency_names)
    enstrophy_residual = (history.enstrophy .- history.enstrophy[1]) .- sum(enstrophy[name] for name in tendency_names)
    return (; t, energy, enstrophy, energy_residual, enstrophy_residual)
end
