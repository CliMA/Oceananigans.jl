# The barotropic tendency is the depth-integrated velocity change the step realized, `(∫u dz - ∫u⁻ dz)/Δt`.
@kernel function _compute_realized_barotropic_tendency!(Gᵁ, Gⱽ, grid, u, v, u⁻, v⁻, Δt)
    i, j = @index(Global, NTuple)

    locU = (Face(), Center(), Center())
    locV = (Center(), Face(), Center())

    δU = zero(grid)
    δV = zero(grid)

    for k in 1:grid.Nz
        @inbounds δU += Δzᶠᶜᶜ(i, j, k, grid) * (u[i, j, k] - u⁻[i, j, k]) * !peripheral_node(i, j, k, grid, locU...)
        @inbounds δV += Δzᶜᶠᶜ(i, j, k, grid) * (v[i, j, k] - v⁻[i, j, k]) * !peripheral_node(i, j, k, grid, locV...)
    end

    @inbounds Gᵁ[i, j, 1] = δU / Δt
    @inbounds Gⱽ[i, j, 1] = δV / Δt
end

# For AB2 and for an SSP stage, `transport_velocities` holds the prognostic velocities at `tⁿ` and at the stage input respectively
@inline baseline_velocities(model, ::SplitRungeKuttaTimeStepper) = model.timestepper.Ψ⁻
@inline baseline_velocities(model, ::QuasiAdamsBashforth2TimeStepper) = model.transport_velocities
@inline baseline_velocities(model, ::SSPRungeKutta3TimeStepper) = model.transport_velocities

weight_slow_forcing!(GUⁿ, GVⁿ, grid, timestepper, stage) = nothing

# Ĝ accumulates βₘ Gᵐ over the stages, and the last stage drives its solve with the full sum.
function weight_slow_forcing!(GUⁿ, GVⁿ, grid, ts::SSPRungeKutta3TimeStepper, stage)
    FT = eltype(grid)
    β = convert(FT, ssp_quadrature_weights(ts.coefficients)[stage])
    keep = convert(FT, stage > 1)
    Sᵁ, Sⱽ = stage == ts.Nstages ? (GUⁿ, GVⁿ) : (ts.Ĝ.U, ts.Ĝ.V)
    launch!(architecture(grid), grid, :xy, _accumulate_slow_forcing!, Sᵁ, Sⱽ, ts.Ĝ.U, ts.Ĝ.V, GUⁿ, GVⁿ, β, keep)
    return nothing
end

@kernel function _accumulate_slow_forcing!(Sᵁ, Sⱽ, Ĝᵁ, Ĝⱽ, Gᵁ, Gⱽ, β, keep)
    i, j = @index(Global, NTuple)
    @inbounds Sᵁ[i, j, 1] = keep * Ĝᵁ[i, j, 1] + β * Gᵁ[i, j, 1]
    @inbounds Sⱽ[i, j, 1] = keep * Ĝⱽ[i, j, 1] + β * Gⱽ[i, j, 1]
end

function HydrostaticFreeSurfaceModels.compute_free_surface_tendency!(grid, model, free_surface::SplitExplicitFreeSurface, Δt)
    GUⁿ = model.timestepper.Gⁿ.U
    GVⁿ = model.timestepper.Gⁿ.V

    u,  v,  _ = model.velocities
    baseline  = baseline_velocities(model, model.timestepper)
    u⁻, v⁻    = baseline.u, baseline.v

    @apply_regionally begin
        launch!(architecture(grid), grid, :xy, _compute_realized_barotropic_tendency!, GUⁿ, GVⁿ, grid, u, v, u⁻, v⁻, Δt)
        weight_slow_forcing!(GUⁿ, GVⁿ, grid, model.timestepper, model.clock.stage)
    end

    fill_halo_regions!((GUⁿ, GVⁿ); async=true)

    return nothing
end
