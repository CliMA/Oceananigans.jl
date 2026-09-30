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

# Note that for AB2, `transport_velocities` holds the value of the prognostic velocities at `tⁿ`
@inline baseline_velocities(model, ::SplitRungeKuttaTimeStepper) = model.timestepper.Ψ⁻
@inline baseline_velocities(model, ::QuasiAdamsBashforth2TimeStepper) = model.transport_velocities

function compute_free_surface_tendency!(grid, model, free_surface::SplitExplicitFreeSurface, Δt)
    GUⁿ = model.timestepper.Gⁿ.U
    GVⁿ = model.timestepper.Gⁿ.V

    u,  v,  _ = model.velocities
    baseline  = baseline_velocities(model, model.timestepper)
    u⁻, v⁻    = baseline.u, baseline.v

    @apply_regionally launch!(architecture(grid), grid, :xy, _compute_realized_barotropic_tendency!, GUⁿ, GVⁿ, grid, u, v, u⁻, v⁻, Δt)

    fill_halo_regions!((GUⁿ, GVⁿ); async=true)

    return nothing
end
