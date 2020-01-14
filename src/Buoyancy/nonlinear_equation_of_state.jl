""" Return the geopotential depth at `i, j, k` at cell centers. """
@inline Dᵃᵃᶜ(i, j, k, grid) = @inbounds -grid.zC[k]

""" Return the geopotential depth at `i, j, k` at cell z-interfaces. """
@inline Dᵃᵃᶠ(i, j, k, grid) = @inbounds -grid.zF[k]

# Basic functionality
@inline ρ′(i, j, k, grid, eos, C) = @inbounds ρ′(C.T[i, j, k], C.S[i, j, k], Dᵃᵃᶜ(i, j, k, grid), eos)

@inline thermal_expansionᶜᶜᶜ(i, j, k, grid, eos, C) = @inbounds thermal_expansion(C.T[i, j, k], C.S[i, j, k], Dᵃᵃᶜ(i, j, k, grid), eos)
@inline thermal_expansionᶠᶜᶜ(i, j, k, grid, eos, C) = @inbounds thermal_expansion(ℑxᶠᵃᵃ(i, j, k, grid, C.T), ℑxᶠᵃᵃ(i, j, k, grid, C.S), Dᵃᵃᶜ(i, j, k, grid), eos)
@inline thermal_expansionᶜᶠᶜ(i, j, k, grid, eos, C) = @inbounds thermal_expansion(ℑyᵃᶠᵃ(i, j, k, grid, C.T), ℑyᵃᶠᵃ(i, j, k, grid, C.S), Dᵃᵃᶜ(i, j, k, grid), eos)
@inline thermal_expansionᶜᶜᶠ(i, j, k, grid, eos, C) = @inbounds thermal_expansion(ℑzᵃᵃᶠ(i, j, k, grid, C.T), ℑzᵃᵃᶠ(i, j, k, grid, C.S), Dᵃᵃᶠ(i, j, k, grid), eos)

@inline haline_contractionᶜᶜᶜ(i, j, k, grid, eos, C) = @inbounds haline_contraction(C.T[i, j, k], C.S[i, j, k], Dᵃᵃᶜ(i, j, k, grid), eos)
@inline haline_contractionᶠᶜᶜ(i, j, k, grid, eos, C) = @inbounds haline_contraction(ℑxᶠᵃᵃ(i, j, k, grid, C.T), ℑxᶠᵃᵃ(i, j, k, grid, C.S), Dᵃᵃᶜ(i, j, k, grid), eos)
@inline haline_contractionᶜᶠᶜ(i, j, k, grid, eos, C) = @inbounds haline_contraction(ℑyᵃᶠᵃ(i, j, k, grid, C.T), ℑyᵃᶠᵃ(i, j, k, grid, C.S), Dᵃᵃᶜ(i, j, k, grid), eos)
@inline haline_contractionᶜᶜᶠ(i, j, k, grid, eos, C) = @inbounds haline_contraction(ℑzᵃᵃᶠ(i, j, k, grid, C.T), ℑzᵃᵃᶠ(i, j, k, grid, C.S), Dᵃᵃᶠ(i, j, k, grid), eos)

@inline buoyancy_perturbation(i, j, k, grid, b::AbstractBuoyancy{<:AbstractNonlinearEquationOfState}, C) =
    - b.gravitational_acceleration * ρ′(i, j, k, grid, b.equation_of_state, C) / b.equation_of_state.ρ₀
