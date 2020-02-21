using JULES.Operators: ∂xᶠᵃᵃ, ∂yᵃᶠᵃ, ∂zᵃᵃᶠ

####
#### Pressure gradient ∇p terms for entropy S = ρs
####

@inline ∂p∂x(i, j, k, grid, tvar, gravity, momenta, total_density, densities, tracers) = ∂xᶠᵃᵃ(i, j, k, grid, diagnose_p, tvar, gravity, momenta, total_density, densities, tracers)
@inline ∂p∂y(i, j, k, grid, tvar, gravity, momenta, total_density, densities, tracers) = ∂yᵃᶠᵃ(i, j, k, grid, diagnose_p, tvar, gravity, momenta, total_density, densities, tracers)
@inline ∂p∂z(i, j, k, grid, tvar, gravity, momenta, total_density, densities, tracers) = ∂zᵃᵃᶠ(i, j, k, grid, diagnose_p, tvar, gravity, momenta, total_density, densities, tracers)
