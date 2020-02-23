using JULES.Operators: ∂xᶠᵃᵃ, ∂yᵃᶠᵃ, ∂zᵃᵃᶠ

####
#### Pressure gradient ∇p terms for entropy S = ρs
####

@inline ∂p∂x(i, j, k, grid, tvar, gravity, momenta, total_density, gases, tracers) = ∂xᶠᵃᵃ(i, j, k, grid, diagnose_p, tvar, gases, gravity, total_density, momenta, tracers)
@inline ∂p∂y(i, j, k, grid, tvar, gravity, momenta, total_density, gases, tracers) = ∂yᵃᶠᵃ(i, j, k, grid, diagnose_p, tvar, gases, gravity, total_density, momenta, tracers)
@inline ∂p∂z(i, j, k, grid, tvar, gravity, momenta, total_density, gases, tracers) = ∂zᵃᵃᶠ(i, j, k, grid, diagnose_p, tvar, gases, gravity, total_density, momenta, tracers)
