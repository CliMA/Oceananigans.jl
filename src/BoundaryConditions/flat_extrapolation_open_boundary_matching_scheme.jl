using Oceananigans.Operators: Δxᶜᶜᶜ, Δyᶜᶜᶜ, Δzᶜᶜᶜ

"""
    FlatExtrapolation

Zero gradient perpendicular velocity boundary condition.

We find the boundary value by Taylor expanding the gradient at the boundary point (`xᵢ`)
to second order:
```math
f′(xᵢ) ≈ f′(xᵢ₋₁) + f′′(xᵢ₋₁)(xᵢ₋₁ - xᵢ) + O(Δx²) = f′(xᵢ₋₁) + f′′(xᵢ₋₁)Δx + O(Δx²),
```
where ``Δx=xᵢ₋₁ - xᵢ`` (for simplicity, we will also assume the spacing is constant at
all ``i`` for now).
We can substitute the gradient at some point ``j`` (``f′(xⱼ)``) with the central
difference approximation:
```math
f′(xⱼ) ≈ (f(xⱼ₊₁) - f(xⱼ₋₁)) / 2Δx,
```
and the second derivative at some point ``j`` (``f′′(xⱼ)``) can be approximated as:
```math
f′′(xⱼ) ≈ (f′(xⱼ₊₁) - f′(xⱼ₋₁)) / 2Δx = ((f(xⱼ₊₂) - f(xⱼ)) - (f(xⱼ) - f(xⱼ₋₂))) / (2Δx)².
```
When we then substitute for the boundary adjacent point ``f′′(xᵢ₋₁)`` we know that
``f′(xⱼ₊₁)=f′(xᵢ)=0`` so the Taylor expansion becomes:
```math
f(xᵢ) ≈ f(xᵢ₋₂) - (f(xᵢ₋₁) - f(xᵢ₋₃))/2 + O(Δx²).
```

When the grid spacing is not constant the above can be repeated resulting in the factor
of 1/2 changes to ``Δx₋₁/(Δx₋₂ + Δx₋₃)`` instead, i.e.:
```math
f(xᵢ) ≈ f(xᵢ₋₂) - (f(xᵢ₋₁) - f(xᵢ₋₃))Δxᵢ₋₁/(Δxᵢ₋₂ + Δxᵢ₋₃) + O(Δx²)
```.
"""
struct FlatExtrapolation{FT}
    relaxation_timescale :: FT
end

const FEOBC = BoundaryCondition{<:Open{<:FlatExtrapolation}}

function FlatExtrapolationOpenBoundaryCondition(val = nothing; relaxation_timescale = Inf, kwargs...)
    classification = Open(FlatExtrapolation(relaxation_timescale))

    return BoundaryCondition(classification, val; kwargs...)
end

@inline function relax(l, m, grid, ϕ, bc, clock, model_fields)
    Δt = clock.last_stage_Δt
    τ = bc.classification.matching_scheme.relaxation_timescale

    Δt̄ = min(1, Δt / τ)
    ϕₑₓₜ = getbc(bc, l, m, grid, clock, model_fields)

    Δϕ = (ϕₑₓₜ - ϕ) * Δt̄
    not_relaxing = isnothing(bc.condition) | !isfinite(clock.last_stage_Δt)
    Δϕ =  ifelse(not_relaxing, zero(ϕ), Δϕ)

    return ϕ + Δϕ
end

@inline function _fill_west_halo!(j, k, grid, ϕ, bc::FEOBC, loc, clock, model_fields)
    Δx₁ = Δxᶜᶜᶜ(1, j, k, grid)
    Δx₂ = Δxᶜᶜᶜ(2, j, k, grid)
    Δx₃ = Δxᶜᶜᶜ(3, j, k, grid)

    spacing_factor = Δx₁ / (Δx₂ + Δx₃)

    gradient_free_ϕ = @inbounds ϕ[3, j, k] - (ϕ[2, j, k] - ϕ[4, j, k]) * spacing_factor

    @inbounds ϕ[1, j, k] = relax(j, k, grid, gradient_free_ϕ, bc, clock, model_fields)

    return nothing
end

@inline function _fill_east_halo!(j, k, grid, ϕ, bc::FEOBC, loc, clock, model_fields)
    i = grid.Nx + 1

    Δx₁ = Δxᶜᶜᶜ(i-1, j, k, grid)
    Δx₂ = Δxᶜᶜᶜ(i-2, j, k, grid)
    Δx₃ = Δxᶜᶜᶜ(i-3, j, k, grid)

    spacing_factor = Δx₁ / (Δx₂ + Δx₃)

    gradient_free_ϕ = @inbounds ϕ[i - 2, j, k] - (ϕ[i - 1, j, k] - ϕ[i - 3, j, k]) * spacing_factor

    @inbounds ϕ[i, j, k] = relax(j, k, grid, gradient_free_ϕ, bc, clock, model_fields)

    return nothing
end

@inline function _fill_south_halo!(i, k, grid, ϕ, bc::FEOBC, loc, clock, model_fields)
    Δy₁ = Δyᶜᶜᶜ(i, 1, k, grid)
    Δy₂ = Δyᶜᶜᶜ(i, 2, k, grid)
    Δy₃ = Δyᶜᶜᶜ(i, 3, k, grid)

    spacing_factor = Δy₁ / (Δy₂ + Δy₃)

    gradient_free_ϕ = ϕ[i, 3, k] - (ϕ[i, 2, k] - ϕ[i, 4, k]) * spacing_factor

    @inbounds ϕ[i, 1, k] = relax(i, k, grid, gradient_free_ϕ, bc, clock, model_fields)

    return nothing
end

@inline function _fill_north_halo!(i, k, grid, ϕ, bc::FEOBC, loc, clock, model_fields)
    j = grid.Ny + 1

    Δy₁ = Δyᶜᶜᶜ(i, j-1, k, grid)
    Δy₂ = Δyᶜᶜᶜ(i, j-2, k, grid)
    Δy₃ = Δyᶜᶜᶜ(i, j-3, k, grid)

    spacing_factor = Δy₁ / (Δy₂ + Δy₃)

    gradient_free_ϕ = @inbounds ϕ[i, j - 2, k] - (ϕ[i, j - 1, k] - ϕ[i, j - 3, k]) * spacing_factor

    @inbounds ϕ[i, j, k] = relax(i, k, grid, gradient_free_ϕ, bc, clock, model_fields)

    return nothing
end

@inline function _fill_bottom_halo!(i, j, grid, ϕ, bc::FEOBC, loc, clock, model_fields)
    Δz₁ = Δzᶜᶜᶜ(i, j, 1, grid)
    Δz₂ = Δzᶜᶜᶜ(i, j, 2, grid)
    Δz₃ = Δzᶜᶜᶜ(i, j, 3, grid)

    spacing_factor = Δz₁ / (Δz₂ + Δz₃)

    gradient_free_ϕ = @inbounds ϕ[i, j, 3] - (ϕ[i, j, 2] - ϕ[i, j, 4]) * spacing_factor

    @inbounds ϕ[i, j, 1] = relax(i, j, grid, gradient_free_ϕ, bc, clock, model_fields)

    return nothing
end

@inline function _fill_top_halo!(i, j, grid, ϕ, bc::FEOBC, loc, clock, model_fields)
    k = grid.Nz + 1

    Δz₁ = Δzᶜᶜᶜ(i, j, k-1, grid)
    Δz₂ = Δzᶜᶜᶜ(i, j, k-2, grid)
    Δz₃ = Δzᶜᶜᶜ(i, j, k-3, grid)

    spacing_factor = Δz₁ / (Δz₂ + Δz₃)

    gradient_free_ϕ = @inbounds ϕ[i, j, k - 2] - (ϕ[i, j, k - 1] - ϕ[i, j, k - 3]) * spacing_factor

    @inbounds ϕ[i, j, k] = relax(i, j, grid, gradient_free_ϕ, bc, clock, model_fields)

    return nothing
end
