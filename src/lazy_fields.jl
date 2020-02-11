import Base: getindex

struct LazyPrimitiveField{X, Y, Z, A, G, C, D} <: AbstractLocatedField{X, Y, Z, A, G}
    architecture :: A
    grid :: G
    conservative_field :: C
    density :: D
end

@inline getindex(f::LazyPrimitiveField{Cell, Cell, Cell}, inds...) =
    @inbounds f.conservative_field[inds...] / f.density[inds...]

@inline getindex(f::LazyPrimitiveField{Face, Cell, Cell}, inds...) =
    @inbounds f.conservative_field[inds..] / ℑxᶠᵃᵃ(inds..., f.grid, f.density)

@inline getindex(f::LazyPrimitiveField{Cell, Face, Cell}, inds...) =
    @inbounds f.conservative_field[inds..] / ℑyᵃᶠᵃ(inds..., f.grid, f.density)

@inline getindex(f::LazyPrimitiveField{Cell, Cell, Face}, inds...) =
    @inbounds f.conservative_field[inds..] / ℑzᵃᵃᶠ(inds..., f.grid, f.density)

struct LazyTotalDensityField{X, Y, Z, A, G, D} <: AbstractLocatedField{X, Y, Z, A, G}
    architecture :: A
    grid :: G
    densities :: D
end

@inline getindex(f::LazyTotalDensityField{Cell, Cell, Cell}, inds...) =
    @inbounds sum(ρ[inds...] for ρ in f.densities)

struct LazyPressureField{X, Y, Z, A, G, M, E, D} <: AbstractLocatedField{X, Y, Z, A, G}
    architecture :: A
    grid :: G
    thermodynamic_variable :: TV
    gravity :: GR
    momenta :: M
    total_density :: R
    gases :: GS
end

@inline getindex(f::LazyPressureField{Cell, Cell, Cell}, inds...) =
    diagnose_p(inds..., f.grid, tvar, gravity, momenta, total_density, densities, tracers)
