#####
##### Flux divergences of GhostCells WENO on immersed grids: the local extrema of a node and of its neighbors are computed
##### once and shared by the faces around the node, which also receive the location of the advected field
#####

# Bounds preservation has its own tracer flux divergence; the time discretization enters through the flux functions
const GhostCellWENOWithoutBoundsPreservation = WENO{<:Any, <:Any, <:Any, <:Any, Nothing, <:GhostCells}
const VectorInvariantVerticalGhostCells = VectorInvariant{<:Any, <:Any, <:Any, <:Any, <:Any, <:GhostCellWENO}

@inline node_extrema(i, j, k, grid, scheme, d, ℓ, ψ) = local_extrema(i, j, k, grid, face_boundary_distances(scheme, ψ, i, j, k, grid, ℓ..., d), ψ)

# The local extrema of the neighbors behind and ahead of the node along d
@inline neighbor_extrema(i, j, k, grid, ::Type{Flat}, scheme, d, ℓ, ψ) = nothing, nothing

@inline function neighbor_extrema(i, j, k, grid, topology, scheme, d, ℓ, ψ)
    di, dj, dk = unit_vector(d)
    ℰ⁻ = node_extrema(i - di, j - dj, k - dk, grid, scheme, d, ℓ, ψ)
    ℰ⁺ = node_extrema(i + di, j + dj, k + dk, grid, scheme, d, ℓ, ψ)
    return ℰ⁻, ℰ⁺
end

for divergence in (:div_Uc, :div_𝐯u, :div_𝐯v, :div_𝐯w)
    @eval begin
        @inline $divergence(i, j, k, grid::ImmersedBoundaryGrid, ::GhostCellWENOWithoutBoundsPreservation, ::ZeroU, ψ)           = zero(grid)
        @inline $divergence(i, j, k, grid::ImmersedBoundaryGrid, ::GhostCellWENOWithoutBoundsPreservation, U,       ::ZeroField) = zero(grid)
        @inline $divergence(i, j, k, grid::ImmersedBoundaryGrid, ::GhostCellWENOWithoutBoundsPreservation, ::ZeroU, ::ZeroField) = zero(grid)
    end
end

@inline function div_Uc(i, j, k, grid::ImmersedBoundaryGrid, advection::GhostCellWENOWithoutBoundsPreservation, U, c)
    scheme = advection.buffer_scheme
    ℓ = (Center(), Center(), Center())

    ℰ₀       = node_extrema(i, j, k, grid, scheme, Val(1), ℓ, c)
    ℰˣ⁻, ℰˣ⁺ = neighbor_extrema(i, j, k, grid, topology(grid, 1), scheme, Val(1), ℓ, c)
    ℰʸ⁻, ℰʸ⁺ = neighbor_extrema(i, j, k, grid, topology(grid, 2), scheme, Val(2), ℓ, c)
    ℰᶻ⁻, ℰᶻ⁺ = neighbor_extrema(i, j, k, grid, topology(grid, 3), scheme, Val(3), ℓ, c)

    Fˣ⁺ = _advective_tracer_flux_x(i+1, j, k, grid, advection, U.u, FaceExtrema(c, ℓ, ℰ₀,  ℰˣ⁺))
    Fˣ⁻ = _advective_tracer_flux_x(i,   j, k, grid, advection, U.u, FaceExtrema(c, ℓ, ℰˣ⁻, ℰ₀))
    Fʸ⁺ = _advective_tracer_flux_y(i, j+1, k, grid, advection, U.v, FaceExtrema(c, ℓ, ℰ₀,  ℰʸ⁺))
    Fʸ⁻ = _advective_tracer_flux_y(i, j,   k, grid, advection, U.v, FaceExtrema(c, ℓ, ℰʸ⁻, ℰ₀))
    Fᶻ⁺ = _advective_tracer_flux_z(i, j, k+1, grid, advection, U.w, FaceExtrema(c, ℓ, ℰ₀,  ℰᶻ⁺))
    Fᶻ⁻ = _advective_tracer_flux_z(i, j, k,   grid, advection, U.w, FaceExtrema(c, ℓ, ℰᶻ⁻, ℰ₀))

    return V⁻¹ᶜᶜᶜ(i, j, k, grid) * ((Fˣ⁺ - Fˣ⁻) + (Fʸ⁺ - Fʸ⁻) + (Fᶻ⁺ - Fᶻ⁻))
end

@inline function div_𝐯u(i, j, k, grid::ImmersedBoundaryGrid, advection::GhostCellWENOWithoutBoundsPreservation, U, u)
    scheme = advection.buffer_scheme
    ℓ = (Face(), Center(), Center())

    ℰ₀       = node_extrema(i, j, k, grid, scheme, Val(1), ℓ, u)
    ℰˣ⁻, ℰˣ⁺ = neighbor_extrema(i, j, k, grid, topology(grid, 1), scheme, Val(1), ℓ, u)
    ℰʸ⁻, ℰʸ⁺ = neighbor_extrema(i, j, k, grid, topology(grid, 2), scheme, Val(2), ℓ, u)
    ℰᶻ⁻, ℰᶻ⁺ = neighbor_extrema(i, j, k, grid, topology(grid, 3), scheme, Val(3), ℓ, u)

    Fˣ⁺ = _advective_momentum_flux_Uu(i,   j, k, grid, advection, U[1], FaceExtrema(u, ℓ, ℰ₀,  ℰˣ⁺))
    Fˣ⁻ = _advective_momentum_flux_Uu(i-1, j, k, grid, advection, U[1], FaceExtrema(u, ℓ, ℰˣ⁻, ℰ₀))
    Fʸ⁺ = _advective_momentum_flux_Vu(i, j+1, k, grid, advection, U[2], FaceExtrema(u, ℓ, ℰ₀,  ℰʸ⁺))
    Fʸ⁻ = _advective_momentum_flux_Vu(i, j,   k, grid, advection, U[2], FaceExtrema(u, ℓ, ℰʸ⁻, ℰ₀))
    Fᶻ⁺ = _advective_momentum_flux_Wu(i, j, k+1, grid, advection, U[3], FaceExtrema(u, ℓ, ℰ₀,  ℰᶻ⁺))
    Fᶻ⁻ = _advective_momentum_flux_Wu(i, j, k,   grid, advection, U[3], FaceExtrema(u, ℓ, ℰᶻ⁻, ℰ₀))

    return V⁻¹ᶠᶜᶜ(i, j, k, grid) * ((Fˣ⁺ - Fˣ⁻) + (Fʸ⁺ - Fʸ⁻) + (Fᶻ⁺ - Fᶻ⁻))
end

@inline function div_𝐯v(i, j, k, grid::ImmersedBoundaryGrid, advection::GhostCellWENOWithoutBoundsPreservation, U, v)
    scheme = advection.buffer_scheme
    ℓ = (Center(), Face(), Center())

    ℰ₀       = node_extrema(i, j, k, grid, scheme, Val(1), ℓ, v)
    ℰˣ⁻, ℰˣ⁺ = neighbor_extrema(i, j, k, grid, topology(grid, 1), scheme, Val(1), ℓ, v)
    ℰʸ⁻, ℰʸ⁺ = neighbor_extrema(i, j, k, grid, topology(grid, 2), scheme, Val(2), ℓ, v)
    ℰᶻ⁻, ℰᶻ⁺ = neighbor_extrema(i, j, k, grid, topology(grid, 3), scheme, Val(3), ℓ, v)

    Fˣ⁺ = _advective_momentum_flux_Uv(i+1, j, k, grid, advection, U[1], FaceExtrema(v, ℓ, ℰ₀,  ℰˣ⁺))
    Fˣ⁻ = _advective_momentum_flux_Uv(i,   j, k, grid, advection, U[1], FaceExtrema(v, ℓ, ℰˣ⁻, ℰ₀))
    Fʸ⁺ = _advective_momentum_flux_Vv(i, j,   k, grid, advection, U[2], FaceExtrema(v, ℓ, ℰ₀,  ℰʸ⁺))
    Fʸ⁻ = _advective_momentum_flux_Vv(i, j-1, k, grid, advection, U[2], FaceExtrema(v, ℓ, ℰʸ⁻, ℰ₀))
    Fᶻ⁺ = _advective_momentum_flux_Wv(i, j, k+1, grid, advection, U[3], FaceExtrema(v, ℓ, ℰ₀,  ℰᶻ⁺))
    Fᶻ⁻ = _advective_momentum_flux_Wv(i, j, k,   grid, advection, U[3], FaceExtrema(v, ℓ, ℰᶻ⁻, ℰ₀))

    return V⁻¹ᶜᶠᶜ(i, j, k, grid) * ((Fˣ⁺ - Fˣ⁻) + (Fʸ⁺ - Fʸ⁻) + (Fᶻ⁺ - Fᶻ⁻))
end

@inline function div_𝐯w(i, j, k, grid::ImmersedBoundaryGrid, advection::GhostCellWENOWithoutBoundsPreservation, U, w)
    scheme = advection.buffer_scheme
    ℓ = (Center(), Center(), Face())

    ℰ₀       = node_extrema(i, j, k, grid, scheme, Val(1), ℓ, w)
    ℰˣ⁻, ℰˣ⁺ = neighbor_extrema(i, j, k, grid, topology(grid, 1), scheme, Val(1), ℓ, w)
    ℰʸ⁻, ℰʸ⁺ = neighbor_extrema(i, j, k, grid, topology(grid, 2), scheme, Val(2), ℓ, w)
    ℰᶻ⁻, ℰᶻ⁺ = neighbor_extrema(i, j, k, grid, topology(grid, 3), scheme, Val(3), ℓ, w)

    Fˣ⁺ = _advective_momentum_flux_Uw(i+1, j, k, grid, advection, U[1], FaceExtrema(w, ℓ, ℰ₀,  ℰˣ⁺))
    Fˣ⁻ = _advective_momentum_flux_Uw(i,   j, k, grid, advection, U[1], FaceExtrema(w, ℓ, ℰˣ⁻, ℰ₀))
    Fʸ⁺ = _advective_momentum_flux_Vw(i, j+1, k, grid, advection, U[2], FaceExtrema(w, ℓ, ℰ₀,  ℰʸ⁺))
    Fʸ⁻ = _advective_momentum_flux_Vw(i, j,   k, grid, advection, U[2], FaceExtrema(w, ℓ, ℰʸ⁻, ℰ₀))
    Fᶻ⁺ = _advective_momentum_flux_Ww(i, j, k,   grid, advection, U[3], FaceExtrema(w, ℓ, ℰ₀,  ℰᶻ⁺))
    Fᶻ⁻ = _advective_momentum_flux_Ww(i, j, k-1, grid, advection, U[3], FaceExtrema(w, ℓ, ℰᶻ⁻, ℰ₀))

    return V⁻¹ᶜᶜᶠ(i, j, k, grid) * ((Fˣ⁺ - Fˣ⁻) + (Fʸ⁺ - Fʸ⁻) + (Fᶻ⁺ - Fᶻ⁻))
end

@inline function vertical_advection_U(i, j, k, grid::ImmersedBoundaryGrid, scheme::VectorInvariantVerticalGhostCells, U)
    advection = scheme.vertical_advection_scheme
    ℓ = (Face(), Center(), Center())

    ℰ₀       = node_extrema(i, j, k, grid, advection.buffer_scheme, Val(3), ℓ, U.u)
    ℰᶻ⁻, ℰᶻ⁺ = neighbor_extrema(i, j, k, grid, topology(grid, 3), advection.buffer_scheme, Val(3), ℓ, U.u)

    Φᵟ  = upwinded_divergence_flux_Uᶠᶜᶜ(i, j, k, grid, scheme, U.u, U.v)
    Fᶻ⁺ = _advective_momentum_flux_Wu(i, j, k+1, grid, advection, U.w, FaceExtrema(U.u, ℓ, ℰ₀,  ℰᶻ⁺))
    Fᶻ⁻ = _advective_momentum_flux_Wu(i, j, k,   grid, advection, U.w, FaceExtrema(U.u, ℓ, ℰᶻ⁻, ℰ₀))

    return 1/Vᶠᶜᶜ(i, j, k, grid) * (Φᵟ + (Fᶻ⁺ - Fᶻ⁻))
end

@inline function vertical_advection_V(i, j, k, grid::ImmersedBoundaryGrid, scheme::VectorInvariantVerticalGhostCells, U)
    advection = scheme.vertical_advection_scheme
    ℓ = (Center(), Face(), Center())

    ℰ₀       = node_extrema(i, j, k, grid, advection.buffer_scheme, Val(3), ℓ, U.v)
    ℰᶻ⁻, ℰᶻ⁺ = neighbor_extrema(i, j, k, grid, topology(grid, 3), advection.buffer_scheme, Val(3), ℓ, U.v)

    Φᵟ  = upwinded_divergence_flux_Vᶜᶠᶜ(i, j, k, grid, scheme, U.u, U.v)
    Fᶻ⁺ = _advective_momentum_flux_Wv(i, j, k+1, grid, advection, U.w, FaceExtrema(U.v, ℓ, ℰ₀,  ℰᶻ⁺))
    Fᶻ⁻ = _advective_momentum_flux_Wv(i, j, k,   grid, advection, U.w, FaceExtrema(U.v, ℓ, ℰᶻ⁻, ℰ₀))

    return 1/Vᶜᶠᶜ(i, j, k, grid) * (Φᵟ + (Fᶻ⁺ - Fᶻ⁻))
end
