#####
##### Momentum advection by GhostCells WENO on immersed grids: the velocities carry the location of their nodes, at which
##### the stencils look for peripheral nodes
#####

const VectorInvariantVerticalGhostCells = VectorInvariant{<:Any, <:Any, <:Any, <:Any, <:Any, <:GhostCellWENO}

for divergence in (:div_𝐯u, :div_𝐯v, :div_𝐯w)
    @eval begin
        @inline $divergence(i, j, k, grid::ImmersedBoundaryGrid, ::GhostCellWENO, ::ZeroU, ψ::AbstractArray) = zero(grid)
        @inline $divergence(i, j, k, grid::ImmersedBoundaryGrid, ::GhostCellWENO, U,       ::ZeroField)     = zero(grid)
        @inline $divergence(i, j, k, grid::ImmersedBoundaryGrid, ::GhostCellWENO, ::ZeroU, ::ZeroField)     = zero(grid)
    end
end

@inline div_𝐯u(i, j, k, grid::ImmersedBoundaryGrid, advection::GhostCellWENO, U, u::AbstractArray) = div_𝐯u(i, j, k, grid, advection, U, LocatedField(u, (Face(), Center(), Center())))
@inline div_𝐯v(i, j, k, grid::ImmersedBoundaryGrid, advection::GhostCellWENO, U, v::AbstractArray) = div_𝐯v(i, j, k, grid, advection, U, LocatedField(v, (Center(), Face(), Center())))
@inline div_𝐯w(i, j, k, grid::ImmersedBoundaryGrid, advection::GhostCellWENO, U, w::AbstractArray) = div_𝐯w(i, j, k, grid, advection, U, LocatedField(w, (Center(), Center(), Face())))

@inline function vertical_advection_U(i, j, k, grid::ImmersedBoundaryGrid, scheme::VectorInvariantVerticalGhostCells, U)
    u  = LocatedField(U.u, (Face(), Center(), Center()))
    Φᵟ = upwinded_divergence_flux_Uᶠᶜᶜ(i, j, k, grid, scheme, U.u, U.v)
    𝒜ᶻ = δzᵃᵃᶜ(i, j, k, grid, _advective_momentum_flux_Wu, scheme.vertical_advection_scheme, U.w, u)

    return 1/Vᶠᶜᶜ(i, j, k, grid) * (Φᵟ + 𝒜ᶻ)
end

@inline function vertical_advection_V(i, j, k, grid::ImmersedBoundaryGrid, scheme::VectorInvariantVerticalGhostCells, U)
    v  = LocatedField(U.v, (Center(), Face(), Center()))
    Φᵟ = upwinded_divergence_flux_Vᶜᶠᶜ(i, j, k, grid, scheme, U.u, U.v)
    𝒜ᶻ = δzᵃᵃᶜ(i, j, k, grid, _advective_momentum_flux_Wv, scheme.vertical_advection_scheme, U.w, v)

    return 1/Vᶜᶠᶜ(i, j, k, grid) * (Φᵟ + 𝒜ᶻ)
end
