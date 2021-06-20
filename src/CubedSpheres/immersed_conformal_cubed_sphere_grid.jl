import Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid

function ImmersedBoundaryGrid(grid::ConformalCubedSphereGrid, immersed_boundary)
    faces = Tuple(ImmersedBoundaryGrid(get_face(grid, i), immersed_boundary) for i = 1:6)
    FT = eltype(grid)
    face_connectivity = grid.face_connectivity

    cubed_sphere_immersed_grid = ConformalCubedSphereGrid{FT, typeof(faces), typeof(face_connectivity)}(faces, face_connectivity)
     
    return cubed_sphere_immersed_grid 
end

import Oceananigans.Operators: Γᶠᶠᵃ

@inline function Γᶠᶠᵃ(i, j, k, ibg::ImmersedBoundaryGrid{F, TX, TY, TZ, G, I}, u, v) where {F,TX,TY,TZ,G<:ConformalCubedSphereFaceGrid,I}
    Γᶠᶠᵃ(i, j, k, ibg.grid, u, v)
end
