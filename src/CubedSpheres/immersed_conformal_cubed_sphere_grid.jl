import Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid

function ImmersedBoundaryGrid(grid::OldConformalCubedSphereGrid, immersed_boundary)
    faces = Tuple(ImmersedBoundaryGrid(get_face(grid, i), immersed_boundary) for i = 1:6)
    FT = eltype(grid)
    face_connectivity = grid.face_connectivity

    cubed_sphere_immersed_grid = OldConformalCubedSphereGrid{FT, typeof(faces), typeof(face_connectivity)}(faces, face_connectivity)
     
    return cubed_sphere_immersed_grid 
end

import Oceananigans.Operators: Γᶠᶠᶜ

@inline Γᶠᶠᶜ(i, j, k, ibg::ImmersedBoundaryGrid{F, TX, TY, TZ, G, I}, u, v) where
    {F, TX, TY, TZ, G<:OrthogonalSphericalShellGrid, I} = Γᶠᶠᶜ(i, j, k, ibg.grid, u, v)
