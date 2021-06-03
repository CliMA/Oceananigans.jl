import Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid

function ImmersedBoundaryGrid(grid::ConformalCubedSphereGrid, immersed_boundary)
    faces = Tuple(ImmersedBoundaryGrid(get_face(grid, i), immersed_boundary) for i = 1:6)
    FT = eltype(grid)
    face_connectivity = grid.face_connectivity

    cubed_sphere_immersed_grid = ConformalCubedSphereGrid{FT, typeof(faces), typeof(face_connectivity)}(faces, face_connectivity)
     
    return cubed_sphere_immersed_grid 
end

import Oceananigans.Grids: new_data

function new_data(FT, arch, ibg::ImmersedBoundaryGrid{F,TX,TY,TZ,G}, loc ) where {F,TX,TY,TZ,G<:ConformalCubedSphereGrid}
         println("Hello from IBG cube sphere new_data")
         return new_data(FT, arch, ibg.grid, loc )
end
