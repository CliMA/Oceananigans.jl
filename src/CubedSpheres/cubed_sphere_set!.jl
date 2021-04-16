using Oceananigans.Architectures: AbstractCPUArchitecture
using Oceananigans.Fields: AbstractField

import Oceananigans.Fields: set!

const CPUCubedSphereField = AbstractField{X, Y, Z, <:CPU, <:ConformalCubedSphereGrid} where {X, Y, Z}
const CPUCubedSphereFaceField = AbstractField{X, Y, Z, <:CPU, <:ConformalCubedSphereFaceGrid} where {X, Y, Z}

function set!(u::CPUCubedSphereFaceField, v::CPUCubedSphereFaceField)
    for (u_face, v_face) in zip(u.faces, v.faces)
        @. u_face.data.parent = v_face.data.parent
    end
    return nothing
end

set!(field::CPUCubedSphereField, f::Function) = [set!(field_face, f) for field_face in faces(field)]

function set!(field::CPUCubedSphereFaceField{LX, LY, LZ}, f::Function) where {LX, LY, LZ}
    grid = field.grid

    for i in 1:grid.Nx, j in 1:grid.Ny, k in 1:grid.Nz
        λ = λnode(LX(), LY(), LZ(), i, j, k, grid)
        φ = φnode(LX(), LY(), LZ(), i, j, k, grid)
        z = znode(LX(), LY(), LZ(), i, j, k, grid)
        field[i, j, k] = f(λ, φ, z)
    end

    return nothing
end
