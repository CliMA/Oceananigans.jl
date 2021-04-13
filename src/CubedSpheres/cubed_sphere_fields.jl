using Statistics

using Oceananigans.Fields: AbstractField, call_func

import Base: getindex, size, show, minimum, maximum
import Statistics: mean
import Oceananigans.Fields: Field, CenterField, XFaceField, YFaceField, ZFaceField, FunctionField, ReducedField, interior
import Oceananigans.BoundaryConditions: fill_halo_regions!

abstract type AbstractCubedSphereField{X, Y, Z, A, G} <: AbstractField{X, Y, Z, A, G} end

struct ConformalCubedSphereField{X, Y, Z, A, G, F, B} <: AbstractCubedSphereField{X, Y, Z, A, G}
                   grid :: G
                  faces :: F
    boundary_conditions :: B
end

#####
##### Regular fields
#####

Field(X, Y, Z, arch, grid::ConformalCubedSphereGrid, ::Nothing, data) =
    Field(X, Y, Z, arch, grid, FieldBoundaryConditions(grid, (X, Y, Z)), data)

function Field(X, Y, Z, arch, grid::ConformalCubedSphereGrid, bcs, data)

    faces = Tuple(
        Field(X, Y, Z, arch, face_grid, inject_cubed_sphere_exchange_boundary_conditions(bcs, face_number, grid.face_connectivity))
        for (face_number, face_grid) in enumerate(grid.faces)
    )

    # This field needs BCs otherwise errors happen so I'll assume all faces have
    # the same boundary conditions. A very bad assumption...
    cubed_sphere_bcs = faces[1].boundary_conditions

    return ConformalCubedSphereField{X, Y, Z, typeof(arch), typeof(grid), typeof(faces), typeof(cubed_sphere_bcs)}(grid, faces, cubed_sphere_bcs)
end

#####
##### Convinient field constructors
#####

# Gotta short-circuit for `::ConformalCubedSphereGrid` because these
# Field constructors allocate `data` in the function signature.

CenterField(FT::DataType, arch, grid::ConformalCubedSphereGrid, bcs=nothing, data=nothing) = Field(Center, Center, Center, arch, grid, bcs, data)
 XFaceField(FT::DataType, arch, grid::ConformalCubedSphereGrid, bcs=nothing, data=nothing) = Field(Face,   Center, Center, arch, grid, bcs, data)
 YFaceField(FT::DataType, arch, grid::ConformalCubedSphereGrid, bcs=nothing, data=nothing) = Field(Center, Face,   Center, arch, grid, bcs, data)
 ZFaceField(FT::DataType, arch, grid::ConformalCubedSphereGrid, bcs=nothing, data=nothing) = Field(Center, Center, Face,   arch, grid, bcs, data)

#####
##### Utils
#####

Base.size(field::AbstractCubedSphereField) = (size(field.faces[1])..., length(field.faces))

Base.minimum(field::AbstractCubedSphereField; dims=:) = minimum(minimum(field_face; dims) for field_face in field.faces)
Base.maximum(field::AbstractCubedSphereField; dims=:) = maximum(maximum(field_face; dims) for field_face in field.faces)
Statistics.mean(field::AbstractCubedSphereField; dims=:) = mean(mean(field_face; dims) for field_face in field.faces)

Base.minimum(f, field::AbstractCubedSphereField; dims=:) = minimum(minimum(f, field_face; dims) for field_face in field.faces)
Base.maximum(f, field::AbstractCubedSphereField; dims=:) = maximum(maximum(f, field_face; dims) for field_face in field.faces)
Statistics.mean(f, field::AbstractCubedSphereField; dims=:) = mean(mean(f, field_face; dims) for field_face in field.faces)

interior(field::AbstractCubedSphereField) = cat(Tuple(interior(field_face) for field_face in field.faces)..., dims=4)

const ConformalCubedSphereFaceField{LX, LY, LZ, A} = AbstractField{LX, LY, LZ, A, <:ConformalCubedSphereFaceGrid}

λnodes(field::ConformalCubedSphereFaceField{LX, LY, LZ}) where {LX, LY, LZ} = λnodes(LX(), LY(), LZ(), field.grid)
φnodes(field::ConformalCubedSphereFaceField{LX, LY, LZ}) where {LX, LY, LZ} = φnodes(LX(), LY(), LZ(), field.grid)

λnodes(field::ConformalCubedSphereField{LX, LY, LZ}) where {LX, LY, LZ} = λnodes(LX(), LY(), LZ(), field.grid)
φnodes(field::ConformalCubedSphereField{LX, LY, LZ}) where {LX, LY, LZ} = φnodes(LX(), LY(), LZ(), field.grid)

#####
##### Function fields
#####

struct ConformalCubedSphereFunctionField{X, Y, Z, F, G} <: AbstractCubedSphereField{X, Y, Z, F, G}
    faces :: F
end

function FunctionField{X, Y, Z}(func, grid::ConformalCubedSphereGrid; clock=nothing, parameters=nothing) where {X, Y, Z}

    faces = Tuple(FunctionField{X, Y, Z}(func, face_grid; clock, parameters) for face_grid in grid.faces)

    return ConformalCubedSphereFunctionField{X, Y, Z, typeof(faces), typeof(grid)}(faces)
end

const ConformalCubedSphereFaceFunctionField = FunctionField{X, Y, Z, C, P, F, <:ConformalCubedSphereFaceGrid} where {X, Y, Z, C, P, F}

@inline Base.getindex(f::ConformalCubedSphereFaceFunctionField{X, Y, Z}, i, j, k) where {X, Y, Z} =
    call_func(f.clock, f.parameters, f.func,
              λnode(X(), Y(), Z(), i, j, k, f.grid),
              φnode(X(), Y(), Z(), i, j, k, f.grid),
              znode(Z(), Y(), Z(), i, j, k, f.grid))

fill_halo_regions!(::ConformalCubedSphereFunctionField, args...) = nothing

#####
##### Reduced fields
#####

struct ConformalCubedSphereReducedField{X, Y, Z, A, G, F, B} <: AbstractCubedSphereField{X, Y, Z, A, G}
                   grid :: G
                  faces :: F
    boundary_conditions :: B
end

function ReducedField(X, Y, Z, arch, grid::ConformalCubedSphereGrid; dims, data=nothing, boundary_conditions=nothing)

    faces = Tuple(
        ReducedField(X, Y, Z, arch, grid_face, dims=dims, data=data, boundary_conditions=inject_cubed_sphere_exchange_boundary_conditions(FieldBoundaryConditions(grid_face, (X, Y, Z)), face_number, grid.face_connectivity))
        for (face_number, grid_face) in enumerate(grid.faces)
    )

    # This field needs BCs otherwise errors happen so I'll assume all faces have
    # the same boundary conditions. A very bad assumption...
    cubed_sphere_bcs = faces[1].boundary_conditions

    return ConformalCubedSphereReducedField{X, Y, Z, typeof(arch), typeof(grid), typeof(faces), typeof(cubed_sphere_bcs)}(grid, faces, cubed_sphere_bcs)
end

#####
##### Pretty printing
#####

function Base.show(io::IO, field::AbstractCubedSphereField{X, Y, Z}) where {X, Y, Z}
    Nx, Ny, Nz, Nf = size(field)
    print(io, "$(Base.typename(typeof(field))){$X, $Y, $Z}: $Nf faces with size = ($Nx, $Ny, $Nz)")
end
