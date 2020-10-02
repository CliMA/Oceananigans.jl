using Adapt

"""
    Field{X, Y, Z, A, G, B} <: AbstractField{X, Y, Z, A, G}

A field defined at the location (`X`, `Y`, `Z`), each of which can be either `Cell`
or `Face`, and with data stored in a container of type `A` (typically an array).
The field is defined on a grid `G` and has field boundary conditions `B`.
"""
struct Field{X, Y, Z, A, G, B} <: AbstractField{X, Y, Z, A, G}
                   data :: A
                   grid :: G
    boundary_conditions :: B

    function Field{X, Y, Z}(data, grid, bcs) where {X, Y, Z}
        validate_field_data(X, Y, Z, data, grid)
        return new{X, Y, Z, typeof(data), typeof(grid), typeof(bcs)}(data, grid, bcs)
    end
end

"""
    Field(X, Y, Z, arch, grid, [  bcs = FieldBoundaryConditions(grid, (X, Y, Z)),
                                 data = new_data(arch, grid, (X, Y, Z)) ] )

Construct a `Field` on `grid` with `data` on architecture `arch` with
boundary conditions `bcs`. Each of `(X, Y, Z)` is either `Cell` or `Face` and determines
the field's location in `(x, y, z)`.

Example
=======

julia> ω = Field(Face, Face, Cell, CPU(), RegularCartesianmodel.grid)

"""
function Field(X, Y, Z, arch, grid,
                bcs = FieldBoundaryConditions(grid, (X, Y, Z)),
               data = new_data(eltype(grid), arch, grid, (X, Y, Z)))

    return Field{X, Y, Z}(data, grid, bcs)
end

#####
##### Convenience constructor for Field that uses a 3-tuple of locations rather than a list of locations:
#####

# Type "destantiation": convert Face() to Face and Cell() to Cell if needed.
destantiate(X) = typeof(X)
destantiate(X::DataType) = X

"""
    Field(L::Tuple, arch, grid, data, bcs)

Construct a `Field` at the location defined by the 3-tuple `L`,
whose elements are `Cell` or `Face`.
"""
Field(L::Tuple, args...) = Field(destantiate.(L)..., args...)

#####
##### Special constructors for tracers and velocity fields
#####

"""
    CellField([ FT=eltype(grid) ], arch::AbstractArchitecture, grid,
              [  bcs = TracerBoundaryConditions(grid),
                data = new_data(FT, arch, grid, (Cell, Cell, Cell) ] )

Return a `Field{Cell, Cell, Cell}` on architecture `arch` and `grid` containing `data`
with field boundary conditions `bcs`.
"""
function CellField(FT::DataType, arch, grid,
                    bcs = TracerBoundaryConditions(grid),
                   data = new_data(FT, arch, grid, (Cell, Cell, Cell)))

    return Field{Cell, Cell, Cell}(data, grid, bcs)
end

"""
    XFaceField([ FT=eltype(grid) ], arch::AbstractArchitecture, grid,
               [  bcs = UVelocityBoundaryConditions(grid),
                 data = new_data(FT, arch, grid, (Face, Cell, Cell) ] )

Return a `Field{Face, Cell, Cell}` on architecture `arch` and `grid` containing `data`
with field boundary conditions `bcs`.
"""
function XFaceField(FT::DataType, arch, grid,
                     bcs = UVelocityBoundaryConditions(grid),
                    data = new_data(FT, arch, grid, (Face, Cell, Cell)))

    return Field{Face, Cell, Cell}(data, grid, bcs)
end

"""
    YFaceField([ FT=eltype(grid) ], arch::AbstractArchitecture, grid,
               [  bcs = VVelocityBoundaryConditions(grid),
                 data = new_data(FT, arch, grid, (Cell, Face, Cell)) ] )

Return a `Field{Cell, Face, Cell}` on architecture `arch` and `grid` containing `data`
with field boundary conditions `bcs`.
"""
function YFaceField(FT::DataType, arch, grid,
                     bcs = VVelocityBoundaryConditions(grid),
                    data = new_data(FT, arch, grid, (Cell, Face, Cell)))

    return Field{Cell, Face, Cell}(data, grid, bcs)
end

"""
    ZFaceField([ FT=eltype(grid) ], arch::AbstractArchitecture, grid,
               [  bcs = WVelocityBoundaryConditions(grid),
                 data = new_data(FT, arch, grid, (Cell, Cell, Face)) ] )

Return a `Field{Cell, Cell, Face}` on architecture `arch` and `grid` containing `data`
with field boundary conditions `bcs`.
"""
function ZFaceField(FT::DataType, arch, grid,
                     bcs = WVelocityBoundaryConditions(grid),
                    data = new_data(FT, arch, grid, (Cell, Cell, Face)))

    return Field{Cell, Cell, Face}(data, grid, bcs)
end

 CellField(arch::AbstractArchitecture, grid, args...) =  CellField(eltype(grid), arch, grid, args...)
XFaceField(arch::AbstractArchitecture, grid, args...) = XFaceField(eltype(grid), arch, grid, args...)
YFaceField(arch::AbstractArchitecture, grid, args...) = YFaceField(eltype(grid), arch, grid, args...)
ZFaceField(arch::AbstractArchitecture, grid, args...) = ZFaceField(eltype(grid), arch, grid, args...)

@propagate_inbounds Base.setindex!(f::Field, v, inds...) = @inbounds setindex!(f.data, v, inds...)

gpufriendly(f::Field) = data(f)
