using Adapt

"""
    Field{X, Y, Z, A, G, B} <: AbstractField{X, Y, Z, A, G}

A field defined at the location (`X`, `Y`, `Z`), each of which can be either `Center`
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
boundary conditions `bcs`. Each of `(X, Y, Z)` is either `Center` or `Face` and determines
the field's location in `(x, y, z)`.

Example
=======

```jldoctest
julia> using Oceananigans, Oceananigans.Grids

julia> ω = Field(Face, Face, Center, CPU(), RegularRectilinearGrid(size=(1, 1, 1), extent=(1, 1, 1)))
Field located at (Face, Face, Center)
├── data: OffsetArrays.OffsetArray{Float64,3,Array{Float64,3}}, size: (3, 3, 3)
├── grid: RegularRectilinearGrid{Float64, Periodic, Periodic, Bounded}(Nx=1, Ny=1, Nz=1)
└── boundary conditions: x=(west=Periodic, east=Periodic), y=(south=Periodic, north=Periodic), z=(bottom=ZeroFlux, top=ZeroFlux)
```
"""
function Field(X, Y, Z, arch, grid,
                bcs = FieldBoundaryConditions(grid, (X, Y, Z)),
               data = new_data(eltype(grid), arch, grid, (X, Y, Z)))

    return Field{X, Y, Z}(data, grid, bcs)
end

#####
##### Convenience constructor for Field that uses a 3-tuple of locations rather than a list of locations:
#####

# Type "destantiation": convert Face() to Face and Center() to Center if needed.
destantiate(X) = typeof(X)
destantiate(X::DataType) = X

"""
    Field(L::Tuple, arch, grid, data, bcs)

Construct a `Field` at the location defined by the 3-tuple `L`,
whose elements are `Center` or `Face`.
"""
Field(L::Tuple, args...) = Field(destantiate.(L)..., args...)

#####
##### Special constructors for tracers and velocity fields
#####

"""
    CenterField([ FT=eltype(grid) ], arch::AbstractArchitecture, grid,
              [  bcs = TracerBoundaryConditions(grid),
                data = new_data(FT, arch, grid, (Center, Center, Center) ] )

Return a `Field{Center, Center, Center}` on architecture `arch` and `grid` containing `data`
with field boundary conditions `bcs`.
"""
function CenterField(FT::DataType, arch, grid,
                    bcs = TracerBoundaryConditions(grid),
                   data = new_data(FT, arch, grid, (Center, Center, Center)))

    return Field(Center, Center, Center, arch, grid, bcs, data)
end

"""
    XFaceField([ FT=eltype(grid) ], arch::AbstractArchitecture, grid,
               [  bcs = UVelocityBoundaryConditions(grid),
                 data = new_data(FT, arch, grid, (Face, Center, Center) ] )

Return a `Field{Face, Center, Center}` on architecture `arch` and `grid` containing `data`
with field boundary conditions `bcs`.
"""
function XFaceField(FT::DataType, arch, grid,
                     bcs = UVelocityBoundaryConditions(grid),
                    data = new_data(FT, arch, grid, (Face, Center, Center)))

    return Field(Face, Center, Center, arch, grid, bcs, data)
end

"""
    YFaceField([ FT=eltype(grid) ], arch::AbstractArchitecture, grid,
               [  bcs = VVelocityBoundaryConditions(grid),
                 data = new_data(FT, arch, grid, (Center, Face, Center)) ] )

Return a `Field{Center, Face, Center}` on architecture `arch` and `grid` containing `data`
with field boundary conditions `bcs`.
"""
function YFaceField(FT::DataType, arch, grid,
                     bcs = VVelocityBoundaryConditions(grid),
                    data = new_data(FT, arch, grid, (Center, Face, Center)))

    return Field(Center, Face, Center, arch, grid, bcs, data)
end

"""
    ZFaceField([ FT=eltype(grid) ], arch::AbstractArchitecture, grid,
               [  bcs = WVelocityBoundaryConditions(grid),
                 data = new_data(FT, arch, grid, (Center, Center, Face)) ] )

Return a `Field{Center, Center, Face}` on architecture `arch` and `grid` containing `data`
with field boundary conditions `bcs`.
"""
function ZFaceField(FT::DataType, arch, grid,
                     bcs = WVelocityBoundaryConditions(grid),
                    data = new_data(FT, arch, grid, (Center, Center, Face)))

    return Field(Center, Center, Face, arch, grid, bcs, data)
end

CenterField(arch::AbstractArchitecture, grid, args...) = CenterField(eltype(grid), arch, grid, args...)
 XFaceField(arch::AbstractArchitecture, grid, args...) =  XFaceField(eltype(grid), arch, grid, args...)
 YFaceField(arch::AbstractArchitecture, grid, args...) =  YFaceField(eltype(grid), arch, grid, args...)
 ZFaceField(arch::AbstractArchitecture, grid, args...) =  ZFaceField(eltype(grid), arch, grid, args...)

@propagate_inbounds Base.setindex!(f::Field, v, inds...) = @inbounds setindex!(f.data, v, inds...)

Adapt.adapt_structure(to, field::Field) = Adapt.adapt(to, field.data)
