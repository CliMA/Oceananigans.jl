import Base: size, getindex, similar, *, +, -

const facenum = Dict(:x=>1, :y=>2, :z=>3)

"""
    CellField(grid::Grid)

Construct a `CellField` whose values are defined at the center of a cell.
"""
struct CellField{T,G<:Grid{T}} <: Field{G}
    data::AbstractArray{T}
    grid::G
end

function CellField(grid::Grid{T}) where T
    sz = size(grid)
    data = zeros(T, sz)
    CellField(data, grid)
end

similar(f::CellField) = CellField(f.grid)

set!(u, v) = @. u.data = v
set!(u, v::Field) = @. u.data = v.data
set!(u::Field{G}, f::Function) where {G<:RegularCartesianGrid} = @. u.data = f(u.grid.xCA, u.grid.yCA, u.grid.zCA)


"""
    FaceField(grid::Grid, face)

Construct a `FaceField` whose values are defined on the face of a cell. `face` is an integer that 
encodes the face where the field is defined. The `x`, `y`, and `z` faces correspond to `face=1`, `2`, and `3`
respectively.
"""
struct FaceField{T,G<:Grid{T},F} <: Field{G}
    data::AbstractArray{T}
    grid::G
    function FaceField(grid::Grid{T}, face::Int) where T
      sz = [size(grid, d) for d = 1:dim]
      sz[face] += 1 
      data = zeros(T, Tuple(sz))
      new{T, typeof(grid), face}(data, grid)
    end
end

FaceField(grid, face::Symbol) = FaceField(grid, facenum[face])

similar(f::FaceField{T,G,F}) where {T,G,F} = FaceField(f.grid, F)

function size(f::FaceField{T,G,F}) where {T,G,F}
  sz = [size(f.grid, d) for d=1:dim]
  sz[F] += 1
  Tuple(sz)
end

# Define algebraic operations on fields as element-wise calculations on their data.
for op in (:+, :-, :*)
  @eval begin
    function $op(f1::Field, f2::Field)
      f3 = similar(f1)
      @. f3.data = $op(f1.data, f2.data)
      f3
    end
  end
end

size(f::CellField) = size(f.grid)

getindex(f::Field, inds...) = getindex(f.data, inds...)

∂x!(df, f, dir) = throw(NotImplementedError())
∂y!(df, f, dir) = throw(NotImplementedError())
∂z!(df, f, dir) = throw(NotImplementedError())

function ∂x!(df::FaceField{T,G,F}, f::CellField) where {T,G<:RegularCartesianGrid,F}
  # derivative of f wrt to x
  nothing
end

function ∂y!(df::FaceField{T,G,F}, f::CellField) where {T,G<:RegularCartesianGrid,F}
  # derivative of f wrt to y
  nothing
end

function ∂z!(df::FaceField{T,G,F}, f::CellField) where {T,G<:RegularCartesianGrid,F}
  # derivative of f wrt to z
  nothing
end










#=
struct ZoneField{T <: AbstractFloat} <: Field
    f::Array
end

struct FaceField{T <: AbstractFloat} <: Field
    f::Array
end

function ZoneField(g::RegularCartesianGrid, T=Float64)
    f = Array{T,g.dim}(undef, g.Nx, g.Ny, g.Nz)
    ZoneField{T}(f)
end

function FaceField(g::RegularCartesianGrid, T=Float64)
    f = Array{T,g.dim}(undef, g.Nx + 1, g.Ny + 1, g.Nz + 1)
    FaceField{T}(f)
end

struct Fields{T <: AbstractFloat} <: FieldCollection
    u::FaceField{T}
    v::FaceField{T}
    w::FaceField{T}
    T::ZoneField{T}
    ρ::ZoneField{T}
    p::ZoneField{T}
    S::ZoneField{T}
    pHY::ZoneField{T}
    pHY′::ZoneField{T}
    pNHS::ZoneField{T}
end

function Fields(g::RegularCartesianGrid, T=Float64)
    u = FaceField(g, T)
    v = FaceField(g, T)
    w = FaceField(g, T)
    θ = ZoneField(g, T)
    S = ZoneField(g, T)
    ρ = ZoneField(g, T)
    p = ZoneField(g, T)
    pHY = ZoneField(g, T)
    pHY′ = ZoneField(g, T)
    pNHS = ZoneField(g, T)
    Fields(u, v, w, θ, S, ρ, p, pHY, pHY′, pNHS)
end

struct SourceTermFields{T <: AbstractFloat} <: FieldCollection
    Gu::FaceField{T}
    Gv::FaceField{T}
    Gw::FaceField{T}
    Gθ::ZoneField{T}
    GS::ZoneField{T}
end

function SourceTermFields(g::RegularCartesianGrid, T=Float64)
    Gu = FaceField(g, T)
    Gv = FaceField(g, T)
    Gw = FaceField(g, T)
    Gθ = ZoneField(g, T)
    GS = ZoneField(g, T)
    SourceTermFields{T}(Gu, Gv, Gw, Gθ, GS)
end

struct ForcingFields{T <: AbstractFloat} <: FieldCollection
    Fu::FaceField{T}
    Fv::FaceField{T}
    Fw::FaceField{T}
    Fθ::ZoneField{T}
    FS::ZoneField{T}
end

function ForcingFields(g::RegularCartesianGrid, T=Float64)
    Fu = FaceField(g, T)
    Fv = FaceField(g, T)
    Fw = FaceField(g, T)
    Fθ = ZoneField(g, T)
    FS = ZoneField(g, T)
    ForcingFields{T}(Fu, Fv, Fw, Fθ, FS)
end
=#
