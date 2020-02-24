using Base: @propagate_inbounds
import Base: getindex

using Oceananigans.Operators: ℑxᶠᵃᵃ, ℑyᵃᶠᵃ, ℑzᵃᵃᶠ
using Oceananigans.Fields

struct LazyPrimitiveField{X, Y, Z, A, G, C, D} <: AbstractField{X, Y, Z, A, G}
          architecture :: A
                  grid :: G
    conservative_field :: C
               density :: D
end

LazyPrimitiveField(LX, LY, LZ, arch, grid, ρϕ, ρ) =
    LazyPrimitiveField{LX, LY, LZ, typeof(arch), typeof(grid), typeof(ρϕ), typeof(ρ)}(arch, grid, ρϕ, ρ)

@inline @propagate_inbounds getindex(f::LazyPrimitiveField{Cell, Cell, Cell}, I...) =
    @inbounds f.conservative_field[I...] / f.density[I...]

@inline @propagate_inbounds getindex(f::LazyPrimitiveField{Face, Cell, Cell}, I...) =
    @inbounds f.conservative_field[I...] / ℑxᶠᵃᵃ(I..., f.grid, f.density)

@inline @propagate_inbounds getindex(f::LazyPrimitiveField{Cell, Face, Cell}, I...) =
    @inbounds f.conservative_field[I...] / ℑyᵃᶠᵃ(I..., f.grid, f.density)

@inline @propagate_inbounds getindex(f::LazyPrimitiveField{Cell, Cell, Face}, I...) =
    @inbounds f.conservative_field[I...] / ℑzᵃᵃᶠ(I..., f.grid, f.density)

LazyVelocityFields(arch, grid, ρ, ρũ) =
    (u = LazyPrimitiveField(Face, Cell, Cell, arch, grid, ρũ.ρu.data, ρ.data),
     v = LazyPrimitiveField(Cell, Face, Cell, arch, grid, ρũ.ρv.data, ρ.data),
     w = LazyPrimitiveField(Cell, Cell, Face, arch, grid, ρũ.ρw.data, ρ.data))

function LazyTracerFields(arch, grid, ρ, ρc̃)
    c_names = [filter(c -> c != 'ρ', string(c)) for c in keys(ρc̃)]
    c_names = filter(s -> s != "", c_names) .|> Symbol |> Tuple  # Don't include the ρ tracer.

    c_fields = Tuple(
        LazyPrimitiveField(Cell, Cell, Cell, arch, grid, getproperty(ρc̃, Symbol(:ρ, c)).data, ρ.data)
        for c in c_names
    )

    return NamedTuple{c_names}(c_fields)
end
