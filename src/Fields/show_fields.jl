import Base: show
import Oceananigans.Grids: short_show
using Oceananigans.Grids: show_domain

location_str(::Face) = "Face"
location_str(::Cell) = "Cell"
show_location(X, Y, Z) = "($(location_str(X())), $(location_str(Y())), $(location_str(Z())))"
show_location(field::AbstractField{X, Y, Z}) where {X, Y, Z} = show_location(X, Y, Z)

short_show(field::Field) = string("Field located at ", show_location(field))

show(io::IO, field::Field) =
    print(io, "$(short_show(field))\n",
          "├── data: $(typeof(field.data)), size: $(size(field.data))\n",
          "├── grid: $(short_show(field.grid))\n",
          "└── boundary conditions: $(short_show(field.boundary_conditions))")

short_show(array::OffsetArray{T, D, A}) where {T, D, A} = string("OffsetArray{$T, $D, $A}")