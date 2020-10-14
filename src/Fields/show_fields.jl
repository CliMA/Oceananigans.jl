import Base: show
import Oceananigans: short_show

location_str(::Type{Face})    = "Face"
location_str(::Type{Cell})    = "Cell"
location_str(::Type{Nothing}) = " -- "

show_location(X, Y, Z) = "($(location_str(X)), $(location_str(Y)), $(location_str(Z)))"

show_location(field::AbstractField{X, Y, Z}) where {X, Y, Z} = show_location(X, Y, Z)

short_show(m::Missing) = "$m"

short_show(field::Field) = string("Field located at ", show_location(field))

short_show(field::AveragedField) = string("AveragedField located at ", show_location(field), " of ", short_show(field.operand))
short_show(field::ComputedField) = string("ComputedField located at ", show_location(field), " of ", short_show(field.operand))

tree_show(field::AveragedField) = string("AveragedField located at ", show_location(field), " of ", tree_show(field.operand))
tree_show(field::ComputedField) = string("ComputedField located at ", show_location(field), " of ", tree_show(field.operand))

show(io::IO, field::Field) =
    print(io, "$(short_show(field))\n",
          "├── data: $(typeof(field.data)), size: $(size(field.data))\n",
          "├── grid: $(short_show(field.grid))\n",
          "└── boundary conditions: $(short_show(field.boundary_conditions))")

show(io::IO, field::AveragedField) =
    print(io, "$(short_show(field))\n",
          "├── data: $(typeof(field.data)), size: $(size(field.data))\n",
          "├── grid: $(short_show(field.grid))", '\n',
          "├── dims: $(field.dims)", '\n',
          "└── operand: $(short_show(field.operand))")

show(io::IO, field::ComputedField) =
    print(io, "$(short_show(field))\n",
          "├── data: $(typeof(field.data)), size: $(size(field.data))\n",
          "├── grid: $(short_show(field.grid))", '\n',
          "└── operand: $(short_show(field.operand))")

short_show(array::OffsetArray{T, D, A}) where {T, D, A} = string("OffsetArray{$T, $D, $A}")
