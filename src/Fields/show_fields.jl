import Base: show

using Oceananigans.Grids: show_domain

show_location(X, Y, Z) = "($(typeof(X())), $(typeof(Y())), $(typeof(Z())))"

show_location(field::AbstractField{X, Y, Z}) where {X, Y, Z} = show_location(X, Y, Z)

short_show(field::Field) = string("Field at ", show_location(field))

show(io::IO, field::Field{X, Y, Z}) where {X, Y, Z} =
    print(io, "$(short_show(field))\n",
          "├── data: $(typeof(field.data)), size: $(size(field.data))\n",
          "└── grid: $(typeof(field.grid))\n",
          "    ├── size: $(size(field.grid))\n",
          "    └── domain: $(show_domain(field.grid))")
