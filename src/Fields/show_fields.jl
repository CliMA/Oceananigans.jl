show_location(X, Y, Z) = string("(", string(typeof(X())), ", ",
                                     string(typeof(Y())), ", ",
                                     string(typeof(Z())), ")")

show_location(field::AbstractLocatedField{X, Y, Z}) where {X, Y, Z} = show_location(X, Y, Z)

short_show(a) = string(typeof(a))
shortname(a::Array) = string(typeof(a).name.wrapper)

show(io::IO, field::Field) =
    print(io,
          short_show(field), '\n',
          "├── data: ", typeof(field.data), '\n',
          "└── grid: ", typeof(field.grid), '\n',
          "    ├── size: ", size(field.grid), '\n',
          "    └── domain: ", show_domain(field.grid), '\n')

short_show(field::AbstractLocatedField) = string("Field at ", show_location(field))

short_show(array::OffsetArray{T, D, A}) where {T, D, A} = string("OffsetArray{$T, $D, $A}")
