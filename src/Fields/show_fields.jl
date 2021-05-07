import Oceananigans: short_show

location_str(::Type{Face})    = "Face"
location_str(::Type{Center})  = "Center"
location_str(::Type{Nothing}) = "⋅"

show_location(X, Y, Z) = "($(location_str(X)), $(location_str(Y)), $(location_str(Z)))"

show_location(field::AbstractField{X, Y, Z}) where {X, Y, Z} = show_location(X, Y, Z)

short_show(m::Missing) = "$m"

short_show(field::AbstractField) = string(typeof(field).name.wrapper, " located at ", show_location(field))
short_show(field::AveragedField) = string("AveragedField over dims=$(field.dims) located at ", show_location(field), " of ", short_show(field.operand))
short_show(field::ComputedField) = string("ComputedField located at ", show_location(field), " of ", short_show(field.operand))

Base.show(io::IO, field::AbstractField{X, Y, Z, A, G}) where {X, Y, Z, A, G} =
    print(io, "$(short_show(field))\n",
          "├── architecture: $A\n",
          "└── grid: $(short_show(field.grid))")

Base.show(io::IO, field::Field) =
    print(io, "$(short_show(field))\n",
          "├── data: $(typeof(field.data)), size: $(size(field.data))\n",
          "├── grid: $(short_show(field.grid))\n",
          "└── boundary conditions: $(short_show(field.boundary_conditions))")

show_status(::Nothing) = "nothing"
show_status(status) = "time=$(status.time)"

Base.show(io::IO, field::AveragedField) =
    print(io, "$(short_show(field))\n",
          "├── data: $(typeof(field.data)), size: $(size(field.data))\n",
          "├── grid: $(short_show(field.grid))", '\n',
          "├── dims: $(field.dims)", '\n',
          "├── operand: $(short_show(field.operand))", '\n',
          "└── status: ", show_status(field.status), '\n')

Base.show(io::IO, field::ComputedField) =
    print(io, "$(short_show(field))\n",
          "├── data: $(typeof(field.data)), size: $(size(field.data))\n",
          "├── grid: $(short_show(field.grid))", '\n',
          "├── operand: $(short_show(field.operand))", '\n',
          "└── status: ", show_status(field.status), '\n')

Base.show(io::IO, field::KernelComputedField) =
    print(io, "$(short_show(field))\n",
          "├── data: $(typeof(field.data)), size: $(size(field.data))\n",
          "├── grid: $(short_show(field.grid))", '\n',
          "├── computed_dependencies: $(Tuple(short_show(d) for d in field.computed_dependencies))", '\n',
          "├── kernel: $(short_show(field.kernel))", '\n',
          "└── status: ", show_status(field.status), '\n')

short_show(array::OffsetArray{T, D, A}) where {T, D, A} = string("OffsetArray{$T, $D, $A}")

Base.show(io::IO, ::MIME"text/plain", f::AbstractField) = show(io, f)
