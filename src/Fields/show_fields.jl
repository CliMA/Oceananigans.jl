location_str(::Type{Face})    = "Face"
location_str(::Type{Center})  = "Center"
location_str(::Type{Nothing}) = "⋅"

show_location(LX, LY, LZ) = "($(location_str(LX)), $(location_str(LY)), $(location_str(LZ)))"

show_location(field::AbstractField{LX, LY, LZ}) where {LX, LY, LZ} = show_location(LX, LY, LZ)

short_show(m::Missing) = "$m"

short_show(field::AbstractField) = string(typeof(field).name.wrapper, " located at ", show_location(field))

Base.show(io::IO, field::AbstractField{LX, LY, LZ}) where {LX, LY, LZ} =
    print(io, "$(short_show(field))\n",
          "├── architecture: $(architecture(field))\n",
          "└── grid: $(short_show(field.grid))")

Base.show(io::IO, field::Field) =
    print(io, "$(short_show(field))\n",
          "├── data: $(typeof(field.data)), size: $(size(field))\n",
          "├── grid: $(short_show(field.grid))\n",
          "└── boundary conditions: $(short_show(field.boundary_conditions))")

show_status(::Nothing) = "nothing"
show_status(status) = "time=$(status.time)"

Base.show(io::IO, field::ZeroField) = print(io, "ZeroField")

short_show(array::OffsetArray{T, D, A}) where {T, D, A} = string("OffsetArray{$T, $D, $A}")

Base.show(io::IO, ::MIME"text/plain", f::AbstractField) = show(io, f)

