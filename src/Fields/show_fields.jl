location_str(::Type{Face})    = "Face"
location_str(::Type{Center})  = "Center"
location_str(::Type{Nothing}) = "⋅"

show_location(LX, LY, LZ) = "($(location_str(LX)), $(location_str(LY)), $(location_str(LZ)))"

show_location(field::AbstractField{LX, LY, LZ}) where {LX, LY, LZ} = show_location(LX, LY, LZ)

Base.summary(m::Missing) = "$m"

Base.summary(field::AbstractField) = string(typeof(field).name.wrapper, " located at ", show_location(field))

Base.show(io::IO, field::AbstractField{LX, LY, LZ}) where {LX, LY, LZ} =
    print(io, "$(summary(field))\n",
          "├── architecture: $(architecture(field))\n",
          "└── grid: $(summary(field.grid))")

Base.show(io::IO, field::Field) =
    print(io, "$(summary(field))\n",
          "├── data: $(typeof(field.data)), size: $(size(field))\n",
          "├── grid: $(summary(field.grid))\n",
          "└── boundary conditions: $(summary(field.boundary_conditions))")

show_status(::Nothing) = "nothing"
show_status(status) = "time=$(status.time)"

Base.show(io::IO, field::ZeroField) = print(io, "ZeroField")

Base.show(io::IO, ::MIME"text/plain", f::AbstractField) = show(io, f)

