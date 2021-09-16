import Oceananigans: short_show

location_str(::Type{Face})    = "Face"
location_str(::Type{Center})  = "Center"
location_str(::Type{Nothing}) = "⋅"

function show_size(field)
    Nx, Ny, Nz = size(field)
    return "$Nx×$Ny×$Nz"
end

show_location(X, Y, Z) = "($(location_str(X)), $(location_str(Y)), $(location_str(Z)))"

show_location(field::AbstractField{X, Y, Z}) where {X, Y, Z} = show_location(X, Y, Z)

short_show(m::Missing) = "$m"

short_show(field::AbstractField) = string(show_size(field), " ", typeof(field).name.wrapper, " located at ", show_location(field))
short_show(field::AveragedField) = string(show_size(field), " ", "AveragedField over dims=$(field.dims) located at ", show_location(field), " of ", short_show(field.operand))
short_show(field::ComputedField) = string(show_size(field), " ", "ComputedField located at ", show_location(field), " of ", short_show(field.operand))

Base.show(io::IO, field::AbstractField{X, Y, Z, A}) where {X, Y, Z, A} =
    print(io, "$(short_show(field))\n",
          "├── architecture: $A\n",
          "└── grid: $(short_show(field.grid))")

function Base.show(io::IO, field::Field)
    print(io, "$(short_show(field))", '\n',
          "├── data: ", summary(field.data), '\n',
          "├── grid: $(short_show(field.grid))", '\n',
          "└── boundary conditions: $(short_show(field.boundary_conditions))")

    return nothing
end

show_status(::Nothing) = "nothing"
show_status(status) = "time=$(status.time)"

function Base.show(io::IO, field::AveragedField)
    print(io, "$(short_show(field))", '\n',
          "├── data: ", summary(field.data), '\n',
          "├── grid: $(short_show(field.grid))", '\n',
          "├── dims: $(field.dims)", '\n',
          "├── operand: $(short_show(field.operand))", '\n',
          "└── status: ", show_status(field.status))

    return nothing
end

function Base.show(io::IO, field::ComputedField)
    print(io, "$(short_show(field))", '\n',
          "├── data: ", summary(field.data), '\n',
          "├── grid: $(short_show(field.grid))", '\n',
          "├── operand: $(short_show(field.operand))", '\n',
          "└── status: $(show_status(field.status))")

    return nothing
end

function Base.show(io::IO, field::KernelComputedField)
    print(io, "$(short_show(field))", '\n',
          "├── data: ", summary(field.data), '\n',
          "├── grid: $(short_show(field.grid))", '\n',
          "├── computed_dependencies: $(Tuple(short_show(d) for d in field.computed_dependencies))", '\n',
          "├── kernel: $(short_show(field.kernel))", '\n',
          "└── status: $(show_status(field.status))")

    return nothing
end

Base.show(io::IO, field::ZeroField) = print(io, "ZeroField")

short_show(array::OffsetArray{T, D, A}) where {T, D, A} = string("OffsetArray{$T, $D, $A}")

Base.show(io::IO, ::MIME"text/plain", f::AbstractField) = show(io, f)
