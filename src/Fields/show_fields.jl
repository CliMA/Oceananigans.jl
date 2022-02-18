using Printf
using Oceananigans.Grids: size_summary, scalar_summary
using Oceananigans.BoundaryConditions: bc_str

location_str(::Type{Face})    = "Face"
location_str(::Type{Center})  = "Center"
location_str(::Type{Nothing}) = "⋅"
show_location(LX, LY, LZ) = "($(location_str(LX)), $(location_str(LY)), $(location_str(LZ)))"
show_location(field::AbstractField) = show_location(location(field)...)

function Base.summary(field::Field)
    LX, LY, LZ = location(field)
    prefix = string(size_summary(size(field)), " Field{$LX, $LY, $LZ}")

    grid_name = typeof(field.grid).name.wrapper
    reduced_dims = reduced_dimensions(field)

    suffix = reduced_dims === () ?
        string(" on ", grid_name, " on ", summary(architecture(field))) :
        string(" reduced over dims = ", reduced_dims,
               " on ", grid_name, " on ", summary(architecture(field)))

    return string(prefix, suffix)
end

data_summary(field) = string("max=", scalar_summary(maximum(field)), ", ",
                             "min=", scalar_summary(minimum(field)), ", ",
                             "mean=", scalar_summary(mean(field)))

function Base.show(io::IO, field::Field)

    bcs = field.boundary_conditions

    prefix =
        string("$(summary(field))\n",
               "├── grid: ", summary(field.grid), '\n',
               "├── boundary conditions: ", summary(bcs), '\n',
               "│   └── west: ", bc_str(bcs.west), ", east: ", bc_str(bcs.east),
                     " south: ", bc_str(bcs.south), ", north: ", bc_str(bcs.north),
                     " bottom: ", bc_str(bcs.bottom), ", top: ", bc_str(bcs.top),
                     " immersed: ", bc_str(bcs.immersed), '\n')

    middle = isnothing(field.operand) ? "" :
        string("├── operand: ", summary(field.operand), '\n',
               "├── status: ", summary(field.status), '\n')

    suffix = string("└── data: ", summary(field.data), '\n',
                    "    └── ", data_summary(field))

    print(io, prefix, middle, suffix)
end

Base.summary(status::FieldStatus) = "time=$(status.time)"

Base.summary(::ZeroField{N}) where N = "ZeroField{$N}"
Base.show(io::IO, z::ZeroField) = print(io, summary(z))

Base.show(io::IO, ::MIME"text/plain", f::AbstractField) = show(io, f)

