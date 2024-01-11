using Printf
using Oceananigans.Grids: size_summary
using Oceananigans.Utils: prettysummary
using Oceananigans.BoundaryConditions: bc_str

import Oceananigans.Grids: grid_name

location_str(::Type{Face})    = "Face"
location_str(::Type{Center})  = "Center"
location_str(::Type{Nothing}) = "⋅"
show_location(LX, LY, LZ) = "($(location_str(LX)), $(location_str(LY)), $(location_str(LZ)))"
show_location(field::AbstractField) = show_location(location(field)...)

grid_name(field::Field) = grid_name(field.grid)

function Base.summary(field::Field)
    LX, LY, LZ = location(field)
    prefix = string(size_summary(size(field)), " Field{$LX, $LY, $LZ}")

    reduced_dims = reduced_dimensions(field)

    suffix = reduced_dims === () ?
        string(" on ", grid_name(field), " on ", summary(architecture(field))) :
        string(" reduced over dims = ", reduced_dims,
               " on ", grid_name(field), " on ", summary(architecture(field)))

    return string(prefix, suffix)
end

data_summary(field) = string("max=", prettysummary(maximum(field)), ", ",
                             "min=", prettysummary(minimum(field)), ", ",
                             "mean=", prettysummary(mean(field)))

indices_summary(field) = replace(string(field.indices), "Colon()"=> ":")

function Base.show(io::IO, field::Field)

    bcs = field.boundary_conditions

    prefix = string("$(summary(field))\n",
                    "├── grid: ", summary(field.grid), "\n")

    bcs_str = isnothing(bcs) ? "├── boundary conditions: Nothing \n" :
        string("├── boundary conditions: ", summary(bcs), "\n",
        "│   └── west: ", bc_str(bcs.west), ", east: ", bc_str(bcs.east),
               ", south: ", bc_str(bcs.south), ", north: ", bc_str(bcs.north),
               ", bottom: ", bc_str(bcs.bottom), ", top: ", bc_str(bcs.top),
               ", immersed: ", bc_str(bcs.immersed), "\n")

    indices_str = indices_summary(field) == "(:, :, :)" ?
                      "" :
                      string("├── indices: ", indices_summary(field), "\n")

    operand_str = isnothing(field.operand) ? "" :
                      string("├── operand: ", summary(field.operand), "\n",
                             "├── status: ", summary(field.status), "\n")

    data_str = string("└── data: ", summary(field.data), "\n",
                      "    └── ", data_summary(field))

    print(io, prefix, bcs_str, indices_str, operand_str, data_str)
end

Base.summary(status::FieldStatus) = "time=$(status.time)"

Base.summary(::ZeroField{N}) where N = "ZeroField{$N}"
Base.summary(::OneField{N}) where N  = "OneField{$N}"

Base.show(io::IO, z::Union{ZeroField, OneField}) = print(io, summary(z))

@inline Base.summary(f::CF) = string("ConstantField(", prettysummary(f.constant), ")")
Base.show(io::IO, f::CF) = print(io, summary(f))

Base.show(io::IO, ::MIME"text/plain", f::AbstractField) = show(io, f)

const FieldTuple = Tuple{Field, Vararg{Field}}
const NamedFieldTuple = NamedTuple{S, <:FieldTuple} where S

function Base.show(io::IO, ft::NamedFieldTuple)
    names = keys(ft)
    N = length(ft)

    grid = first(ft).grid
    all_same_grid = true
    for field in ft
        if field.grid !== grid
            all_same_grid = false
        end
    end

    print(io, "NamedTuple with ", N, " Fields ")

    if all_same_grid
        print(io, "on ", summary(grid), ":\n")
    else
        print(io, "on different grids:", "\n")
    end

    for name in names[1:end-1]
        field = ft[name]
        print(io, "├── $name: ", summary(field), "\n")

        if !all_same_grid
            print(io, "│   └── grid: ", summary(field.grid), "\n")
        end
    end

    name = names[end]
    field = ft[name]
    print(io, "└── $name: ", summary(field))

    if !all_same_grid
        print(io, "\n")
        print(io, "    └── grid: ", summary(field.grid))
    end
end
