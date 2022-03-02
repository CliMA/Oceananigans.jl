using Printf
using Oceananigans.Grids: size_summary, scalar_summary

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

    prefix =
        string("$(summary(field))\n",
               "├── grid: ", summary(field.grid), '\n',
               "├── boundary conditions: ", summary(field.boundary_conditions), '\n')

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

const FieldTuple = NamedTuple{S, <:NTuple{N, Field}} where {S, N}

function Base.show(io::IO, ft::FieldTuple)
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
        print(io, "on different grids:", '\n')
    end

    for name in names[1:end-1]
        field = ft[name]
        print(io, "├── $name: ", summary(field), '\n')

        if !all_same_grid
            print(io, "│   └── grid: ", summary(field.grid), '\n')
        end
    end

    name = names[end]
    field = ft[name]
    print(io, "└── $name: ", summary(field))

    if !all_same_grid
        print(io, '\n')
        print(io, "    └── grid: ", summary(field.grid))
    end
end

