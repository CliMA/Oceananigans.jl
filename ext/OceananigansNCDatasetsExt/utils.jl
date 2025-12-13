#####
##### Utilities
#####

"""
    squeeze_data(fd::AbstractField, field_data; array_type=Array{eltype(fd)})

Returns the data of the field with the any dimensions where location is Nothing squeezed. For example:
```Julia
infil> grid = RectilinearGrid(size=(2,3,4), extent=(1,1,1))
2×3×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 2×3×3 halo
├── Periodic x ∈ [0.0, 1.0)  regularly spaced with Δx=0.5
├── Periodic y ∈ [0.0, 1.0)  regularly spaced with Δy=0.333333
└── Bounded  z ∈ [-1.0, 0.0] regularly spaced with Δz=0.25

infil> c = Field{Center, Center, Nothing}(grid)
2×3×1 Field{Center, Center, Nothing} reduced over dims = (3,) on RectilinearGrid on CPU
├── grid: 2×3×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 2×3×3 halo
├── boundary conditions: FieldBoundaryConditions
│   └── west: Periodic, east: Periodic, south: Periodic, north: Periodic, bottom: Nothing, top: Nothing, immersed: Nothing
└── data: 6×9×1 OffsetArray(::Array{Float64, 3}, -1:4, -2:6, 1:1) with eltype Float64 with indices -1:4×-2:6×1:1
    └── max=0.0, min=0.0, mean=0.0

infil> interior(c) |> size
(2, 3, 1)

infil> squeeze_data(c)
2×3 Matrix{Float64}:
 0.0  0.0  0.0
 0.0  0.0  0.0

infil> squeeze_data(c) |> size
(2, 3)
```

Note that this will only remove (squeeze) singleton dimensions.
"""
function squeeze_data(fd::AbstractField, field_data; array_type=Array{eltype(fd)})
    reduced_dims = effective_reduced_dimensions(fd)
    field_data_cpu = array_type(field_data) # Need to convert to the array type of the field

    indices = Any[:, :, :]
    for i in 1:3
        if i ∈ reduced_dims
            indices[i] = 1
        end
    end
    return getindex(field_data_cpu, indices...)
end

squeeze_data(func, func_data; kwargs...) = func_data
squeeze_data(wta::WindowedTimeAverage{<:AbstractField}, data; kwargs...) = squeeze_data(wta.operand, data; kwargs...)
squeeze_data(fd::AbstractField; kwargs...) = squeeze_data(fd, parent(fd); kwargs...)
squeeze_data(fd::WindowedTimeAverage{<:AbstractField}; kwargs...) = squeeze_data(fd.operand; kwargs...)

"""
    effective_reduced_dimensions(field)

Return dimensions that are effectively reduced, considering both location-based reduction
(e.g. a `Nothing` location) and grid topology (i.e. a `Flat` topology is considered a reduction).
"""
function effective_reduced_dimensions(field)
    loc_reduced = reduced_dimensions(field)

    topo_reduced = []
    for (dim, topo) in enumerate(topology(field))
        if topo == Flat
            push!(topo_reduced, dim)
        end
    end

    all_reduced = (loc_reduced..., topo_reduced...)
    return Tuple(unique(all_reduced))
end

dictify(outputs) = outputs
dictify(outputs::NamedTuple) = Dict(string(k) => dictify(v) for (k, v) in zip(keys(outputs), values(outputs)))

# We collect to ensure we return an array which NCDatasets.jl needs
# instead of a range or offset array.
function collect_dim(ξ, ℓ, T, N, H, inds, with_halos)
    if with_halos
        return collect(ξ)
    else
        inds = validate_index(inds, ℓ, T, N, H)
        inds = restrict_to_interior(inds, ℓ, T, N)
        return collect(ξ[inds])
    end
end

function create_time_dimension!(dataset; attrib=nothing, dimension_type=Float64)
    if "time" ∉ keys(dataset.dim)
        # Create an unlimited dimension "time"
        defDim(dataset, "time", Inf)
        defVar(dataset, "time", dimension_type, ("time",), attrib=attrib)
    end
end

#####
##### Conversion utilities
#####

# Using OrderedDict to preserve order of keys (important when saving positional arguments), and string(key) because that's what NetCDF supports as global_attributes.
convert_for_netcdf(dict::AbstractDict) = OrderedDict(string(key) => convert_for_netcdf(value) for (key, value) in dict)
convert_for_netcdf(x::Number) = x
convert_for_netcdf(x::Bool) = string(x)
convert_for_netcdf(x::NTuple{N, Number}) where N = collect(x)
convert_for_netcdf(x) = string(x)
convert_for_netcdf(::GPU) = "GPU()"
convert_for_netcdf(::CenterImmersedCondition) = "CenterImmersedCondition()"
convert_for_netcdf(::InterfaceImmersedCondition) = "InterfaceImmersedCondition()"

materialize_from_netcdf(dict::AbstractDict) = OrderedDict(Symbol(key) => materialize_from_netcdf(value) for (key, value) in dict)
materialize_from_netcdf(x::Number) = x
materialize_from_netcdf(x::Array) = Tuple(x)
materialize_from_netcdf(x::String) = @eval $(Meta.parse(x))

#####
##### Extension utilities
#####

ext(::Type{NetCDFWriter}) = ".nc"
