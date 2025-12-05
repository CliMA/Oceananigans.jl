module OceananigansNCDatasetsExt

using NCDatasets

using Dates: AbstractTime, UTC, now, DateTime
using Printf: @sprintf
using OrderedCollections: OrderedDict
using SeawaterPolynomials: BoussinesqEquationOfState

using Oceananigans: initialize!, prettytime, pretty_filesize, AbstractModel
using Oceananigans.Architectures: CPU, GPU, on_architecture
using Oceananigans.AbstractOperations: KernelFunctionOperation, AbstractOperation
using Oceananigans.BuoyancyFormulations: BuoyancyForce, BuoyancyTracer, SeawaterBuoyancy, LinearEquationOfState
using Oceananigans.Fields
using Oceananigans.Fields: Reduction, reduced_dimensions, reduced_location, location, indices
using Oceananigans.Grids: Center, Face, Flat, Periodic, Bounded,
                          AbstractGrid, RectilinearGrid, LatitudeLongitudeGrid, StaticVerticalDiscretization,
                          topology, halo_size, xspacings, yspacings, zspacings, λspacings, φspacings,
                          parent_index_range, nodes, ξnodes, ηnodes, rnodes, validate_index, peripheral_node,
                          constructor_arguments, architecture
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBottom, GFBIBG, GridFittedBoundary, PartialCellBottom, PCBIBG,
                                       CenterImmersedCondition, InterfaceImmersedCondition
using Oceananigans.Models: ShallowWaterModel, LagrangianParticles
using Oceananigans.Utils: TimeInterval, IterationInterval, WallTimeInterval, materialize_schedule,
                          versioninfo_with_gpu, oceananigans_versioninfo, prettykeys
using Oceananigans.OutputWriters:
    auto_extension,
    output_averaging_schedule,
    show_averaging_schedule,
    AveragedTimeInterval,
    WindowedTimeAverage,
    NoFileSplitting,
    update_file_splitting_schedule!,
    construct_output,
    time_average_outputs,
    restrict_to_interior,
    fetch_output,
    convert_output,
    fetch_and_convert_output,
    show_array_type
using NCDatasets: AbstractDataset

import NCDatasets: defVar
import Oceananigans: write_output!
import Oceananigans.OutputWriters:
    NetCDFWriter,
    write_grid_reconstruction_data!,
    convert_for_netcdf,
    materialize_from_netcdf,
    reconstruct_grid,
    trilocation_dim_name,
    dimension_name_generator_free_surface

const c = Center()
const f = Face()
const BoussinesqSeawaterBuoyancy = SeawaterBuoyancy{FT, <:BoussinesqEquationOfState, T, S} where {FT, T, S}
const BuoyancyBoussinesqEOSModel = BuoyancyForce{<:BoussinesqSeawaterBuoyancy, g} where {g}

#####
##### Extend defVar to be able to write fields to NetCDF directly
#####

"""
    squeeze_data(fd::AbstractField; array_type=Array{eltype(fd)})

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

defVar(ds::AbstractDataset, name, op::AbstractOperation; kwargs...) = defVar(ds, name, Field(op); kwargs...)
defVar(ds::AbstractDataset, name, op::Reduction; kwargs...) = defVar(ds, name, Field(op); kwargs...)

function defVar(ds::AbstractDataset, field_name, fd::AbstractField;
                array_type=Array{eltype(fd)},
                time_dependent=false,
                with_halos=false,
                dimension_name_generator = trilocation_dim_name,
                dimension_type=Float64,
                write_data=true,
                kwargs...)

    # effective_dim_names are the dimensions that will be used to write the field data (excludes reduced and dimensions where location is Nothing)
    effective_dim_names = create_field_dimensions!(ds, fd, dimension_name_generator; time_dependent, with_halos, array_type, dimension_type)

    # Write the data to the NetCDF file (or don't, but still create the space for it there)
    if write_data
        # Squeeze the data to remove dimensions where location is Nothing and add a time dimension if the field is time-dependent
        constructed_fd = construct_output(fd, fd.grid, (:, :, :), with_halos)
        squeezed_field_data = squeeze_data(constructed_fd; array_type)
        squeezed_reshaped_field_data = time_dependent ? reshape(squeezed_field_data, size(squeezed_field_data)..., 1) : squeezed_field_data

        defVar(ds, field_name, squeezed_reshaped_field_data, effective_dim_names; kwargs...)
    else
        defVar(ds, field_name, eltype(array_type), effective_dim_names; kwargs...)
    end
end

defVar(ds::AbstractDataset, field_name::Union{AbstractString, Symbol}, data::Array{Bool}, dim_names; kwargs...) = defVar(ds, field_name, Int8.(data), dim_names; kwargs...)

#####
##### Dimension validation
#####

"""
    create_field_dimensions!(ds, fd::AbstractField, dimension_name_generator; time_dependent=false, with_halos=false, array_type=Array{eltype(fd)})

Creates all dimensions for the given field `fd` in the NetCDF dataset `ds`. If the dimensions
already exist, they are validated to match the expected dimensions.

Arguments:
- `ds`: NetCDF dataset
- `fd`: AbstractField being written
- `dim_names`: Tuple of dimension names to create/validate
- `dimension_name_generator`: Function to generate dimension names
"""
function create_field_dimensions!(ds, fd::AbstractField, dimension_name_generator; time_dependent=false, with_halos=false, array_type=Array{eltype(fd)}, dimension_type=Float64)
    # Assess and create the dimensions for the field

    dimension_attributes = default_dimension_attributes(fd.grid, dimension_name_generator)
    spatial_dim_names = field_dimensions(fd, dimension_name_generator)
    spatial_dim_data = nodes(fd; with_halos)

    # Create dictionary of spatial dimensions and their data. Using OrderedDict to ensure the order of the dimensions is preserved.
    spatial_dim_names_dict = OrderedDict(spatial_dim_name => spatial_dim_array for (spatial_dim_name, spatial_dim_array) in zip(spatial_dim_names, spatial_dim_data))
    effective_spatial_dim_names = create_spatial_dimensions!(ds, spatial_dim_names_dict, dimension_attributes; dimension_type)

    # Create time dimension if needed
    if time_dependent
        "time" ∉ keys(ds.dim) && create_time_dimension!(ds, dimension_type=dimension_type)
        return (effective_spatial_dim_names..., "time") # Add "time" dimension if the field is time-dependent
    else
        return effective_spatial_dim_names
    end
end

#####
##### Utils
#####

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


"""
    create_spatial_dimensions!(dataset, dims, attributes_dict; array_type=Array{Float32}, kwargs...)

Create spatial dimensions in the NetCDF dataset and define corresponding variables to store
their coordinate values. Each dimension variable has itself as its sole dimension (e.g., the
`x` variable has dimension `x`). The dimensions are created if they don't exist, and validated
against provided arrays if they do exist. An error is thrown if the dimension already exists
but is different from the provided array.
"""
function create_spatial_dimensions!(dataset, dims, attributes_dict; dimension_type=Float64, kwargs...)
    effective_dim_names = []
    for (i, (dim_name, dim_array)) in enumerate(dims)
        dim_array isa Nothing && continue # Don't create anything if dim_array is Nothing
        dim_name == "" && continue # Don't create anything if dim_name is an empty string
        push!(effective_dim_names, dim_name)

        dim_array = dimension_type.(dim_array) # Transform dim_array to the correct float type
        if dim_name ∉ keys(dataset.dim)
            # Create missing dimension
            defVar(dataset, dim_name, dim_array, (dim_name,), attrib=attributes_dict[dim_name]; kwargs...)
        else
            # Validate existing dimension
            dataset_dim_array = collect(dataset[dim_name])
            if dataset_dim_array != collect(dim_array)
                throw(ArgumentError("Dimension '$dim_name' already exists in dataset but is different from expected.\n" *
                                    "  Actual:   $(dataset_dim_array) (length=$(length(dataset_dim_array)))\n" *
                                    "  Expected: $(dim_array) (length=$(length(dim_array)))"))
            end
        end
    end
    return tuple(effective_dim_names...)
end

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

#####
##### Gathering of grid dimensions
#####

function maybe_add_particle_dims!(dims, outputs)
    if "particles" in keys(outputs)  # TODO: Change this to look for ::LagrangianParticles in outputs?
        dims["particle_id"] = collect(1:length(outputs["particles"]))
    end
    return dims
end

function gather_vertical_dimensions(coordinate::StaticVerticalDiscretization, TZ, Nz, Hz, z_indices, with_halos, dim_name_generator)
    zᵃᵃᶠ_name = dim_name_generator("z", coordinate, nothing, nothing, f, Val(:z))
    zᵃᵃᶜ_name = dim_name_generator("z", coordinate, nothing, nothing, c, Val(:z))

    zᵃᵃᶠ_data = collect_dim(coordinate.cᵃᵃᶠ, f, TZ(), Nz, Hz, z_indices, with_halos)
    zᵃᵃᶜ_data = collect_dim(coordinate.cᵃᵃᶜ, c, TZ(), Nz, Hz, z_indices, with_halos)

    return Dict(zᵃᵃᶠ_name => zᵃᵃᶠ_data,
                zᵃᵃᶜ_name => zᵃᵃᶜ_data)
end

function gather_dimensions(outputs, grid::RectilinearGrid, indices, with_halos, dim_name_generator)
    TX, TY, TZ = topology(grid)
    Nx, Ny, Nz = size(grid)
    Hx, Hy, Hz = halo_size(grid)

    dims = Dict()

    if TX != Flat
        xᶠᵃᵃ_name = dim_name_generator("x", grid, f, nothing, nothing, Val(:x))
        xᶜᵃᵃ_name = dim_name_generator("x", grid, c, nothing, nothing, Val(:x))

        xᶠᵃᵃ_data = collect_dim(grid.xᶠᵃᵃ, f, TX(), Nx, Hx, indices[1], with_halos)
        xᶜᵃᵃ_data = collect_dim(grid.xᶜᵃᵃ, c, TX(), Nx, Hx, indices[1], with_halos)

        dims[xᶠᵃᵃ_name] = xᶠᵃᵃ_data
        dims[xᶜᵃᵃ_name] = xᶜᵃᵃ_data
    end

    if TY != Flat
        yᵃᶠᵃ_name = dim_name_generator("y", grid, nothing, f, nothing, Val(:y))
        yᵃᶜᵃ_name = dim_name_generator("y", grid, nothing, c, nothing, Val(:y))

        yᵃᶠᵃ_data = collect_dim(grid.yᵃᶠᵃ, f, TY(), Ny, Hy, indices[2], with_halos)
        yᵃᶜᵃ_data = collect_dim(grid.yᵃᶜᵃ, c, TY(), Ny, Hy, indices[2], with_halos)

        dims[yᵃᶠᵃ_name] = yᵃᶠᵃ_data
        dims[yᵃᶜᵃ_name] = yᵃᶜᵃ_data
    end

    if TZ != Flat
        vertical_dims = gather_vertical_dimensions(grid.z, TZ, Nz, Hz, indices[3], with_halos, dim_name_generator)
        dims = merge(dims, vertical_dims)
    end

    maybe_add_particle_dims!(dims, outputs)

    return dims
end

function gather_dimensions(outputs, grid::LatitudeLongitudeGrid, indices, with_halos, dim_name_generator)
    TΛ, TΦ, TZ = topology(grid)
    Nλ, Nφ, Nz = size(grid)
    Hλ, Hφ, Hz = halo_size(grid)

    dims = Dict()

    if TΛ != Flat
        λᶠᵃᵃ_name = dim_name_generator("λ", grid, f, nothing, nothing, Val(:x))
        λᶜᵃᵃ_name = dim_name_generator("λ", grid, c, nothing, nothing, Val(:x))

        λᶠᵃᵃ_data = collect_dim(grid.λᶠᵃᵃ, f, TΛ(), Nλ, Hλ, indices[1], with_halos)
        λᶜᵃᵃ_data = collect_dim(grid.λᶜᵃᵃ, c, TΛ(), Nλ, Hλ, indices[1], with_halos)

        dims[λᶠᵃᵃ_name] = λᶠᵃᵃ_data
        dims[λᶜᵃᵃ_name] = λᶜᵃᵃ_data
    end

    if TΦ != Flat
        φᵃᶠᵃ_name = dim_name_generator("φ", grid, nothing, f, nothing, Val(:y))
        φᵃᶜᵃ_name = dim_name_generator("φ", grid, nothing, c, nothing, Val(:y))

        φᵃᶠᵃ_data = collect_dim(grid.φᵃᶠᵃ, f, TΦ(), Nφ, Hφ, indices[2], with_halos)
        φᵃᶜᵃ_data = collect_dim(grid.φᵃᶜᵃ, c, TΦ(), Nφ, Hφ, indices[2], with_halos)

        dims[φᵃᶠᵃ_name] = φᵃᶠᵃ_data
        dims[φᵃᶜᵃ_name] = φᵃᶜᵃ_data
    end

    if TZ != Flat
        vertical_dims = gather_vertical_dimensions(grid.z, TZ, Nz, Hz, indices[3], with_halos, dim_name_generator)
        dims = merge(dims, vertical_dims)
    end

    maybe_add_particle_dims!(dims, outputs)

    return dims
end

gather_dimensions(outputs, grid::ImmersedBoundaryGrid, args...) =
    gather_dimensions(outputs, grid.underlying_grid, args...)

#####
##### Gathering of grid metrics
#####

function gather_grid_metrics(grid::RectilinearGrid, indices, dim_name_generator)
    TX, TY, TZ = topology(grid)

    metrics = Dict()

    if TX != Flat
        Δxᶠᵃᵃ_name = dim_name_generator("Δx", grid, f, nothing, nothing, Val(:x))
        Δxᶜᵃᵃ_name = dim_name_generator("Δx", grid, c, nothing, nothing, Val(:x))

        Δxᶠᵃᵃ_field = Field(xspacings(grid, f); indices)
        Δxᶜᵃᵃ_field = Field(xspacings(grid, c); indices)

        metrics[Δxᶠᵃᵃ_name] = Δxᶠᵃᵃ_field
        metrics[Δxᶜᵃᵃ_name] = Δxᶜᵃᵃ_field
    end

    if TY != Flat
        Δyᵃᶠᵃ_name = dim_name_generator("Δy", grid, nothing, f, nothing, Val(:y))
        Δyᵃᶜᵃ_name = dim_name_generator("Δy", grid, nothing, c, nothing, Val(:y))

        Δyᵃᶠᵃ_field = Field(yspacings(grid, f); indices)
        Δyᵃᶜᵃ_field = Field(yspacings(grid, c); indices)

        metrics[Δyᵃᶠᵃ_name] = Δyᵃᶠᵃ_field
        metrics[Δyᵃᶜᵃ_name] = Δyᵃᶜᵃ_field
    end

    if TZ != Flat
        Δzᵃᵃᶠ_name = dim_name_generator("Δz", grid, nothing, nothing, f, Val(:z))
        Δzᵃᵃᶜ_name = dim_name_generator("Δz", grid, nothing, nothing, c, Val(:z))

        Δzᵃᵃᶠ_field = Field(zspacings(grid, f); indices)
        Δzᵃᵃᶜ_field = Field(zspacings(grid, c); indices)

        metrics[Δzᵃᵃᶠ_name] = Δzᵃᵃᶠ_field
        metrics[Δzᵃᵃᶜ_name] = Δzᵃᵃᶜ_field
    end

    return metrics
end

function gather_grid_metrics(grid::LatitudeLongitudeGrid, indices, dim_name_generator)
    TΛ, TΦ, TZ = topology(grid)

    metrics = Dict()

    if TΛ != Flat
        Δλᶠᵃᵃ_name = dim_name_generator("Δλ", grid, f, nothing, nothing, Val(:x))
        Δλᶜᵃᵃ_name = dim_name_generator("Δλ", grid, c, nothing, nothing, Val(:x))

        Δλᶠᵃᵃ_field = Field(λspacings(grid, f); indices)
        Δλᶜᵃᵃ_field = Field(λspacings(grid, c); indices)

        metrics[Δλᶠᵃᵃ_name] = Δλᶠᵃᵃ_field
        metrics[Δλᶜᵃᵃ_name] = Δλᶜᵃᵃ_field

        Δxᶠᶠᵃ_name = dim_name_generator("Δx", grid, f, f, nothing, Val(:x))
        Δxᶠᶜᵃ_name = dim_name_generator("Δx", grid, f, c, nothing, Val(:x))
        Δxᶜᶠᵃ_name = dim_name_generator("Δx", grid, c, f, nothing, Val(:x))
        Δxᶜᶜᵃ_name = dim_name_generator("Δx", grid, c, c, nothing, Val(:x))

        Δxᶠᶠᵃ_field = Field(xspacings(grid, f, f); indices)
        Δxᶠᶜᵃ_field = Field(xspacings(grid, f, c); indices)
        Δxᶜᶠᵃ_field = Field(xspacings(grid, c, f); indices)
        Δxᶜᶜᵃ_field = Field(xspacings(grid, c, c); indices)

        metrics[Δxᶠᶠᵃ_name] = Δxᶠᶠᵃ_field
        metrics[Δxᶠᶜᵃ_name] = Δxᶠᶜᵃ_field
        metrics[Δxᶜᶠᵃ_name] = Δxᶜᶠᵃ_field
        metrics[Δxᶜᶜᵃ_name] = Δxᶜᶜᵃ_field
    end

    if TΦ != Flat
        Δφᵃᶠᵃ_name = dim_name_generator("Δλ", grid, nothing, f, nothing, Val(:y))
        Δφᵃᶜᵃ_name = dim_name_generator("Δλ", grid, nothing, c, nothing, Val(:y))

        Δφᵃᶠᵃ_field = Field(φspacings(grid, f); indices)
        Δφᵃᶜᵃ_field = Field(φspacings(grid, c); indices)

        metrics[Δφᵃᶠᵃ_name] = Δφᵃᶠᵃ_field
        metrics[Δφᵃᶜᵃ_name] = Δφᵃᶜᵃ_field

        Δyᶠᶠᵃ_name = dim_name_generator("Δy", grid, f, f, nothing, Val(:y))
        Δyᶠᶜᵃ_name = dim_name_generator("Δy", grid, f, c, nothing, Val(:y))
        Δyᶜᶠᵃ_name = dim_name_generator("Δy", grid, c, f, nothing, Val(:y))
        Δyᶜᶜᵃ_name = dim_name_generator("Δy", grid, c, c, nothing, Val(:y))

        Δyᶠᶠᵃ_field = Field(yspacings(grid, f, f); indices)
        Δyᶠᶜᵃ_field = Field(yspacings(grid, f, c); indices)
        Δyᶜᶠᵃ_field = Field(yspacings(grid, c, f); indices)
        Δyᶜᶜᵃ_field = Field(yspacings(grid, c, c); indices)

        metrics[Δyᶠᶠᵃ_name] = Δyᶠᶠᵃ_field
        metrics[Δyᶠᶜᵃ_name] = Δyᶠᶜᵃ_field
        metrics[Δyᶜᶠᵃ_name] = Δyᶜᶠᵃ_field
        metrics[Δyᶜᶜᵃ_name] = Δyᶜᶜᵃ_field
    end

    if TZ != Flat
        Δzᵃᵃᶠ_name = dim_name_generator("Δz", grid, nothing, nothing, f, Val(:z))
        Δzᵃᵃᶜ_name = dim_name_generator("Δz", grid, nothing, nothing, c, Val(:z))

        Δzᵃᵃᶠ_field = Field(zspacings(grid, f); indices)
        Δzᵃᵃᶜ_field = Field(zspacings(grid, c); indices)

        metrics[Δzᵃᵃᶠ_name] = Δzᵃᵃᶠ_field
        metrics[Δzᵃᵃᶜ_name] = Δzᵃᵃᶜ_field
    end

    return metrics
end

gather_grid_metrics(grid::ImmersedBoundaryGrid, args...) =
    gather_grid_metrics(grid.underlying_grid, args...)

#####
##### Gathering of immersed boundary fields
#####

# TODO: Proper masks for 2D models?
flat_loc(T, L) = T == Flat ? nothing : L

const PCBorGFBIBG = Union{GFBIBG, PCBIBG}

# For Immersed Boundary Grids (IBG) with either a Grid Fitted Bottom (GFB) or a Partial Cell Bottom (PCB)
function gather_immersed_boundary(grid::PCBorGFBIBG, indices, dim_name_generator)
    op_peripheral_nodes_ccc = KernelFunctionOperation{Center, Center, Center}(peripheral_node, grid, Center(), Center(), Center())
    op_peripheral_nodes_fcc = KernelFunctionOperation{Face, Center, Center}(peripheral_node, grid, Face(), Center(), Center())
    op_peripheral_nodes_cfc = KernelFunctionOperation{Center, Face, Center}(peripheral_node, grid, Center(), Face(), Center())
    op_peripheral_nodes_ccf = KernelFunctionOperation{Center, Center, Face}(peripheral_node, grid, Center(), Center(), Face())

    return Dict("bottom_height" => Field(grid.immersed_boundary.bottom_height; indices),
                "peripheral_nodes_ccc" => Field(op_peripheral_nodes_ccc; indices),
                "peripheral_nodes_fcc" => Field(op_peripheral_nodes_fcc; indices),
                "peripheral_nodes_cfc" => Field(op_peripheral_nodes_cfc; indices),
                "peripheral_nodes_ccf" => Field(op_peripheral_nodes_ccf; indices))
end

const GFBoundaryIBG = ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:GridFittedBoundary}

# For Immersed Boundary Grids (IBG) with a Grid Fitted Boundary (also GFB!)
function gather_immersed_boundary(grid::GFBoundaryIBG, indices, dim_name_generator)
    op_peripheral_nodes_ccc = KernelFunctionOperation{Center, Center, Center}(peripheral_node, grid, Center(), Center(), Center())
    op_peripheral_nodes_fcc = KernelFunctionOperation{Face, Center, Center}(peripheral_node, grid, Face(), Center(), Center())
    op_peripheral_nodes_cfc = KernelFunctionOperation{Center, Face, Center}(peripheral_node, grid, Center(), Face(), Center())
    op_peripheral_nodes_ccf = KernelFunctionOperation{Center, Center, Face}(peripheral_node, grid, Center(), Center(), Face())

    return Dict("peripheral_nodes" => Field(grid.immersed_boundary.mask; indices),
                "peripheral_nodes_ccc" => Field(op_peripheral_nodes_ccc; indices),
                "peripheral_nodes_fcc" => Field(op_peripheral_nodes_fcc; indices),
                "peripheral_nodes_cfc" => Field(op_peripheral_nodes_cfc; indices),
                "peripheral_nodes_ccf" => Field(op_peripheral_nodes_ccf; indices))
end

#####
##### Mapping outputs/fields to dimensions
#####

function field_dimensions(fd::AbstractField, grid::RectilinearGrid, dim_name_generator)
    LX, LY, LZ = location(fd)
    TX, TY, TZ = topology(grid)

    x_dim_name = LX == Nothing ? "" : dim_name_generator("x", grid, LX(), nothing, nothing, Val(:x))
    y_dim_name = LY == Nothing ? "" : dim_name_generator("y", grid, nothing, LY(), nothing, Val(:y))
    z_dim_name = LZ == Nothing ? "" : dim_name_generator("z", grid, nothing, nothing, LZ(), Val(:z))

    return tuple(x_dim_name, y_dim_name, z_dim_name)
end

function field_dimensions(fd::AbstractField, grid::LatitudeLongitudeGrid, dim_name_generator)
    LΛ, LΦ, LZ = location(fd)
    TΛ, TΦ, TZ = topology(grid)

    λ_dim_name = LΛ == Nothing ? "" : dim_name_generator("λ", grid, LΛ(), nothing, nothing, Val(:x))
    φ_dim_name = LΦ == Nothing ? "" : dim_name_generator("φ", grid, nothing, LΦ(), nothing, Val(:y))
    z_dim_name = LZ == Nothing ? "" : dim_name_generator("z", grid, nothing, nothing, LZ(), Val(:z))

    return tuple(λ_dim_name, φ_dim_name, z_dim_name)
end

field_dimensions(fd::AbstractField, grid::ImmersedBoundaryGrid, dim_name_generator) =
    field_dimensions(fd, grid.underlying_grid, dim_name_generator)

field_dimensions(fd::AbstractField, dim_name_generator) =
    field_dimensions(fd, fd.grid, dim_name_generator)

#####
##### Dimension attributes
#####

const base_dimension_attributes = Dict("time"        => Dict("long_name" => "Time", "units" => "s"),
                                       "particle_id" => Dict("long_name" => "Particle ID"))

function default_vertical_dimension_attributes(coordinate::StaticVerticalDiscretization, dim_name_generator)
    zᵃᵃᶠ_name = dim_name_generator("z", coordinate, nothing, nothing, f, Val(:z))
    zᵃᵃᶜ_name = dim_name_generator("z", coordinate, nothing, nothing, c, Val(:z))

    Δzᵃᵃᶠ_name = dim_name_generator("Δz", coordinate, nothing, nothing, f, Val(:z))
    Δzᵃᵃᶜ_name = dim_name_generator("Δz", coordinate, nothing, nothing, c, Val(:z))

    zᵃᵃᶠ_attrs = Dict("long_name" => "Cell face locations in the z-direction.",   "units" => "m")
    zᵃᵃᶜ_attrs = Dict("long_name" => "Cell center locations in the z-direction.", "units" => "m")

    Δzᵃᵃᶠ_attrs = Dict("long_name" => "Spacings between cell centers (located at cell faces) in the z-direction.", "units" => "m")
    Δzᵃᵃᶜ_attrs = Dict("long_name" => "Spacings between cell faces (located at cell centers) in the z-direction.", "units" => "m")

    return Dict(zᵃᵃᶠ_name => zᵃᵃᶠ_attrs,
                zᵃᵃᶜ_name => zᵃᵃᶜ_attrs,
                Δzᵃᵃᶠ_name => Δzᵃᵃᶠ_attrs,
                Δzᵃᵃᶜ_name => Δzᵃᵃᶜ_attrs)
end

function default_dimension_attributes(grid::RectilinearGrid, dim_name_generator)
    xᶠᵃᵃ_name = dim_name_generator("x", grid, f, nothing, nothing, Val(:x))
    xᶜᵃᵃ_name = dim_name_generator("x", grid, c, nothing, nothing, Val(:x))
    yᵃᶠᵃ_name = dim_name_generator("y", grid, nothing, f, nothing, Val(:y))
    yᵃᶜᵃ_name = dim_name_generator("y", grid, nothing, c, nothing, Val(:y))

    Δxᶠᵃᵃ_name = dim_name_generator("Δx", grid, f, nothing, nothing, Val(:x))
    Δxᶜᵃᵃ_name = dim_name_generator("Δx", grid, c, nothing, nothing, Val(:x))
    Δyᵃᶠᵃ_name = dim_name_generator("Δy", grid, nothing, f, nothing, Val(:y))
    Δyᵃᶜᵃ_name = dim_name_generator("Δy", grid, nothing, c, nothing, Val(:y))

    xᶠᵃᵃ_attrs = Dict("long_name" => "Cell face locations in the x-direction.",   "units" => "m")
    xᶜᵃᵃ_attrs = Dict("long_name" => "Cell center locations in the x-direction.", "units" => "m")
    yᵃᶠᵃ_attrs = Dict("long_name" => "Cell face locations in the y-direction.",   "units" => "m")
    yᵃᶜᵃ_attrs = Dict("long_name" => "Cell center locations in the y-direction.", "units" => "m")

    Δxᶠᵃᵃ_attrs = Dict("long_name" => "Spacings between cell centers (located at the cell faces) in the x-direction.", "units" => "m")
    Δxᶜᵃᵃ_attrs = Dict("long_name" => "Spacings between cell faces (located at the cell centers) in the x-direction.", "units" => "m")
    Δyᵃᶠᵃ_attrs = Dict("long_name" => "Spacings between cell centers (located at cell faces) in the y-direction.", "units" => "m")
    Δyᵃᶜᵃ_attrs = Dict("long_name" => "Spacings between cell faces (located at cell centers) in the y-direction.", "units" => "m")

    horizontal_dimension_attributes = Dict(xᶠᵃᵃ_name  => xᶠᵃᵃ_attrs,
                                           xᶜᵃᵃ_name  => xᶜᵃᵃ_attrs,
                                           yᵃᶠᵃ_name  => yᵃᶠᵃ_attrs,
                                           yᵃᶜᵃ_name  => yᵃᶜᵃ_attrs,
                                           Δxᶠᵃᵃ_name => Δxᶠᵃᵃ_attrs,
                                           Δxᶜᵃᵃ_name => Δxᶜᵃᵃ_attrs,
                                           Δyᵃᶠᵃ_name => Δyᵃᶠᵃ_attrs,
                                           Δyᵃᶜᵃ_name => Δyᵃᶜᵃ_attrs)

    vertical_dimension_attributes = default_vertical_dimension_attributes(grid.z, dim_name_generator)

    return merge(base_dimension_attributes,
                 horizontal_dimension_attributes,
                 vertical_dimension_attributes)
end

function default_dimension_attributes(grid::LatitudeLongitudeGrid, dim_name_generator)
    λᶠᵃᵃ_name = dim_name_generator("λ", grid, f, nothing, nothing, Val(:x))
    λᶜᵃᵃ_name = dim_name_generator("λ", grid, c, nothing, nothing, Val(:x))

    λᶠᵃᵃ_attrs = Dict("long_name" => "Cell face locations in the zonal direction.",   "units" => "degrees east")
    λᶜᵃᵃ_attrs = Dict("long_name" => "Cell center locations in the zonal direction.", "units" => "degrees east")

    φᵃᶠᵃ_name = dim_name_generator("φ", grid, nothing, f, nothing, Val(:y))
    φᵃᶜᵃ_name = dim_name_generator("φ", grid, nothing, c, nothing, Val(:y))

    φᵃᶠᵃ_attrs = Dict("long_name" => "Cell face locations in the meridional direction.",   "units" => "degrees north")
    φᵃᶜᵃ_attrs = Dict("long_name" => "Cell center locations in the meridional direction.", "units" => "degrees north")

    Δλᶠᵃᵃ_name = dim_name_generator("Δλ", grid, f, nothing, nothing, Val(:x))
    Δλᶜᵃᵃ_name = dim_name_generator("Δλ", grid, c, nothing, nothing, Val(:x))

    Δλᶠᵃᵃ_attrs = Dict("long_name" => "Angular spacings between cell faces in the zonal direction.",   "units" => "degrees")
    Δλᶜᵃᵃ_attrs = Dict("long_name" => "Angular spacings between cell centers in the zonal direction.", "units" => "degrees")

    Δφᵃᶠᵃ_name = dim_name_generator("Δλ", grid, nothing, f, nothing, Val(:y))
    Δφᵃᶜᵃ_name = dim_name_generator("Δλ", grid, nothing, c, nothing, Val(:y))

    Δφᵃᶠᵃ_attrs = Dict("long_name" => "Angular spacings between cell faces in the meridional direction.",   "units" => "degrees")
    Δφᵃᶜᵃ_attrs = Dict("long_name" => "Angular spacings between cell centers in the meridional direction.", "units" => "degrees")

    Δxᶠᶠᵃ_name = dim_name_generator("Δx", grid, f, f, nothing, Val(:x))
    Δxᶠᶜᵃ_name = dim_name_generator("Δx", grid, f, c, nothing, Val(:x))
    Δxᶜᶠᵃ_name = dim_name_generator("Δx", grid, c, f, nothing, Val(:x))
    Δxᶜᶜᵃ_name = dim_name_generator("Δx", grid, c, c, nothing, Val(:x))

    Δxᶠᶠᵃ_attrs = Dict("long_name" => "Geodesic spacings in the zonal direction between the cell located at (Face, Face).",
                       "units" => "m")

    Δxᶠᶜᵃ_attrs = Dict("long_name" => "Geodesic spacings in the zonal direction between the cell located at (Face, Center).",
                       "units" => "m")

    Δxᶜᶠᵃ_attrs = Dict("long_name" => "Geodesic spacings in the zonal direction between the cell located at (Center, Face).",
                       "units" => "m")

    Δxᶜᶜᵃ_attrs = Dict("long_name" => "Geodesic spacings in the zonal direction between the cell located at (Center, Center).",
                       "units" => "m")

    Δyᶠᶠᵃ_name = dim_name_generator("Δy", grid, f, f, nothing, Val(:y))
    Δyᶠᶜᵃ_name = dim_name_generator("Δy", grid, f, c, nothing, Val(:y))
    Δyᶜᶠᵃ_name = dim_name_generator("Δy", grid, c, f, nothing, Val(:y))
    Δyᶜᶜᵃ_name = dim_name_generator("Δy", grid, c, c, nothing, Val(:y))

    Δyᶠᶠᵃ_attrs = Dict("long_name" => "Geodesic spacings in the meridional direction between the cell located at (Face, Face).",
                       "units" => "m")

    Δyᶠᶜᵃ_attrs = Dict("long_name" => "Geodesic spacings in the meridional direction between the cell located at (Face, Center).",
                       "units" => "m")

    Δyᶜᶠᵃ_attrs = Dict("long_name" => "Geodesic spacings in the meridional direction between the cell located at (Center, Face).",
                       "units" => "m")

    Δyᶜᶜᵃ_attrs = Dict("long_name" => "Geodesic spacings in the meridional direction between the cell located at (Center, Center).",
                       "units" => "m")

    horizontal_dimension_attributes = Dict(λᶠᵃᵃ_name  => λᶠᵃᵃ_attrs,
                                           λᶜᵃᵃ_name  => λᶜᵃᵃ_attrs,
                                           φᵃᶠᵃ_name  => φᵃᶠᵃ_attrs,
                                           φᵃᶜᵃ_name  => φᵃᶜᵃ_attrs,
                                           Δλᶠᵃᵃ_name => Δλᶠᵃᵃ_attrs,
                                           Δλᶜᵃᵃ_name => Δλᶜᵃᵃ_attrs,
                                           Δφᵃᶠᵃ_name => Δφᵃᶠᵃ_attrs,
                                           Δφᵃᶜᵃ_name => Δφᵃᶜᵃ_attrs,
                                           Δxᶠᶠᵃ_name => Δxᶠᶠᵃ_attrs,
                                           Δxᶠᶜᵃ_name => Δxᶠᶜᵃ_attrs,
                                           Δxᶜᶠᵃ_name => Δxᶜᶠᵃ_attrs,
                                           Δxᶜᶜᵃ_name => Δxᶜᶜᵃ_attrs,
                                           Δyᶠᶠᵃ_name => Δyᶠᶠᵃ_attrs,
                                           Δyᶠᶜᵃ_name => Δyᶠᶜᵃ_attrs,
                                           Δyᶜᶠᵃ_name => Δyᶜᶠᵃ_attrs,
                                           Δyᶜᶜᵃ_name => Δyᶜᶜᵃ_attrs)

    vertical_dimension_attributes = default_vertical_dimension_attributes(grid.z, dim_name_generator)

    return merge(base_dimension_attributes,
                 horizontal_dimension_attributes,
                 vertical_dimension_attributes)
end

default_dimension_attributes(grid::ImmersedBoundaryGrid, dim_name_generator) =
    default_dimension_attributes(grid.underlying_grid, dim_name_generator)

#####
##### Variable attributes
#####

default_velocity_attributes(::RectilinearGrid) = Dict(
    "u" => Dict("long_name" => "Velocity in the +x-direction.", "units" => "m/s"),
    "v" => Dict("long_name" => "Velocity in the +y-direction.", "units" => "m/s"),
    "w" => Dict("long_name" => "Velocity in the +z-direction.", "units" => "m/s"))

default_velocity_attributes(::LatitudeLongitudeGrid) = Dict(
    "u" => Dict("long_name" => "Velocity in the zonal direction (+ = east).", "units" => "m/s"),
    "v" => Dict("long_name" => "Velocity in the meridional direction (+ = north).", "units" => "m/s"),
    "w" => Dict("long_name" => "Velocity in the vertical direction (+ = up).", "units" => "m/s"),
    "η" => Dict("long_name" => "Sea surface height", "units" => "m/s"),
    "eta" => Dict("long_name" => "Sea surface height", "units" => "m/s")) # non-unicode default

default_velocity_attributes(ibg::ImmersedBoundaryGrid) = default_velocity_attributes(ibg.underlying_grid)

default_tracer_attributes(::Nothing) = Dict()

default_tracer_attributes(::BuoyancyForce{<:BuoyancyTracer}) = Dict("b" => Dict("long_name" => "Buoyancy", "units" => "m/s²"))

default_tracer_attributes(::BuoyancyForce{<:SeawaterBuoyancy{FT, <:LinearEquationOfState}}) where FT = Dict(
    "T" => Dict("long_name" => "Temperature", "units" => "°C"),
    "S" => Dict("long_name" => "Salinity",    "units" => "practical salinity unit (psu)"))

default_tracer_attributes(::BuoyancyBoussinesqEOSModel) = Dict("T" => Dict("long_name" => "Conservative temperature", "units" => "°C"),
                                                               "S" => Dict("long_name" => "Absolute salinity",        "units" => "g/kg"))

function default_output_attributes(model)
    velocity_attrs = default_velocity_attributes(model.grid)
    buoyancy = model isa ShallowWaterModel ? nothing : model.buoyancy
    tracer_attrs = default_tracer_attributes(buoyancy)
    return merge(velocity_attrs, tracer_attrs)
end

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

function netcdf_grid_constructor_info(grid)
    underlying_grid_args, underlying_grid_kwargs = constructor_arguments(grid)

    immersed_grid_args = Dict()

    underlying_grid_type = typeof(grid).name.wrapper |> string # Save type of grid for reconstruction
    grid_metadata = Dict(:immersed_boundary_type => nothing,
                         :underlying_grid_type => underlying_grid_type)
    return underlying_grid_args, underlying_grid_kwargs, immersed_grid_args, grid_metadata
end

function netcdf_grid_constructor_info(grid::ImmersedBoundaryGrid)
    underlying_grid_args, underlying_grid_kwargs, immersed_grid_args = constructor_arguments(grid)

    immersed_boundary_type = typeof(grid.immersed_boundary).name.wrapper |> string # Save type of immersed boundary for reconstruction
    underlying_grid_type   = typeof(grid.underlying_grid).name.wrapper |> string # Save type of underlying grid for reconstruction

    grid_metadata = Dict(:immersed_boundary_type => immersed_boundary_type,
                         :underlying_grid_type => underlying_grid_type)
    return underlying_grid_args, underlying_grid_kwargs, immersed_grid_args, grid_metadata
end

function write_immersed_boundary_data!(ds, grid::ImmersedBoundaryGrid, immersed_grid_args)
    group_name = "immersed_grid_reconstruction_args"
    if (grid.immersed_boundary isa GridFittedBottom) || (grid.immersed_boundary isa PartialCellBottom)
        bottom_height = pop!(immersed_grid_args, :bottom_height)
        ibg_group = defGroup(ds, group_name; attrib=convert_for_netcdf(immersed_grid_args))
        defVar(ibg_group, "bottom_height", bottom_height)

    elseif grid.immersed_boundary isa GridFittedBoundary
        mask = pop!(immersed_grid_args, :mask)
        ibg_group = defGroup(ds, group_name; attrib=convert_for_netcdf(immersed_grid_args))
        defVar(ibg_group, "mask", mask)
    end

    return ds
end

write_immersed_boundary_data!(ds, grid, immersed_grid_args) = nothing

function write_grid_reconstruction_data!(ds, grid; array_type=Array{eltype(grid)}, deflatelevel=0)
    underlying_grid_args, underlying_grid_kwargs, immersed_grid_args, grid_metadata = netcdf_grid_constructor_info(grid)
    underlying_grid_args, underlying_grid_kwargs, grid_metadata = map(convert_for_netcdf, (underlying_grid_args, underlying_grid_kwargs, grid_metadata))

    defGroup(ds, "underlying_grid_reconstruction_args"; attrib = underlying_grid_args)
    defGroup(ds, "underlying_grid_reconstruction_kwargs"; attrib = underlying_grid_kwargs)
    defGroup(ds, "grid_reconstruction_metadata"; attrib = grid_metadata)

    write_immersed_boundary_data!(ds, grid, immersed_grid_args)

    return ds
end

function reconstruct_grid(filename::String)
    ds = NCDataset(filename, "r")
    grid = reconstruct_grid(ds)
    close(ds)
    return grid
end

function reconstruct_immersed_boundary(ds)
    ibg_group = ds.group["immersed_grid_reconstruction_args"]

    grid_reconstruction_metadata = ds.group["grid_reconstruction_metadata"].attrib |> materialize_from_netcdf
    immersed_boundary_type = grid_reconstruction_metadata[:immersed_boundary_type]
    if immersed_boundary_type == GridFittedBottom
        bottom_height = Array(ibg_group["bottom_height"])
        immersed_condition = ibg_group.attrib["immersed_condition"] |> materialize_from_netcdf
        immersed_boundary = immersed_boundary_type(bottom_height, immersed_condition)

    elseif immersed_boundary_type == PartialCellBottom
        bottom_height = Array(ibg_group["bottom_height"])
        minimum_fractional_cell_height = ibg_group.attrib["minimum_fractional_cell_height"] |> materialize_from_netcdf
        immersed_boundary = immersed_boundary_type(bottom_height, minimum_fractional_cell_height)

    elseif immersed_boundary_type == GridFittedBoundary
        mask = Array(ibg_group["mask"])
        immersed_boundary = immersed_boundary_type(mask)

    else
        error("Unsupported immersed boundary type: $immersed_boundary_type")
    end
    return immersed_boundary
end


function reconstruct_grid(ds)
    # Read back the grid reconstruction metadata
    underlying_grid_reconstruction_args   = ds.group["underlying_grid_reconstruction_args"].attrib |> materialize_from_netcdf
    underlying_grid_reconstruction_kwargs = ds.group["underlying_grid_reconstruction_kwargs"].attrib |> materialize_from_netcdf
    grid_reconstruction_metadata          = ds.group["grid_reconstruction_metadata"].attrib |> materialize_from_netcdf

    # Pop out infomration about the underlying grid
    underlying_grid_type = grid_reconstruction_metadata[:underlying_grid_type]
    underlying_grid = underlying_grid_type(values(underlying_grid_reconstruction_args)...; underlying_grid_reconstruction_kwargs...)

    # If this is an ImmersedBoundaryGrid, reconstruct the immersed boundary, otherwise underlying grid is the final grid
    if isnothing(grid_reconstruction_metadata[:immersed_boundary_type])
        grid = underlying_grid
    else
        immersed_boundary = reconstruct_immersed_boundary(ds)
        immersed_boundary = on_architecture(architecture(underlying_grid), immersed_boundary)
        grid = ImmersedBoundaryGrid(underlying_grid, immersed_boundary)
    end

    return grid
end

#####
##### Saving schedule metadata as global attributes
#####

add_schedule_metadata!(attributes, schedule) = nothing

function add_schedule_metadata!(global_attributes, schedule::IterationInterval)
    global_attributes["schedule"] = "IterationInterval"
    global_attributes["interval"] = schedule.interval
    global_attributes["output iteration interval"] = "Output was saved every $(schedule.interval) iteration(s)."

    return nothing
end

function add_schedule_metadata!(global_attributes, schedule::TimeInterval)
    global_attributes["schedule"] = "TimeInterval"
    global_attributes["interval"] = schedule.interval
    global_attributes["output time interval"] = "Output was saved every $(prettytime(schedule.interval))."

    return nothing
end

function add_schedule_metadata!(global_attributes, schedule::WallTimeInterval)
    global_attributes["schedule"] = "WallTimeInterval"
    global_attributes["interval"] = schedule.interval
    global_attributes["output time interval"] =
        "Output was saved every $(prettytime(schedule.interval))."

    return nothing
end

function add_schedule_metadata!(global_attributes, schedule::AveragedTimeInterval)
    global_attributes["schedule"] = "AveragedTimeInterval"
    global_attributes["interval"] = schedule.interval
    global_attributes["output time interval"] = "Output was time-averaged and saved every $(prettytime(schedule.interval))."

    global_attributes["time_averaging_window"] = schedule.window
    global_attributes["time averaging window"] = "Output was time averaged with a window size of $(prettytime(schedule.window))"

    global_attributes["time_averaging_stride"] = schedule.stride
    global_attributes["time averaging stride"] = "Output was time averaged with a stride of $(schedule.stride) iteration(s) within the time averaging window."

    return nothing
end

#####
##### NetCDFWriter constructor
#####

"""
    NetCDFWriter(model, outputs;
                 filename,
                 schedule,
                 grid = model.grid,
                 dir = ".",
                 array_type = Array{Float32},
                 indices = (:, :, :),
                 global_attributes = Dict(),
                 output_attributes = Dict(),
                 dimensions = Dict(),
                 with_halos = false,
                 include_grid_metrics = true,
                 overwrite_existing = nothing,
                 verbose = false,
                 deflatelevel = 0,
                 part = 1,
                 file_splitting = NoFileSplitting(),
                 dimension_name_generator = trilocation_dim_name)

Construct a `NetCDFWriter` that writes `(label, output)` pairs in `outputs` to a NetCDF file.
The `outputs` can be a `Dict` or `NamedTuple` where each `label` is a string and each `output` is
one of:

- A `Field` (e.g., `model.velocities.u`)
- A `Reduction` (e.g., `Average(model.tracers.T, dims=(1, 2))`)
- `LagrangianParticles` for particle tracking data
- A function `f(model)` that returns something to be written to disk

If any of `outputs` are not `AbstractField`, `Reduction`, or `LagrangianParticles`, their spatial
`dimensions` must be provided.

Required arguments
==================

- `model`: The Oceananigans model instance.

- `outputs`: A collection of outputs to write, specified as either:
  * A `Dict` with string keys and Field/function values.
  * A `NamedTuple` of `Field`s or functions.

Required keyword arguments
==========================

- `filename`: Descriptive filename. `".nc"` is appended if not present.

- `schedule`: An `AbstractSchedule` that determines when output is saved. Options include:
  * `TimeInterval(dt)`: Save every `dt` seconds of simulation time.
  * `IterationInterval(n)`: Save every `n` iterations.
  * `AveragedTimeInterval(dt; window, stride)`: Time-average output over a window before saving.
  * `WallTimeInterval(dt)`: Save every `dt` seconds of wall clock time.

Optional keyword arguments
==========================

- `grid`: The grid associated with `outputs`. Defaults to `model.grid`. To use `outputs` on a different
          grid than `model.grid`, provide the proper `grid` here.

- `dir`: Directory to save output to. Default: `"."`.

- `array_type`: Type to convert outputs to before saving. Default: `Array{Float32}`.

- `indices`: Tuple of indices of the output variables to include. Default is `(:, :, :)`, which
             includes the full fields. This allows saving specific slices of the domain.

- `global_attributes`: `Dict` or `NamedTuple` of global attributes or metadata to save with every file.
                       Default: `Dict()`. This is useful for saving information specific to the simulation.
                       Some useful global attributes are included by default but will be overwritten if
                       included in this `Dict`.

- `output_attributes`: `Dict` or `NamedTuple` of attributes to be saved with each field variable.
                       Default: `Dict()`. Reasonable defaults including descriptive names and units are
                       provided for velocities, buoyancy, temperature, and salinity. Attributes provided
                       here will overwrite the defaults.

- `dimensions`: A `Dict` or `NamedTuple` of dimension tuples to apply to outputs (required for function
                outputs that return custom data).

- `with_halos`: Boolean defining whether to include halos in the outputs. Default: `false`.
                Note that to postprocess saved output (e.g., compute derivatives, etc.),
                information about the boundary conditions is often crucial. In those cases,
                you might need to set `with_halos = true`. Cannot be used with custom `indices`.

- `include_grid_metrics`: Include grid metrics such as grid spacings, areas, and volumes as
                          additional variables. Default: `true`. Note that even with
                          `include_grid_metrics = false`, core grid coordinates are still saved.

- `overwrite_existing`: If `false`, `NetCDFWriter` will append to existing files. If `true`,
                        it will overwrite existing files or create new ones. Default: `true` if the
                        file does not exist, `false` if it does.

- `verbose`: Log variable compute times, file write times, and file sizes. Default: `false`.

- `deflatelevel`: Determines the NetCDF compression level of data (integer 0-9; 0 (default) means no compression
                  and 9 means maximum compression). See [NCDatasets.jl documentation](https://alexander-barth.github.io/NCDatasets.jl/stable/variables/#Creating-a-variable)
                  for more information.

- `part`: The starting part number used when file splitting. Default: `1`.

- `file_splitting`: Schedule for splitting the output file. The new files will be suffixed with
                    `_part1`, `_part2`, etc. Options include:
                    * `FileSizeLimit(sz)`: Split when file size exceeds `sz` (e.g., `200KiB`).
                    * `TimeInterval(interval)`: Split every `interval` of simulation time.
                    * `NoFileSplitting()` (default): Don't split files.

- `dimension_name_generator`: A function with signature `(var_name, grid, LX, LY, LZ, dim)` where `dim` is
                              either `Val(:x)`, `Val(:y)`, or `Val(:z)` that returns a string corresponding
                              to the name of the dimension `var_name` on `grid` with location `(LX, LY, LZ)`
                              along `dim`. This advanced option can be used to rename dimensions and variables
                              to satisfy certain naming conventions. Default: `trilocation_dim_name`.

Examples
========

Saving the ``u`` velocity field and temperature fields, the full 3D fields and surface 2D slices
to separate NetCDF files:

```@example netcdf1
using Oceananigans

grid = RectilinearGrid(size=(16, 16, 16), extent=(1, 1, 1))

model = NonhydrostaticModel(grid=grid, tracers=:c)

simulation = Simulation(model, Δt=12, stop_time=3600)

fields = Dict("u" => model.velocities.u, "c" => model.tracers.c)

simulation.output_writers[:field_writer] =
    NetCDFWriter(model, fields, filename="fields.nc", schedule=TimeInterval(60))
```

```@example netcdf1
simulation.output_writers[:surface_slice_writer] =
    NetCDFWriter(model, fields, filename="surface_xy_slice.nc",
                 schedule=TimeInterval(60), indices=(:, :, grid.Nz))
```

```@example netcdf1
simulation.output_writers[:averaged_profile_writer] =
    NetCDFWriter(model, fields,
                 filename = "averaged_z_profile.nc",
                 schedule = AveragedTimeInterval(60, window=20),
                 indices = (1, 1, :))
```

`NetCDFWriter` also accepts output functions that write scalars and arrays to disk,
provided that their `dimensions` are provided:

```@example
using Oceananigans

Nx, Ny, Nz = 16, 16, 16

grid = RectilinearGrid(size=(Nx, Ny, Nz), extent=(1, 2, 3))

model = NonhydrostaticModel(; grid)

simulation = Simulation(model, Δt=1.25, stop_iteration=3)

f(model) = model.clock.time^2 # scalar output

zC = znodes(grid, Center())
g(model) = model.clock.time .* exp.(zC) # vector/profile output

xC, yF = xnodes(grid, Center()), ynodes(grid, Face())
XC = [xC[i] for i in 1:Nx, j in 1:Ny]
YF = [yF[j] for i in 1:Nx, j in 1:Ny]
h(model) = @. model.clock.time * sin(XC) * cos(YF) # xy slice output

outputs = Dict("scalar" => f, "profile" => g, "slice" => h)

dims = Dict("scalar" => (), "profile" => ("zC",), "slice" => ("xC", "yC"))

output_attributes = Dict(
    "scalar"  => Dict("long_name" => "Some scalar", "units" => "bananas"),
    "profile" => Dict("long_name" => "Some vertical profile", "units" => "watermelons"),
    "slice"   => Dict("long_name" => "Some slice", "units" => "mushrooms"))

global_attributes = Dict("location" => "Bay of Fundy", "onions" => 7)

simulation.output_writers[:things] =
    NetCDFWriter(model, outputs,
                 schedule=IterationInterval(1), filename="things.nc", dimensions=dims, verbose=true,
                 global_attributes=global_attributes, output_attributes=output_attributes)
```

`NetCDFWriter` can also be configured for `outputs` that are interpolated or regridded
to a different grid than `model.grid`. To use this functionality, include the keyword argument
`grid = output_grid`.

```@example
using Oceananigans
using Oceananigans.Fields: interpolate!

grid = RectilinearGrid(size=(1, 1, 8), extent=(1, 1, 1));
model = NonhydrostaticModel(; grid)

coarse_grid = RectilinearGrid(size=(grid.Nx, grid.Ny, grid.Nz÷2), extent=(grid.Lx, grid.Ly, grid.Lz))
coarse_u = Field{Face, Center, Center}(coarse_grid)

interpolate_u(model) = interpolate!(coarse_u, model.velocities.u)
outputs = (; u = interpolate_u)

output_writer = NetCDFWriter(model, outputs;
                             grid = coarse_grid,
                             filename = "coarse_u.nc",
                             schedule = IterationInterval(1))
```
"""
function NetCDFWriter(model::AbstractModel, outputs;
                      filename,
                      schedule,
                      grid = model.grid,
                      dir = ".",
                      array_type = Array{Float32},
                      indices = (:, :, :),
                      global_attributes = Dict(),
                      output_attributes = Dict(),
                      dimensions = Dict(),
                      with_halos = false,
                      include_grid_metrics = true,
                      overwrite_existing = nothing,
                      verbose = false,
                      deflatelevel = 0,
                      part = 1,
                      file_splitting = NoFileSplitting(),
                      dimension_name_generator = trilocation_dim_name,
                      dimension_type = Float64)

    if with_halos && indices != (:, :, :)
        throw(ArgumentError("If with_halos=true then you cannot pass indices: $indices"))
    end

    mkpath(dir)
    filename = auto_extension(filename, ".nc")
    filepath = abspath(joinpath(dir, filename))

    initialize!(file_splitting, model)

    schedule = materialize_schedule(schedule)
    update_file_splitting_schedule!(file_splitting, filepath)

    if isnothing(overwrite_existing)
        if isfile(filepath)
            overwrite_existing = false
        else
            overwrite_existing = true
        end
    else
        if isfile(filepath) && !overwrite_existing
            @warn "$filepath already exists and `overwrite_existing = false`. Mode will be set to append to existing file. " *
                  "You might experience errors when writing output if the existing file belonged to a different simulation!"

        elseif isfile(filepath) && overwrite_existing
            @warn "Overwriting existing $filepath."
        end
    end

    outputs = Dict(string(name) => construct_output(outputs[name], grid, indices, with_halos) for name in keys(outputs))

    output_attributes = dictify(output_attributes)
    global_attributes = dictify(global_attributes)
    dimensions = dictify(dimensions)

    # Ensure we can add any kind of metadata to the attributes later by converting to Dict{Any, Any}.
    global_attributes = Dict{Any, Any}(global_attributes)

    dataset, outputs, schedule = initialize_nc_file(model,
                                                    grid,
                                                    filepath,
                                                    outputs,
                                                    schedule,
                                                    array_type,
                                                    indices,
                                                    global_attributes,
                                                    output_attributes,
                                                    dimensions,
                                                    with_halos,
                                                    include_grid_metrics,
                                                    overwrite_existing,
                                                    deflatelevel,
                                                    dimension_name_generator,
                                                    dimension_type)

    return NetCDFWriter(grid,
                        filepath,
                        dataset,
                        outputs,
                        schedule,
                        array_type,
                        indices,
                        global_attributes,
                        output_attributes,
                        dimensions,
                        with_halos,
                        include_grid_metrics,
                        overwrite_existing,
                        verbose,
                        deflatelevel,
                        part,
                        file_splitting,
                        dimension_name_generator,
                        dimension_type)
end

#####
##### NetCDF file initialization
#####

function initialize_nc_file(model,
                            grid,
                            filepath,
                            outputs,
                            schedule,
                            array_type,
                            indices,
                            global_attributes,
                            output_attributes,
                            dimensions,
                            with_halos,
                            include_grid_metrics,
                            overwrite_existing,
                            deflatelevel,
                            dimension_name_generator,
                            dimension_type)

    mode = overwrite_existing ? "c" : "a"

    # Add useful metadata
    useful_attributes = Dict("date" => "This file was generated on $(now()) local time ($(now(UTC)) UTC).",
                             "Julia" => "This file was generated using " * versioninfo_with_gpu(),
                             "Oceananigans" => "This file was generated using " * oceananigans_versioninfo())

    if with_halos
        useful_attributes["output_includes_halos"] =
            "The outputs include data from the halo regions of the grid."
    end

    global_attributes = merge(useful_attributes, global_attributes)

    add_schedule_metadata!(global_attributes, schedule)

    # Convert schedule to TimeInterval and each output to WindowedTimeAverage if
    # schedule::AveragedTimeInterval
    schedule, outputs = time_average_outputs(schedule, outputs, model)

    dims = gather_dimensions(outputs, grid, indices, with_halos, dimension_name_generator)

    # Open the NetCDF dataset file
    dataset = NCDataset(filepath, mode, attrib=sort(collect(pairs(global_attributes)), by=first))

    # Merge the default with any user-supplied output attributes, ensuring the user-supplied ones
    # can overwrite the defaults.
    output_attributes = merge(default_dimension_attributes(grid, dimension_name_generator),
                              default_output_attributes(model),
                              output_attributes)

    # Define variables for each dimension and attributes if this is a new file.
    if mode == "c"
        # This metadata is to support `FieldTimeSeries`.
        write_grid_reconstruction_data!(dataset, grid; array_type, deflatelevel)

        # DateTime and TimeDate are both <: AbstractTime
        time_attrib = model.clock.time isa AbstractTime ?
            Dict("long_name" => "Time", "units" => "seconds since 2000-01-01 00:00:00") :
            Dict("long_name" => "Time", "units" => "seconds")

        create_time_dimension!(dataset, attrib=time_attrib, dimension_type=dimension_type)
        create_spatial_dimensions!(dataset, dims, output_attributes; deflatelevel=1, dimension_type=dimension_type)

        time_independent_vars = Dict()

        if include_grid_metrics
            grid_metrics = gather_grid_metrics(grid, indices, dimension_name_generator)
            merge!(time_independent_vars, grid_metrics)
        end

        if grid isa ImmersedBoundaryGrid
            immersed_boundary_vars = gather_immersed_boundary(grid, indices, dimension_name_generator)
            merge!(time_independent_vars, immersed_boundary_vars)
        end

        if !isempty(time_independent_vars)
            for (output_name, output) in sort(collect(pairs(time_independent_vars)), by=first)
                output = construct_output(output, grid, indices, with_halos)
                attrib = haskey(output_attributes, output_name) ? output_attributes[output_name] : Dict()
                materialized = materialize_output(output, model)

                define_output_variable!(model,
                                        dataset,
                                        materialized,
                                        output_name;
                                        array_type,
                                        deflatelevel,
                                        attrib,
                                        dimensions,
                                        filepath, # for better error messages
                                        dimension_name_generator,
                                        time_dependent = false,
                                        with_halos,
                                        dimension_type)

                save_output!(dataset, output, model, output_name, array_type)
            end
        end

        for (output_name, output) in sort(collect(pairs(outputs)), by=first)
            attrib = haskey(output_attributes, output_name) ? output_attributes[output_name] : Dict()
            materialized = materialize_output(output, model)

            define_output_variable!(model,
                                    dataset,
                                    materialized,
                                    output_name;
                                    array_type,
                                    deflatelevel,
                                    attrib,
                                    dimensions,
                                    filepath, # for better error messages
                                    dimension_name_generator,
                                    time_dependent = true,
                                    with_halos,
                                    dimension_type)
        end

        sync(dataset)
    end

    close(dataset)

    return dataset, outputs, schedule
end

initialize_nc_file(ow::NetCDFWriter, model) = initialize_nc_file(model,
                                                                 ow.grid,
                                                                 ow.filepath,
                                                                 ow.outputs,
                                                                 ow.schedule,
                                                                 ow.array_type,
                                                                 ow.indices,
                                                                 ow.global_attributes,
                                                                 ow.output_attributes,
                                                                 ow.dimensions,
                                                                 ow.with_halos,
                                                                 ow.include_grid_metrics,
                                                                 ow.overwrite_existing,
                                                                 ow.deflatelevel,
                                                                 ow.dimension_name_generator,
                                                                 ow.dimension_type)

#####
##### Variable definition
#####

materialize_output(func, model) = func(model)
materialize_output(field::AbstractField, model) = field
materialize_output(particles::LagrangianParticles, model) = particles
materialize_output(output::WindowedTimeAverage{<:AbstractField}, model) = output

""" Defines empty variables for 'custom' user-supplied `output`. """
function define_output_variable!(model, dataset, output, output_name; array_type,
                                 deflatelevel, attrib, dimension_name_generator,
                                 time_dependent, with_halos,
                                 dimensions, filepath, dimension_type=Float64)

    if output_name ∉ keys(dimensions)
        msg = string("dimensions[$output_name] for output $output_name=$(typeof(output)) into $filepath" *
                     " must be provided when constructing NetCDFWriter")
        throw(ArgumentError(msg))
    end

    dims = dimensions[output_name]
    FT = eltype(array_type)
    all_dims = time_dependent ? (dims..., "time") : dims
    defVar(dataset, output_name, FT, all_dims; deflatelevel, attrib)

    return nothing
end

""" Defines empty field variable. """
function define_output_variable!(model, dataset, output::AbstractField, output_name; array_type,
                                 deflatelevel, attrib, dimension_name_generator,
                                 time_dependent, with_halos,
                                 dimensions, filepath, dimension_type=Float64)

    # If the output is the free surface, we need to handle it differently since it will be writen as a 3D array with a singleton dimension for the z-coordinate
    if output_name == "η" && output == view(model.free_surface.η, output.indices...)
        local default_dimension_name_generator = dimension_name_generator
        dimension_name_generator = (var_name, grid, LX, LY, LZ, dim) -> dimension_name_generator_free_surface(default_dimension_name_generator, var_name, grid, LX, LY, LZ, dim)
    end
    defVar(dataset, output_name, output; array_type, time_dependent, with_halos, dimension_name_generator, deflatelevel, attrib, dimension_type, write_data=false)
    return nothing
end

""" Defines empty field variable for `WindowedTimeAverage`s over fields. """
define_output_variable!(model, dataset, output::WindowedTimeAverage{<:AbstractField}, output_name; kwargs...) =
    define_output_variable!(model, dataset, output.operand, output_name; kwargs...)


""" Defines empty variable for particle trackting. """
function define_output_variable!(model, dataset, output::LagrangianParticles, output_name; array_type,
                                 deflatelevel, kwargs...)

    particle_fields = eltype(output.properties) |> fieldnames .|> string
    T = eltype(array_type)

    for particle_field in particle_fields
        defVar(dataset, particle_field, T, ("particle_id", "time"); deflatelevel)
    end

    return nothing
end

#####
##### Write output
#####

Base.open(nc::NetCDFWriter) = NCDataset(nc.filepath, "a")
Base.close(nc::NetCDFWriter) = close(nc.dataset)

# Saving outputs with no time dependence (e.g. grid metrics)
function save_output!(ds, output, model, output_name, array_type)
    fetched = fetch_output(output, model)
    data = convert_output(fetched, array_type)
    data = squeeze_data(output, data)
    colons = Tuple(Colon() for _ in 1:ndims(data))
    ds[output_name][colons...] = data
    return nothing
end

# Saving time-dependent outputs
function save_output!(ds, output, model, ow, time_index, output_name)
    data = fetch_and_convert_output(output, model, ow)
    data = squeeze_data(output, data)
    colons = Tuple(Colon() for _ in 1:ndims(data))
    ds[output_name][colons..., time_index:time_index] = data
    return nothing
end

function save_output!(ds, output::LagrangianParticles, model, ow, time_index, name)
    data = fetch_and_convert_output(output, model, ow)
    for (particle_field, vals) in pairs(data)
        ds[string(particle_field)][:, time_index] = vals
    end

    return nothing
end

# Convert to a base Julia type (a float or DateTime).
float_or_date_time(t) = t
float_or_date_time(t::AbstractTime) = DateTime(t)

"""
    write_output!(ow::NetCDFWriter, model)

Write output to netcdf file `output_writer.filepath` at specified intervals. Increments the `time` dimension
every time an output is written to the file.
"""
function write_output!(ow::NetCDFWriter, model::AbstractModel)
    # Start a new file if the file_splitting(model) is true
    ow.file_splitting(model) && start_next_file(model, ow)
    update_file_splitting_schedule!(ow.file_splitting, ow.filepath)

    ow.dataset = open(ow)

    ds, verbose, filepath = ow.dataset, ow.verbose, ow.filepath

    time_index = length(ds["time"]) + 1
    ds["time"][time_index] = float_or_date_time(model.clock.time)

    if verbose
        @info "Writing to NetCDF: $filepath..."
        @info "Computing NetCDF outputs for time index $(time_index): $(keys(ow.outputs))..."

        # Time and file size before computing any outputs.
        t0, sz0 = time_ns(), filesize(filepath)
    end

    for (output_name, output) in ow.outputs
        # Time before computing this output.
        verbose && (t0′ = time_ns())

        save_output!(ds, output, model, ow, time_index, output_name)

        if verbose
            # Time after computing this output.
            t1′ = time_ns()
            @info "Computing $output_name done: time=$(prettytime((t1′-t0′) / 1e9))"
        end
    end

    sync(ds)
    close(ow)

    if verbose
        # Time and file size after computing and writing all outputs to disk.
        t1, sz1 = time_ns(), filesize(filepath)
        verbose && @info begin
            @sprintf("Writing done: time=%s, size=%s, Δsize=%s",
                    prettytime((t1-t0)/1e9), pretty_filesize(sz1), pretty_filesize(sz1-sz0))
        end
    end

    return nothing
end

#####
##### Show
#####

Base.summary(ow::NetCDFWriter) =
    string("NetCDFWriter writing ", prettykeys(ow.outputs), " to ", ow.filepath, " on ", summary(ow.schedule))

function Base.show(io::IO, ow::NetCDFWriter)
    dims = NCDataset(ow.filepath, "r") do ds
        join([dim * "(" * string(length(ds[dim])) * "), "
              for dim in keys(ds.dim)])[1:end-2]
    end

    averaging_schedule = output_averaging_schedule(ow)
    num_outputs = length(ow.outputs)

    print(io, "NetCDFWriter scheduled on $(summary(ow.schedule)):", "\n",
              "├── filepath: ", relpath(ow.filepath), "\n",
              "├── dimensions: $dims", "\n",
              "├── $num_outputs outputs: ", prettykeys(ow.outputs), show_averaging_schedule(averaging_schedule), "\n",
              "├── array_type: ", show_array_type(ow.array_type), "\n",
              "├── file_splitting: ", summary(ow.file_splitting), "\n",
              "└── file size: ", pretty_filesize(filesize(ow.filepath)))
end

#####
##### File splitting
#####

function start_next_file(model, ow::NetCDFWriter)
    verbose = ow.verbose

    verbose && @info begin
        schedule_type = summary(ow.file_splitting)
        "Splitting output because $(schedule_type) is activated."
    end

    if ow.part == 1
        part1_path = replace(ow.filepath, r".nc$" => "_part1.nc")
        verbose && @info "Renaming first part: $(ow.filepath) -> $part1_path"
        mv(ow.filepath, part1_path, force=ow.overwrite_existing)
        ow.filepath = part1_path
    end

    ow.part += 1
    ow.filepath = replace(ow.filepath, r"part\d+.nc$" => "part" * string(ow.part) * ".nc")
    ow.overwrite_existing && isfile(ow.filepath) && rm(ow.filepath, force=true)
    verbose && @info "Now writing to: $(ow.filepath)"

    initialize_nc_file(ow, model)

    return nothing
end

#####
##### More utils
#####

ext(::Type{NetCDFWriter}) = ".nc"

end # module
