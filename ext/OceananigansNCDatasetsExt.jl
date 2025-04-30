module OceananigansNCDatasetsExt

using NCDatasets

using Dates: AbstractTime, UTC, now
using Printf: @sprintf

using Oceananigans.Fields

using Oceananigans: initialize!, prettytime, pretty_filesize, AbstractModel
using Oceananigans.Grids: Center, Face, Flat, AbstractGrid, RectilinearGrid, LatitudeLongitudeGrid, StaticVerticalDiscretization
using Oceananigans.Grids: topology, halo_size, xspacings, yspacings, zspacings, λspacings, φspacings,
                          parent_index_range, ξnodes, ηnodes, rnodes, validate_index, peripheral_node
using Oceananigans.Fields: reduced_dimensions, reduced_location, location
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Models: ShallowWaterModel, LagrangianParticles
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBottom, GFBIBG, GridFittedBoundary
using Oceananigans.TimeSteppers: float_or_date_time
using Oceananigans.BuoyancyFormulations: BuoyancyForce, BuoyancyTracer, SeawaterBuoyancy, LinearEquationOfState
using Oceananigans.Utils: TimeInterval, IterationInterval, WallTimeInterval
using Oceananigans.Utils: versioninfo_with_gpu, oceananigans_versioninfo, prettykeys
using SeawaterPolynomials: BoussinesqEquationOfState

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

import Oceananigans: write_output!
import Oceananigans.OutputWriters: NetCDFWriter

const c = Center()
const f = Face()
const BoussinesqSeawaterBuoyancy = SeawaterBuoyancy{FT, <:BoussinesqEquationOfState, T, S} where {FT, T, S}
const BuoyancyBoussinesqEOSModel = BuoyancyForce{<:BoussinesqSeawaterBuoyancy, g} where {g}

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

#####
##### Dimension name generators
#####

function suffixed_dim_name_generator(var_name, grid::AbstractGrid{FT, TX}, LX, LY, LZ, dim::Val{:x}; connector="_", location_letters) where {FT, TX}
    if TX == Flat || isnothing(LX)
        return ""
    else
        return "$(var_name)" * connector * location_letters
    end
end

function suffixed_dim_name_generator(var_name, grid::AbstractGrid{FT, TX, TY}, LX, LY, LZ, dim::Val{:y}; connector="_", location_letters) where {FT, TX, TY}
    if TY == Flat || isnothing(LY)
        return ""
    else
        return "$(var_name)" * connector * location_letters
    end
end

function suffixed_dim_name_generator(var_name, grid::AbstractGrid{FT, TX, TY, TZ}, LX, LY, LZ, dim::Val{:z}; connector="_", location_letters) where {FT, TX, TY, TZ}
    if TZ == Flat || isnothing(LZ)
        return ""
    else
        return "$(var_name)" * connector * location_letters
    end
end

suffixed_dim_name_generator(var_name, ::StaticVerticalDiscretization, LX, LY, LZ, dim::Val{:z}; connector="_", location_letters) = var_name * connector * location_letters

loc2letter(::Face, full=true) = "f"
loc2letter(::Center, full=true) = "c"
loc2letter(::Nothing, full=true) = full ? "a" : ""

minimal_location_string(::RectilinearGrid, LX, LY, LZ, ::Val{:x}) = loc2letter(LX, false)
minimal_location_string(::RectilinearGrid, LX, LY, LZ, ::Val{:y}) = loc2letter(LY, false)

minimal_location_string(::LatitudeLongitudeGrid, LX, LY, LZ, ::Val{:x}) = loc2letter(LX, false) * loc2letter(LY, false)
minimal_location_string(::LatitudeLongitudeGrid, LX, LY, LZ, ::Val{:y}) = loc2letter(LX, false) * loc2letter(LY, false)

minimal_location_string(grid::AbstractGrid,             LX, LY, LZ, dim::Val{:z}) = minimal_location_string(grid.z, LX, LY, LZ, dim)
minimal_location_string(::StaticVerticalDiscretization, LX, LY, LZ, dim::Val{:z}) = loc2letter(LZ, false)
minimal_location_string(grid,                           LX, LY, LZ, dim)          = loc2letter(LX, false) * loc2letter(LY, false) * loc2letter(LZ, false)

minimal_dim_name(var_name, grid, LX, LY, LZ, dim) =
    suffixed_dim_name_generator(var_name, grid, LX, LY, LZ, dim, connector="_", location_letters=minimal_location_string(grid, LX, LY, LZ, dim))

minimal_dim_name(var_name, grid::ImmersedBoundaryGrid, args...) = minimal_dim_name(var_name, grid.underlying_grid, args...)



trilocation_location_string(::RectilinearGrid, LX, LY, LZ, ::Val{:x}) = loc2letter(LX) * "aa"
trilocation_location_string(::RectilinearGrid, LX, LY, LZ, ::Val{:y}) = "a" * loc2letter(LY) * "a"

trilocation_location_string(::LatitudeLongitudeGrid, LX, LY, LZ, ::Val{:x}) = loc2letter(LX) * loc2letter(LY) * "a"
trilocation_location_string(::LatitudeLongitudeGrid, LX, LY, LZ, ::Val{:y}) = loc2letter(LX) * loc2letter(LY) * "a"

trilocation_location_string(grid::AbstractGrid,             LX, LY, LZ, dim::Val{:z}) = trilocation_location_string(grid.z, LX, LY, LZ, dim)
trilocation_location_string(::StaticVerticalDiscretization, LX, LY, LZ, dim::Val{:z}) = "aa" * loc2letter(LZ)
trilocation_location_string(grid,                           LX, LY, LZ, dim)          = loc2letter(LX) * loc2letter(LY) * loc2letter(LZ)

trilocation_dim_name(var_name, grid, LX, LY, LZ, dim) =
    suffixed_dim_name_generator(var_name, grid, LX, LY, LZ, dim, connector="_", location_letters=trilocation_location_string(grid, LX, LY, LZ, dim))

trilocation_dim_name(var_name, grid::ImmersedBoundaryGrid, args...) = trilocation_dim_name(var_name, grid.underlying_grid, args...)


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

# For Immersed Boundary Grids (IBG) with a Grid Fitted Bottom (GFB)
function gather_immersed_boundary(grid::GFBIBG, indices, dim_name_generator)
    op_mask_ccc = KernelFunctionOperation{Center, Center, Center}(peripheral_node, grid, Center(), Center(), Center())
    op_mask_fcc = KernelFunctionOperation{Face, Center, Center}(peripheral_node, grid, Face(), Center(), Center())
    op_mask_cfc = KernelFunctionOperation{Center, Face, Center}(peripheral_node, grid, Center(), Face(), Center())
    op_mask_ccf = KernelFunctionOperation{Center, Center, Face}(peripheral_node, grid, Center(), Center(), Face())

    return Dict("bottom_height" => Field(grid.immersed_boundary.bottom_height; indices),
                "immersed_boundary_mask_ccc" => Field(op_mask_ccc; indices),
                "immersed_boundary_mask_fcc" => Field(op_mask_fcc; indices),
                "immersed_boundary_mask_cfc" => Field(op_mask_cfc; indices),
                "immersed_boundary_mask_ccf" => Field(op_mask_ccf; indices))
end

const GFBoundaryIBG = ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:GridFittedBoundary}

# For Immersed Boundary Grids (IBG) with a Grid Fitted Boundary (also GFB!)
function gather_immersed_boundary(grid::GFBoundaryIBG, indices, dim_name_generator)
    op_mask_ccc = KernelFunctionOperation{Center, Center, Center}(peripheral_node, grid, Center(), Center(), Center())
    op_mask_fcc = KernelFunctionOperation{Face, Center, Center}(peripheral_node, grid, Face(), Center(), Center())
    op_mask_cfc = KernelFunctionOperation{Center, Face, Center}(peripheral_node, grid, Center(), Face(), Center())
    op_mask_ccf = KernelFunctionOperation{Center, Center, Face}(peripheral_node, grid, Center(), Center(), Face())

    return Dict("immersed_boundary_mask" => Field(grid.immersed_boundary.mask; indices),
                "immersed_boundary_mask_ccc" => Field(op_mask_ccc; indices),
                "immersed_boundary_mask_fcc" => Field(op_mask_fcc; indices),
                "immersed_boundary_mask_cfc" => Field(op_mask_cfc; indices),
                "immersed_boundary_mask_ccf" => Field(op_mask_ccf; indices))
end

#####
##### Mapping outputs/fields to dimensions
#####

function field_dimensions(field::AbstractField, grid::RectilinearGrid, dim_name_generator)
    LX, LY, LZ = location(field)

    x_dim_name = dim_name_generator("x", grid, LX(), nothing, nothing, Val(:x))
    y_dim_name = dim_name_generator("y", grid, nothing, LY(), nothing, Val(:y))
    z_dim_name = dim_name_generator("z", grid, nothing, nothing, LZ(), Val(:z))

    x_dim_name = isempty(x_dim_name) ? tuple() : tuple(x_dim_name)
    y_dim_name = isempty(y_dim_name) ? tuple() : tuple(y_dim_name)
    z_dim_name = isempty(z_dim_name) ? tuple() : tuple(z_dim_name)

    return tuple(x_dim_name..., y_dim_name..., z_dim_name...)
end

function field_dimensions(field::AbstractField, grid::LatitudeLongitudeGrid, dim_name_generator)
    LΛ, LΦ, LZ = location(field)

    λ_dim_name = dim_name_generator("λ", grid, LΛ(), nothing, nothing, Val(:x))
    φ_dim_name = dim_name_generator("φ",  grid, nothing, LΦ(), nothing, Val(:y))
    z_dim_name = dim_name_generator("z",         grid, nothing, nothing, LZ(), Val(:z))

    λ_dim_name = isempty(λ_dim_name) ? tuple() : tuple(λ_dim_name)
    φ_dim_name = isempty(φ_dim_name) ? tuple() : tuple(φ_dim_name)
    z_dim_name = isempty(z_dim_name) ? tuple() : tuple(z_dim_name)

    return tuple(λ_dim_name..., φ_dim_name..., z_dim_name...)
end

field_dimensions(field::AbstractField, grid::ImmersedBoundaryGrid, dim_name_generator) =
    field_dimensions(field, grid.underlying_grid, dim_name_generator)

field_dimensions(field::AbstractField, dim_name_generator) =
    field_dimensions(field, field.grid, dim_name_generator)

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

#####
##### Gather grid reconstruction attributes (also used for FieldTimeSeries support)
#####

function grid_reconstruction_attributes(grid::RectilinearGrid)
    TX, TY, TZ = topology(grid)

    dims = Dict()

    attrs = Dict("type" => string(nameof(typeof(grid))),
                 "eltype" => string(eltype(grid)),
                 "TX" => string(TX),
                 "TY" => string(TY),
                 "TZ" => string(TZ),
                 "Nx" => grid.Nx,
                 "Ny" => grid.Ny,
                 "Nz" => grid.Nz,
                 "Hx" => grid.Hx,
                 "Hy" => grid.Hy,
                 "Hz" => grid.Hz)

    if TX == Flat
        attrs["x_spacing"] = "flat"
    else
        attrs["x_spacing"] = grid.Δxᶠᵃᵃ isa Number ? "regular" : "irregular"
        dims["x_f"] = grid.xᶠᵃᵃ[1:grid.Nx+1]
    end

    if TY == Flat
        attrs["y_spacing"] = "flat"
    else
        attrs["y_spacing"] = grid.Δyᵃᶠᵃ isa Number ? "regular" : "irregular"
        dims["y_f"] = grid.yᵃᶠᵃ[1:grid.Ny+1]
    end

    if TZ == Flat
        attrs["z_spacing"] = "flat"
    else
        attrs["z_spacing"] = grid.z.Δᵃᵃᶠ isa Number ? "regular" : "irregular"
        dims["z_f"] = grid.z.cᵃᵃᶠ[1:grid.Nz+1]
    end

    return attrs, dims
end

function grid_reconstruction_attributes(grid::LatitudeLongitudeGrid)
    TX, TY, TZ = topology(grid)

    dims = Dict()

    attrs = Dict("type" => string(nameof(typeof(grid))),
                 "eltype" => string(eltype(grid)),
                 "TX" => string(TX),
                 "TY" => string(TY),
                 "TZ" => string(TZ),
                 "Nx" => grid.Nx,
                 "Ny" => grid.Ny,
                 "Nz" => grid.Nz,
                 "Hx" => grid.Hx,
                 "Hy" => grid.Hy,
                 "Hz" => grid.Hz)

    if TX == Flat
        attrs["λ_spacing"] = "flat"
    else
        attrs["λ_spacing"] = grid.Δλᶠᵃᵃ isa Number ? "regular" : "irregular"
        dims["λ_f"] = grid.λᶠᵃᵃ[1:grid.Nx+1]
    end

    if TY == Flat
        attrs["φ_spacing"] = "flat"
    else
        attrs["φ_spacing"] = grid.Δφᵃᶠᵃ isa Number ? "regular" : "irregular"
        dims["φ_f"] = grid.φᵃᶠᵃ[1:grid.Ny+1]
    end

    if TZ == Flat
        attrs["z_spacing"] = "flat"
    else
        attrs["z_spacing"] = grid.z.Δᵃᵃᶠ isa Number ? "regular" : "irregular"
        dims["z_f"] = grid.z.cᵃᵃᶠ[1:grid.Nz+1]
    end

    return attrs, dims
end

function grid_reconstruction_attributes(ibg::ImmersedBoundaryGrid)
    attrs, dims = grid_reconstruction_attributes(ibg.underlying_grid)

    immersed_attrs = Dict("immersed_boundary_type" => string(nameof(typeof(ibg.immersed_boundary))))

    attrs = merge(attrs, immersed_attrs)

    return attrs, dims
end

function write_grid_reconstruction_metadata!(ds, grid, indices, array_type, deflatelevel)
    grid_attrs, grid_dims = grid_reconstruction_attributes(grid)

    ds_grid = defGroup(ds, "grid_reconstruction"; attrib = sort(collect(pairs(grid_attrs)), by=first))

    for (dim_name, dim_array) in grid_dims
        defVar(ds_grid, dim_name, array_type(dim_array), (dim_name,); deflatelevel)
    end

    return ds
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
                      dimension_name_generator = trilocation_dim_name)

    if with_halos && indices != (:, :, :)
        throw(ArgumentError("If with_halos=true then you cannot pass indices: $indices"))
    end

    mkpath(dir)
    filename = auto_extension(filename, ".nc")
    filepath = abspath(joinpath(dir, filename))

    initialize!(file_splitting, model)
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
                                                    dimension_name_generator)

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
                        dimension_name_generator)
end

#####
##### NetCDF file initialization
#####

function initialize_nc_file(model,
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
                            dimension_name_generator)

    grid = model.grid
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
        write_grid_reconstruction_metadata!(dataset, grid, indices, array_type, deflatelevel)

        # DateTime and TimeDate are both <: AbstractTime
        time_attrib = model.clock.time isa AbstractTime ?
            Dict("long_name" => "Time", "units" => "seconds since 2000-01-01 00:00:00") :
            Dict("long_name" => "Time", "units" => "seconds")

        # Create an unlimited dimension "time"
        # Time should always be Float64 to be extra safe from rounding errors.
        # See: https://github.com/CliMA/Oceananigans.jl/issues/3056
        defDim(dataset, "time", Inf)
        defVar(dataset, "time", Float64, ("time",), attrib=time_attrib)

        # Create spatial dimensions as variables whose dimensions are themselves.
        # Each should already have a default attribute.
        for (dim_name, dim_array) in dims
            defVar(dataset, dim_name, array_type(dim_array), (dim_name,),
                   deflatelevel=deflatelevel, attrib=output_attributes[dim_name])
        end

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
            for (name, output) in sort(collect(pairs(time_independent_vars)), by=first)
                output = construct_output(output, grid, indices, with_halos)
                attributes = haskey(output_attributes, name) ? output_attributes[name] : Dict()
                materialized = materialize_output(output, model)

                define_output_variable!(dataset,
                                        materialized,
                                        name,
                                        array_type,
                                        deflatelevel,
                                        attributes,
                                        dimensions,
                                        filepath, # for better error messages
                                        dimension_name_generator,
                                        false) # time_dependent = false

                save_output!(dataset, output, model, name, array_type)
            end
        end

        for (name, output) in sort(collect(pairs(outputs)), by=first)
            attributes = haskey(output_attributes, name) ? output_attributes[name] : Dict()
            materialized = materialize_output(output, model)

            define_output_variable!(dataset,
                                    materialized,
                                    name,
                                    array_type,
                                    deflatelevel,
                                    attributes,
                                    dimensions,
                                    filepath, # for better error messages
                                    dimension_name_generator,
                                    true) # time_dependent = true)
        end

        sync(dataset)
    end

    close(dataset)

    return dataset, outputs, schedule
end

initialize_nc_file(ow::NetCDFWriter, model) = initialize_nc_file(model,
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
                                                                 ow.dimension_name_generator)

#####
##### Variable definition
#####

materialize_output(func, model) = func(model)
materialize_output(field::AbstractField, model) = field
materialize_output(particles::LagrangianParticles, model) = particles
materialize_output(output::WindowedTimeAverage{<:AbstractField}, model) = output

""" Defines empty variables for 'custom' user-supplied `output`. """
function define_output_variable!(dataset, output, name, array_type,
                                 deflatelevel, attrib, dimensions, filepath,
                                 dimension_name_generator, time_dependent)

    if name ∉ keys(dimensions)
        msg = string("dimensions[$name] for output $name=$(typeof(output)) into $filepath" *
                     " must be provided when constructing NetCDFWriter")
        throw(ArgumentError(msg))
    end

    dims = dimensions[name]
    FT = eltype(array_type)
    defVar(dataset, name, FT, (dims..., "time"); deflatelevel, attrib)

    return nothing
end

""" Defines empty field variable. """
function define_output_variable!(dataset, output::AbstractField, name, array_type,
                                 deflatelevel, attrib, dimensions, filepath,
                                 dimension_name_generator, time_dependent)

    dims = field_dimensions(output, dimension_name_generator)
    FT = eltype(array_type)

    all_dims = time_dependent ? (dims..., "time") : dims

    defVar(dataset, name, FT, all_dims; deflatelevel, attrib)

    return nothing
end

""" Defines empty field variable for `WindowedTimeAverage`s over fields. """
define_output_variable!(dataset, output::WindowedTimeAverage{<:AbstractField}, args...) =
    define_output_variable!(dataset, output.operand, args...)


""" Defines empty variable for particle trackting. """
function define_output_variable!(dataset, output::LagrangianParticles, name, array_type,
                                 deflatelevel, args...)

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
function save_output!(ds, output, model, name, array_type)
    fetched = fetch_output(output, model)
    data = convert_output(fetched, array_type)
    data = drop_output_dims(output, data)
    colons = Tuple(Colon() for _ in 1:ndims(data))
    ds[name][colons...] = data
    return nothing
end

# Saving time-dependent outputs
function save_output!(ds, output, model, ow, time_index, name)
    data = fetch_and_convert_output(output, model, ow)
    data = drop_output_dims(output, data)
    colons = Tuple(Colon() for _ in 1:ndims(data))
    ds[name][colons..., time_index:time_index] = data
    return nothing
end

function save_output!(ds, output::LagrangianParticles, model, ow, time_index, name)
    data = fetch_and_convert_output(output, model, ow)
    for (particle_field, vals) in pairs(data)
        ds[string(particle_field)][:, time_index] = vals
    end

    return nothing
end

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

    for (name, output) in ow.outputs
        # Time before computing this output.
        verbose && (t0′ = time_ns())

        save_output!(ds, output, model, ow, time_index, name)

        if verbose
            # Time after computing this output.
            t1′ = time_ns()
            @info "Computing $name done: time=$(prettytime((t1′-t0′) / 1e9))"
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

drop_output_dims(output, data) = data # fallback
drop_output_dims(output::WindowedTimeAverage{<:Field}, data) = drop_output_dims(output.operand, data)

function drop_output_dims(field::Field, data)
    reduced_dims = reduced_dimensions(field)
    flat_dims = Tuple(i for (i, T) in enumerate(topology(field.grid)) if T == Flat)
    dims = (reduced_dims..., flat_dims...)
    dims = Tuple(Set(dims)) # ensure dims are unique
    return dropdims(data; dims)
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
              "└── array type: ", show_array_type(ow.array_type), "\n",
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
