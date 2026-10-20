using Oceananigans.Fields: Fields, AbstractField, FieldStatus, compute!, compute_at!, set_status!
using Oceananigans.Grids: λnode, φnode, znodes, cpu_face_constructor_z, halo_size, new_data
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, AbstractGridFittedBottom, GridFittedBottom, PartialCellBottom
using Oceananigans.Utils: getregion

#####
##### Remapping from a Yin-Yang grid to a grid in geographic coordinates. Each destination node is interpolated
##### from the panel that owns it, with the stencils and weights of the halo tables, and the components of
##### vectors are rotated into the geographic basis.
#####

struct YinYangRemapping{G, L, S, T}
    destination_grid :: G     # the grid to which fields are remapped
    destination_location :: L # (ℓx, ℓy) of the destination nodes
    source_locations :: S     # tuple of the (ℓx, ℓy) of the source nodes, one per table of each panel
    tables :: T               # (Yin tables, Yang tables), each with one table per source location
end

location_summary(location) = string("(", join(map(ℓ -> string(nameof(typeof(ℓ))), location), ", "), ")")

Base.summary(remapping::YinYangRemapping) =
    string("YinYangRemapping from ", join(map(location_summary, remapping.source_locations), " and "),
           " to ", location_summary(remapping.destination_location), " on ", summary(remapping.destination_grid),
           " with ", length(first(remapping.tables[1])), " Yin and ", length(first(remapping.tables[2])), " Yang nodes")

Base.show(io::IO, remapping::YinYangRemapping) = print(io, summary(remapping))

"""
    latitude_longitude_grid(grid::YinYangGridOfSomeKind; size = (4Nφ, 2Nφ), halo = halo_size(grid))

Return a global `LatitudeLongitudeGrid` with the vertical coordinate of the Yin-Yang `grid` and, by default,
the spacing of its panels, which have `Nφ` cells in latitude. If `grid` has an immersed bottom, so does the
returned grid, with the bottom height of the panel cell that contains each column.
"""
function latitude_longitude_grid(grid::YinYangGridOfSomeKind;
                                 size = (4, 2) .* yin_yang_connectivity(grid).core_size[2],
                                 halo = halo_size(grid))

    panel = first(grid)
    return LatitudeLongitudeGrid(architecture(grid), eltype(grid);
                                 size = (size[1], size[2], Base.size(grid, 3)), halo,
                                 z = cpu_face_constructor_z(panel),
                                 longitude = (-180, 180),
                                 latitude = (-90, 90),
                                 radius = panel.radius)
end

const ImmersedBottomYinYangGrid =
    ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:YinYangGrid, <:AbstractGridFittedBottom, <:Any, <:Any, <:AbstractSerialArchitecture}

function latitude_longitude_grid(grid::ImmersedBottomYinYangGrid; kwargs...)
    underlying_grid = latitude_longitude_grid(grid.underlying_grid; kwargs...)
    location = (Center(), Center())
    _, stencils = destination_stencils(grid, underlying_grid, location, (location,))
    panel_bottom_heights = map(r -> on_architecture(CPU(), getregion(grid.immersed_boundary.bottom_height, r)), (1, 2))
    bottom_height = [panel_bottom_heights[s.panel][round.(Int, only(s.fractional_indices))..., 1] for s in stencils]
    Nx, Ny, _ = Base.size(underlying_grid)
    return ImmersedBoundaryGrid(underlying_grid, with_bottom_height(grid.immersed_boundary, reshape(bottom_height, Nx, Ny)))
end

with_bottom_height(ib::GridFittedBottom, bottom_height) = GridFittedBottom(bottom_height, ib.immersed_condition)
with_bottom_height(ib::PartialCellBottom, bottom_height) = PartialCellBottom(bottom_height, ib.minimum_fractional_cell_height)

"""
    YinYangRemapping(grid::YinYangGridOfSomeKind;
                     destination_grid = latitude_longitude_grid(grid),
                     source_locations = ((Center(), Center()),),
                     destination_location = (Center(), Center()))

Return the tables that interpolate fields on the Yin-Yang `grid` at each of the horizontal `source_locations`
onto the nodes at `destination_location` of `destination_grid`, a grid in geographic coordinates. A horizontal
vector needs the locations of both components, such as `((Face(), Center()), (Center(), Face()))`.
The remapping interpolates point values for diagnostics and output: it does not conserve integrals,
and the source and destination fields must have the same vertical nodes.
"""
function YinYangRemapping(grid::YinYangGridOfSomeKind;
                          destination_grid = latitude_longitude_grid(grid),
                          source_locations = ((Center(), Center()),),
                          destination_location = (Center(), Center()))

    destination_grid isa MultiRegionGrids &&
        throw(ArgumentError("Fields cannot be remapped from a Yin-Yang grid onto a multi-region grid."))

    nodes, stencils = destination_stencils(grid, destination_grid, destination_location, source_locations)
    s = yin_yang_connectivity(grid).stencil_width
    arch = architecture(destination_grid)
    FT = eltype(grid)

    tables = ntuple(2) do r
        filled = findall(stencil -> stencil.panel == r, stencils)
        ntuple(length(source_locations)) do m
            YinYangInterpolationTable(arch, FT, s, nodes[filled],
                                      [stencils[n].starts[m] for n in filled],
                                      [stencils[n].fractional_indices[m] for n in filled],
                                      [stencils[n].ψ for n in filled])
        end
    end

    return YinYangRemapping(destination_grid, destination_location, source_locations, tables)
end

# The nodes at `(ℓx, ℓy)` of `destination_grid`, in column-major order, and the stencils that fill them
function destination_stencils(grid, destination_grid, (ℓx, ℓy), source_locations)
    connectivity = yin_yang_connectivity(grid)
    λᶠ, φᶠ = yin_yang_panel_face_coordinates(connectivity.core_size, connectivity.overlap)
    frame = panel_frame(λᶠ, φᶠ)
    interior_size = (length(λᶠ) - 1, length(φᶠ) - 1)
    destination = on_architecture(CPU(), destination_grid)

    Nx, Ny, _ = Base.size(destination, (ℓx, ℓy, Center()))
    nodes = [(i, j) for j in 1:Ny for i in 1:Nx]
    stencils = map(nodes) do (i, j)
        λ = Float64(λnode(i, j, 1, destination, ℓx, ℓy, Center()))
        φ = Float64(φnode(i, j, 1, destination, ℓx, ℓy, Center()))
        remapping_stencils(λ, φ, frame, interior_size, connectivity.stencil_width, source_locations)
    end

    unfilled = findfirst(isnothing, stencils)
    isnothing(unfilled) ||
        throw(ArgumentError("The destination node $(nodes[unfilled]) has no stencil of interior nodes on either panel."))

    return nodes, stencils
end

# The panel that fills the destination node at geographic `(λ, φ)` (degrees), and its stencils: the owner, unless
# it lacks interior stencils near its edges, or `nothing` if neither panel has them
function remapping_stencils(λ, φ, frame, interior_size, s, source_locations)
    owner = yin_yang_owned(λ, φ) ? 1 : 2
    for panel in (owner, 3 - owner)
        λ′, φ′ = panel == 1 ? (mod(λ + 180, 360) - 180, φ) : yin_yang_transform(λ, φ)
        ψ = panel == 1 ? zero(λ) : - yin_yang_rotation_angle(λ, φ) # Yin's basis is the geographic basis
        fractional_indices = map(location -> panel_fractional_indices(λ′, φ′, frame, location), source_locations)
        starts = map(((i★, j★),) -> interior_stencil_start(i★, j★, s, interior_size), fractional_indices)
        any(isnothing, starts) || return (; panel, starts, fractional_indices, ψ)
    end
    return nothing
end

# The panel components `(u, v)` of a horizontal vector on a Yin-Yang grid
const YinYangVectorComponents = Tuple{<:YinYangField, <:YinYangField}

# By default a scalar is remapped to its own location and a vector to cell centers
YinYangRemapping(destination_grid::AbstractGrid, source::YinYangField, destination_location = horizontal_location(source)) =
    YinYangRemapping(source.grid; destination_grid, destination_location, source_locations = (horizontal_location(source),))

YinYangRemapping(destination_grid::AbstractGrid, (u, v)::YinYangVectorComponents, destination_location = (Center(), Center())) =
    YinYangRemapping(u.grid; destination_grid, destination_location, source_locations = map(horizontal_location, (u, v)))

YinYangRemapping(destination::AbstractField, source) = YinYangRemapping(destination.grid, source, horizontal_location(destination))

function YinYangRemapping((uᵍ, vᵍ)::Tuple{<:AbstractField, <:AbstractField}, source::YinYangVectorComponents)
    horizontal_location(uᵍ) == horizontal_location(vᵍ) ||
        throw(ArgumentError("The geographic components of a remapped vector must share their horizontal location."))
    return YinYangRemapping(uᵍ.grid, source, horizontal_location(uᵍ))
end

#####
##### Remapping kernels
#####

function source_location_index(remapping, source)
    m = findfirst(==(horizontal_location(source)), remapping.source_locations)
    isnothing(m) &&
        throw(ArgumentError("$(summary(remapping)) cannot remap a field at $(location_summary(horizontal_location(source)))."))
    return m
end

function validate_remapping(destination, source, remapping)
    horizontal_location(destination) == remapping.destination_location &&
        Base.size(destination.grid) == Base.size(remapping.destination_grid) ||
        throw(ArgumentError("The destination field does not match the destination grid and location of $(summary(remapping))."))
    zᵈ = vertical_nodes(destination.grid, destination)
    zˢ = vertical_nodes(first(source.grid), source)
    length(zᵈ) == length(zˢ) && zᵈ ≈ zˢ ||
        throw(ArgumentError("The destination and source fields have different vertical nodes."))
    architecture(destination) == architecture(source) ||
        throw(ArgumentError("The destination and source fields are on different architectures."))
    return nothing
end

vertical_nodes(grid, field) = on_architecture(CPU(), znodes(grid, instantiated_location(field)[3]))

launch_remapping!(destination, table, kernel!, args::Vararg{Any, N}) where N =
    launch_yin_yang_interpolation!(destination, 1:Base.size(destination, 3), table, kernel!, args...)

"""
    remap!(destination, source::YinYangField, remapping = YinYangRemapping(destination, source))
    remap!((uᵍ, vᵍ), (u, v), remapping = YinYangRemapping((uᵍ, vᵍ), (u, v)))

Remap the scalar `source`, or the horizontal vector with panel components `(u, v)`, from a Yin-Yang grid
onto the interior nodes of `destination`, or onto the geographic components `(uᵍ, vᵍ)`. The halos of the
source need not be filled, and those of the destination are not filled. On an immersed grid, scalars are
interpolated from wet cells only, and nodes whose stencils hold no wet cell are set to zero.
"""
function remap!(destination::AbstractField, source::YinYangField, remapping::YinYangRemapping = YinYangRemapping(destination, source))
    validate_remapping(destination, source, remapping)
    m = source_location_index(remapping, source)
    loc = instantiated_location(source)

    for r in 1:2
        table = remapping.tables[r][m]
        launch_remapping!(destination, table, _interpolate_yin_yang_scalar!,
                          destination, getregion(source.data, r), getregion(source.grid, r), loc, table)
    end

    return destination
end

function remap!(geographic::Tuple{<:AbstractField, <:AbstractField}, panel::YinYangVectorComponents,
                remapping::YinYangRemapping = YinYangRemapping(geographic, panel))
    remap_vector_component!(geographic[1], panel, remapping, Val(1))
    remap_vector_component!(geographic[2], panel, remapping, Val(2))
    return geographic
end

# `component` is `Val(1)` for the eastward component and `Val(2)` for the northward one
function remap_vector_component!(destination, (u, v), remapping, component)
    validate_remapping(destination, u, remapping)
    mu = source_location_index(remapping, u)
    mv = source_location_index(remapping, v)

    for r in 1:2
        u_table = remapping.tables[r][mu]
        v_table = remapping.tables[r][mv]
        launch_remapping!(destination, u_table, _interpolate_yin_yang_vector_component!, destination,
                          getregion(u.data, r), getregion(v.data, r), nothing, nothing, u_table, v_table, component)
    end

    return destination
end

#####
##### Remapped fields
#####

# The operand of a remapped field
struct YinYangRemap{S, R, C}
    source :: S    # a field on a Yin-Yang grid, or the panel components of a vector
    remapping :: R # the `YinYangRemapping` of `source`
    component :: C # `Val(1)` or `Val(2)` for the components of a vector, `nothing` for a scalar
end

Base.summary(remap::YinYangRemap) = string("YinYangRemap of ", remap_source_summary(remap.source))

remap_source_summary(source) = summary(source)
remap_source_summary(source::Tuple) = string("(", summary(source[1]), ", ", summary(source[2]), ")")

const YinYangRemappedField = Field{<:Any, <:Any, <:Any, <:YinYangRemap}

first_component(source::Tuple) = first(source)
first_component(source) = source

function remapped_field(source, remapping, component)
    grid = remapping.destination_grid
    first_source = first_component(source)
    loc = (remapping.destination_location..., instantiated_location(first_source)[3])
    data = new_data(eltype(first_source), grid, loc)
    operand = YinYangRemap(source, remapping, component)
    return compute!(Field(loc, grid, data, nothing, (:, :, :), operand, FieldStatus()))
end

"""
    remapped_field(source, remapping = latitude_longitude_grid(grid))

Return a computed field that holds `source`, a field on a Yin-Yang `grid`, remapped with `remapping`,
or the pair of geographic components of the vector with panel components `source = (u, v)`.
`remapping` is a `YinYangRemapping` or a destination grid, onto which scalars are remapped at their own
location and vectors at cell centers.
"""
remapped_field(source::YinYangField, destination_grid::AbstractGrid = latitude_longitude_grid(source.grid)) =
    remapped_field(source, YinYangRemapping(destination_grid, source))

remapped_field(source::YinYangVectorComponents, destination_grid::AbstractGrid = latitude_longitude_grid(first(source).grid)) =
    remapped_field(source, YinYangRemapping(destination_grid, source))

remapped_field(source::YinYangField, remapping::YinYangRemapping) = remapped_field(source, remapping, nothing)

remapped_field(source::YinYangVectorComponents, remapping::YinYangRemapping) =
    (remapped_field(source, remapping, Val(1)), remapped_field(source, remapping, Val(2)))

function Fields.compute!(field::YinYangRemappedField, time = nothing)
    remap = field.operand
    compute_sources_at!(remap.source, time)
    remap_operand!(field, remap.source, remap.remapping, remap.component)
    set_status!(field.status, time)
    return field
end

# Resolves the ambiguity with `compute!(::MultiRegionField)`
Fields.compute!(field::Field{<:Any, <:Any, <:Any, <:YinYangRemap, <:MultiRegionGrids}, time = nothing) =
    throw(ArgumentError("Fields cannot be remapped from a Yin-Yang grid onto a multi-region grid."))

compute_sources_at!(source, time) = compute_at!(source, time)
compute_sources_at!(source::Tuple, time) = map(field -> compute_at!(field, time), source)

remap_operand!(field, source, remapping, ::Nothing) = remap!(field, source, remapping)
remap_operand!(field, source, remapping, component) = remap_vector_component!(field, source, remapping, component)

Fields.interpolate!(to_field::Field, from_field::YinYangField) = remap!(to_field, from_field)
