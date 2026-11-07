using Oceananigans.Fields: Fields, AbstractField, Remap, remap!, remapped_field
using Oceananigans.Grids: λnode, φnode, cpu_face_constructor_z, halo_size, interior_z_indices
using Oceananigans.ImmersedBoundaries: GridFittedBottom, PartialCellBottom

#####
##### Remapping from a Yin-Yang grid to a grid in geographic coordinates: each destination node is interpolated
##### from the panel that owns it with the stencils and weights of the halo tables, and the components of
##### vectors are rotated into the geographic basis.
#####

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

function latitude_longitude_grid(grid::ImmersedYinYangGrid; kwargs...)
    underlying_grid = latitude_longitude_grid(grid.underlying_grid; kwargs...)
    bottom_heights = map(r -> on_architecture(CPU(), getregion(grid.immersed_boundary.bottom_height, r)), (1, 2))
    _, owners = owning_panels(grid, underlying_grid, (Center(), Center()), (Center(), Center()))
    bottom_height = [bottom_heights[owner.panel][round.(Int, owner.fractional_indices)..., 1] for owner in owners]
    Nx, Ny, _ = Base.size(underlying_grid)
    return ImmersedBoundaryGrid(underlying_grid, with_bottom_height(grid.immersed_boundary, reshape(bottom_height, Nx, Ny)))
end

with_bottom_height(ib::GridFittedBottom, bottom_height) = GridFittedBottom(bottom_height, ib.immersed_condition)
with_bottom_height(ib::PartialCellBottom, bottom_height) = PartialCellBottom(bottom_height, ib.minimum_fractional_cell_height)

# The nodes `(i, j)` at `(ℓx, ℓy)` of `destination_grid` in column-major order and, for each, the panel of `grid`
# that owns it, the node's fractional indices in that panel's index space at `source_location`, and the angle
# from the panel's basis to the geographic basis
function owning_panels(grid, destination_grid, (ℓx, ℓy), source_location)
    connectivity = yin_yang_connectivity(grid)
    frame = panel_frame(yin_yang_panel_face_coordinates(connectivity.core_size, connectivity.overlap)...)
    destination = on_architecture(CPU(), destination_grid)
    Nx, Ny, _ = size(destination, (ℓx, ℓy, Center()))
    nodes = [(i, j) for j in 1:Ny for i in 1:Nx]
    owners = map(nodes) do (i, j)
        λ = mod(Float64(λnode(i, j, 1, destination, ℓx, ℓy, Center())) + 180, 360) - 180
        φ = Float64(φnode(i, j, 1, destination, ℓx, ℓy, Center()))
        owned = yin_yang_owned(λ, φ)
        λ′, φ′ = owned ? (λ, φ) : yin_yang_transform(λ, φ)
        ψ = owned ? 0.0 : - yin_yang_rotation_angle(λ, φ) # Yin's basis is the geographic basis
        fractional_indices = panel_fractional_indices(λ′, φ′, frame, source_location)
        return (; panel = owned ? 1 : 2, fractional_indices, ψ)
    end
    return nodes, owners
end

"""
    yin_yang_remapping(destination, source)

Return the tables, one per panel, that interpolate `source`, a field on a Yin-Yang grid, or each component
of the vector `source = (u, v)`, onto the nodes of `destination`, a field in geographic coordinates: each
node is interpolated from the panel that owns it. The remapping interpolates point values for diagnostics
and output: it does not conserve integrals.
"""
yin_yang_remapping(destination::AbstractField, source) =
    yin_yang_remapping(destination.grid, horizontal_location(destination), source)

yin_yang_remapping(destination_grid, destination_location, components::Tuple) =
    map(component -> yin_yang_remapping(destination_grid, destination_location, component), components)

function yin_yang_remapping(destination_grid, destination_location, source::YinYangField)
    nodes, owners = owning_panels(source.grid, destination_grid, destination_location, horizontal_location(source))
    s = yin_yang_connectivity(source.grid).stencil_width
    interior_size = size(getregion(source.grid, 1))[1:2]

    return ntuple(2) do r
        owned = findall(owner -> owner.panel == r, owners)
        fractional_indices = [owner.fractional_indices for owner in owners[owned]]
        ψ = [owner.ψ for owner in owners[owned]]
        YinYangInterpolationTable(architecture(destination_grid), eltype(source.grid), s,
                                  nodes[owned], fractional_indices, ψ, interior_size)
    end
end

const YinYangRemapping = NTuple{2, YinYangInterpolationTable}

# A field on a Yin-Yang grid is remapped from the panel that owns each destination node, and the components
# of a horizontal vector are rotated into the geographic basis
Fields.remapping(destination_grid, destination_location, source::YinYangField) =
    yin_yang_remapping(destination_grid, horizontal_remapping_location(destination_location), source)

Fields.remapping(destination_grid, destination_location, components::Tuple{YinYangField, YinYangField}) =
    yin_yang_remapping(destination_grid, horizontal_remapping_location(destination_location), components)

Fields.remapping(destination_grid, destination_location, source::AbstractField{<:Any, <:Any, <:Any, <:MultiRegionGrids}) =
    throw(ArgumentError("Only fields on Yin-Yang grids can be remapped: materialize $(summary(source)) with `Field`."))

# Resolves the ambiguity between `compute!(::RemappedField)` and the multi-region methods of `compute!`
Fields.compute!(field::Field{<:Any, <:Any, <:Any, <:Remap, <:MultiRegionGrids}, time = nothing) =
    throw(ArgumentError("Fields cannot be remapped onto a multi-region grid."))

function horizontal_remapping_location(location)
    ℓx, ℓy, _ = location
    (ℓx isa Nothing) | (ℓy isa Nothing) &&
        throw(ArgumentError("Fields reduced in x or y cannot be remapped from a Yin-Yang grid: " *
                            "remap first, then reduce."))
    return (ℓx, ℓy)
end

Fields.remapping_name(::Union{YinYangRemapping, NTuple{2, YinYangRemapping}}) = "YinYangRemapping"

Fields.validate_remapping_destination(::MultiRegionGrids) =
    throw(ArgumentError("Fields cannot be remapped onto a multi-region grid."))

"""
    remapped_field(source::YinYangField)
    remapped_field((u, v)::Tuple{YinYangField, YinYangField})

Return `source`, or the vector `(u, v)`, remapped onto the [`latitude_longitude_grid`](@ref) of its grid.
"""
Fields.remapped_field(source::YinYangField; kw...) = remapped_field(source, latitude_longitude_grid(source.grid); kw...)

Fields.remapped_field(components::Tuple{YinYangField, YinYangField}; kw...) =
    remapped_field(components, latitude_longitude_grid(first(components).grid); kw...)

#####
##### Remapping kernels
#####

"""
    remap!(destination, source::YinYangField, remapping = yin_yang_remapping(destination, source))
    remap!((uᵍ, vᵍ), (u, v), remapping = yin_yang_remapping(uᵍ, (u, v)))

Remap the scalar `source`, or the horizontal vector with panel components `(u, v)`, from a Yin-Yang grid
onto the interior nodes of `destination`, or onto the geographic components `(uᵍ, vᵍ)`, which share
a location. The source and destination fields must have the same vertical nodes. The halos of the source
need not be filled, and those of the destination are not filled. On an immersed grid, scalars are
interpolated from wet cells only, and nodes whose stencils hold no wet cell are set to zero.
"""
function Fields.remap!(destination, source::YinYangField, remapping::YinYangRemapping, ::Nothing)
    for r in 1:2
        launch_remapping!(destination, source, remapping[r], _interpolate_yin_yang_scalar!, destination.data,
                          getregion(source.data, r), getregion(source.grid, r), instantiated_location(source), remapping[r])
    end
    return nothing
end

function Fields.remap!(destination, (u, v)::Tuple{YinYangField, YinYangField}, (u_remapping, v_remapping)::NTuple{2, YinYangRemapping}, component)
    for r in 1:2
        launch_remapping!(destination, u, u_remapping[r], _interpolate_yin_yang_vector_component!, destination.data,
                          getregion(u.data, r), getregion(v.data, r), nothing, nothing,
                          (u_remapping[r], v_remapping[r]), component)
    end
    return nothing
end

# The grids share their vertical nodes, and the source holds the vertical window of the destination
function launch_remapping!(destination, source, table, kernel!, args::Vararg{Any, N}) where N
    panel = getregion(source, 1)
    Base.size(destination.grid, 3) == Base.size(panel.grid, 3) ||
        throw(ArgumentError("The destination and source grids have different vertical sizes."))
    ks = interior_z_indices(destination)
    ks ⊆ interior_z_indices(panel) ||
        throw(ArgumentError("The vertical indices $ks of the destination are not held by the source."))
    return launch_yin_yang_interpolation!(destination.grid, ks, table, kernel!, args...)
end

Fields.interpolate!(to_field::Field, from_field::YinYangField) = remap!(to_field, from_field)
