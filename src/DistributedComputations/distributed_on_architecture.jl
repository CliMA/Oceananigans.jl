import Oceananigans.Architectures: on_architecture

# We do not support switching from distributed and serial through `on_architecture`.
# We only support moving a type from CPU to GPU and the other way around
# TODO: support changing the number of ranks / the partitioning?

# Disambiguation for methods defined in `src/Architectures.jl`
DisambiguationTypes = Union{Array,
                            CuArray,
                            BitArray,
                            SubArray{<:Any, <:Any, <:CuArray},
                            SubArray{<:Any, <:Any, <:Array},
                            OffsetArray,
                            Tuple,
                            NamedTuple}

on_architecture(arch::Distributed, a::DisambiguationTypes) = on_architecture(child_architecture(arch), a)

function on_architecture(new_arch::Distributed, old_grid::LatitudeLongitudeGrid)
    child_arch = child_architecture(new_arch)
    old_properties = (old_grid.Δλᶠᵃᵃ, old_grid.Δλᶜᵃᵃ, old_grid.λᶠᵃᵃ,  old_grid.λᶜᵃᵃ,
                      old_grid.Δφᵃᶠᵃ, old_grid.Δφᵃᶜᵃ, old_grid.φᵃᶠᵃ,  old_grid.φᵃᶜᵃ,
                      old_grid.z,
                      old_grid.Δxᶠᶜᵃ, old_grid.Δxᶜᶠᵃ, old_grid.Δxᶠᶠᵃ, old_grid.Δxᶜᶜᵃ,
                      old_grid.Δyᶠᶜᵃ, old_grid.Δyᶜᶠᵃ,
                      old_grid.Azᶠᶜᵃ, old_grid.Azᶜᶠᵃ, old_grid.Azᶠᶠᵃ, old_grid.Azᶜᶜᵃ)

    new_properties = Tuple(on_architecture(child_arch, p) for p in old_properties)

    TX, TY, TZ = topology(old_grid)

    return LatitudeLongitudeGrid{TX, TY, TZ}(new_arch,
                                             old_grid.Nx, old_grid.Ny, old_grid.Nz,
                                             old_grid.Hx, old_grid.Hy, old_grid.Hz,
                                             old_grid.Lx, old_grid.Ly, old_grid.Lz,
                                             new_properties...,
                                             old_grid.radius)
end

function on_architecture(new_arch::Distributed, old_grid::RectilinearGrid)
    child_arch = child_architecture(new_arch)
    old_properties = (old_grid.Δxᶠᵃᵃ, old_grid.Δxᶜᵃᵃ, old_grid.xᶠᵃᵃ, old_grid.xᶜᵃᵃ,
                      old_grid.Δyᵃᶠᵃ, old_grid.Δyᵃᶜᵃ, old_grid.yᵃᶠᵃ, old_grid.yᵃᶜᵃ,
                      old_grid.z)

    new_properties = Tuple(on_architecture(child_arch, p) for p in old_properties)

    TX, TY, TZ = topology(old_grid)

    return RectilinearGrid{TX, TY, TZ}(new_arch,
                                       old_grid.Nx, old_grid.Ny, old_grid.Nz,
                                       old_grid.Hx, old_grid.Hy, old_grid.Hz,
                                       old_grid.Lx, old_grid.Ly, old_grid.Lz,
                                       new_properties...)
end

function on_architecture(new_arch::Distributed, old_grid::OrthogonalSphericalShellGrid)
    child_arch = child_architecture(new_arch)

    coordinates = (:λᶜᶜᵃ, :λᶠᶜᵃ, :λᶜᶠᵃ, :λᶠᶠᵃ, :φᶜᶜᵃ, :φᶠᶜᵃ, :φᶜᶠᵃ, :φᶠᶠᵃ, :z)
    grid_spacings = (:Δxᶜᶜᵃ, :Δxᶠᶜᵃ, :Δxᶜᶠᵃ, :Δxᶠᶠᵃ, :Δyᶜᶜᵃ, :Δyᶠᶜᵃ, :Δyᶜᶠᵃ, :Δyᶠᶠᵃ)
    horizontal_areas = (:Azᶜᶜᵃ, :Azᶠᶜᵃ, :Azᶜᶠᵃ, :Azᶠᶠᵃ)

    coordinate_data = Tuple(on_architecture(child_arch, getproperty(old_grid, name)) for name in coordinates)
    grid_spacing_data = Tuple(on_architecture(child_arch, getproperty(old_grid, name)) for name in grid_spacings)
    horizontal_area_data = Tuple(on_architecture(child_arch, getproperty(old_grid, name)) for name in horizontal_areas)

    TX, TY, TZ = topology(old_grid)

    return OrthogonalSphericalShellGrid{TX, TY, TZ}(new_arch,
                                                    old_grid.Nx, old_grid.Ny, old_grid.Nz,
                                                    old_grid.Hx, old_grid.Hy, old_grid.Hz,
                                                    old_grid.Lz,
                                                    coordinate_data...,
                                                    grid_spacing_data...,
                                                    horizontal_area_data...,
                                                    old_grid.radius,
                                                    old_grid.conformal_mapping)
end