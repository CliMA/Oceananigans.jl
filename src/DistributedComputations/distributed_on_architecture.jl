using CUDA: CuArray
using OffsetArrays
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
                      old_grid.Δzᵃᵃᶠ, old_grid.Δzᵃᵃᶜ, old_grid.zᵃᵃᶠ,  old_grid.zᵃᵃᶜ,
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
                      old_grid.Δzᵃᵃᶠ, old_grid.Δzᵃᵃᶜ, old_grid.zᵃᵃᶠ, old_grid.zᵃᵃᶜ)

    new_properties = Tuple(on_architecture(child_arch, p) for p in old_properties)

    TX, TY, TZ = topology(old_grid)

    return RectilinearGrid{TX, TY, TZ}(new_arch,
                                       old_grid.Nx, old_grid.Ny, old_grid.Nz,
                                       old_grid.Hx, old_grid.Hy, old_grid.Hz,
                                       old_grid.Lx, old_grid.Ly, old_grid.Lz,
                                       new_properties...)
end
