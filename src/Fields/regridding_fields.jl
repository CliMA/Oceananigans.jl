using KernelAbstractions: @kernel, @index
using KernelAbstractions.Extras.LoopInfo: @unroll

using Oceananigans.Architectures: arch_array, architecture
using Oceananigans.Operators: Δzᵃᵃᶜ

const SingleColumnGrid = AbstractGrid{<:AbstractFloat, <:Flat, <:Flat, <:Bounded}

regrid!(u, v) = regrid!(u, u.grid, v.grid, v)

function regrid!(u, target_grid, source_grid, v)
    msg = """Regridding
             $(short_show(v)) on $(short_show(source_grid))
             to $(short_show(u)) on $(short_show(target_grid))
             is not supported."""

    return throw(ArgumentError(msg))
end

#####
##### Regridding for single column grids
#####

function regrid!(u, target_grid::SingleColumnGrid, source_grid::SingleColumnGrid, v)
    arch = architecture(u)
    source_z_faces = znodes(Face, source_grid)

    event = launch!(arch, target_grid, :xy, _regrid!, u, v, target_grid, source_grid, source_z_faces)
    wait(device(arch), event)
    return nothing
end

@kernel function _regrid!(target_field, source_field, target_grid, source_grid, source_z_faces)
    i, j = @index(Global, NTuple)

    Nx_target, Ny_target, Nz_target = size(target_grid)
    Nx_source, Ny_source, Nz_source = size(source_grid)
    i_src = ifelse(Nx_target == Nx_source, i, 1)
    j_src = ifelse(Ny_target == Ny_source, j, 1)

    @unroll for k = 1:target_grid.Nz
        @inbounds target_field[i, j, k] = 0

        z₋ = znode(Center(), Center(), Face(), i, j, k,   target_grid)
        z₊ = znode(Center(), Center(), Face(), i, j, k+1, target_grid)

        # Integrate source field from z₋ to z₊
        k₋_src = searchsortedfirst(source_z_faces, z₋)
        k₊_src = searchsortedfirst(source_z_faces, z₊) - 1

        # Add contribution from all full cells in the integration range
        @unroll for k_src = k₋_src:k₊_src
            @inbounds target_field[i, j, k] += source_field[i_src, j_src, k_src] * Δzᵃᵃᶜ(i_src, j_src, k_src, source_grid)
        end

        zk₋_src = znode(Center(), Center(), Face(), i_src, j_src, k₋_src, source_grid)
        zk₊_src = znode(Center(), Center(), Face(), i_src, j_src, k₊_src+1, source_grid)

        # Add contribution to integral from fractional bottom part,
        # if that region is a part of the grid.
        @inbounds target_field[i, j, k] += source_field[i_src, j_src, k₋_src - 1] * (zk₋_src - z₋)

        # Add contribution to integral from fractional top part
        @inbounds target_field[i, j, k] += source_field[i_src, j_src, k₊_src] * (z₊ - zk₊_src)

        @inbounds target_field[i, j, k] /= Δzᵃᵃᶜ(i, j, k, target_grid)
    end
end

