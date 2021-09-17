using KernelAbstractions: @kernel, @index
using KernelAbstractions.Extras.LoopInfo: @unroll

using Oceananigans.Architectures: arch_array, architecture
using Oceananigans.Operators: Δzᵃᵃᶜ

const SingleColumnGrid = AbstractGrid{<:AbstractFloat, <:Flat, <:Flat, <:Bounded}

function set_by_regridding!(u, target_grid, source_grid, v)
    msg = """Using `set!` to regrid
             $(short_show(v)) on $(short_show(source_grid))
             to $(short_show(u)) on $(short_show(target_grid))
             is not supported."""

    return throw(ArgumentError(msg))
end

#####
##### Regridding for single column grids
#####

function set_by_regridding!(u, target_grid::SingleColumnGrid, source_grid::SingleColumnGrid, v)
    arch = architecture(u)
    source_z_faces = znodes(Face, source_grid)

    event = launch!(arch, target_grid, :xy, _set_by_regridding!, u, v, target_grid, source_grid, source_z_faces)
    wait(device(arch), event)
    return nothing
end

"""
   target  source
    --- kt=4    --- ks = 6
                 x
                --- ks = 5
     x           x
                --- ks = 4
    --- kt=3     x 
                --- ks = 3
     x           x k=2
                --- ks = 2
    --- kt=2     x k=1
                --- ks = 1

"""
@kernel function _set_by_regridding!(target_field, source_field, target_grid, source_grid, source_z_faces)
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

function integrate_z(c::AbstractField{<:Any, <:Any, Center}, grid::SingleColumnGrid, z₋, z₊=0)

    Nx, Ny, Nz = size(c)
    z_faces = znodes(Face, grid)

    # Check some stuff
    @assert z₊ > z_faces[1] "Integration region lies outside the domain."
    @assert z₊ > z₋ "Invalid integration range: upper limit greater than lower limit."

    # Find region bounded by the face ≤ z₊ and the face ≤ z₁
    k₁ = searchsortedfirst(z_faces, z₋) - 1
    k₂ = searchsortedfirst(z_faces, z₊) - 1

    arch = architecture(c)
    integral = zeros(arch, grid, Nx, Ny)

    if k₂ ≠ k₁
        # Calculate interior integral, recalling that the
        # top interior cell has index k₂-2.
        launch!(arch, grid, :xy, multiple_cell_integral!, integral, c, grid, (k₁+1, k₂-1), (z₋, z₊))
    else
        launch!(arch, grid, :xy, single_cell_integral!, integral, c, grid, (k₁+1, k₂-1), (z₋, z₊))
    end

    return cell_integral
end

