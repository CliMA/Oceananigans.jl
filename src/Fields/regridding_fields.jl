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

#=
# Set interior of field `c` to values of `data`
function set!(c::AbstractField, data::AbstractArray)

    arch = architecture(c)

    # Reshape `data` to the size of `c`'s interior
    d = arch_array(arch, reshape(data, size(c)))

    # Sets the interior of field `c` to values of `data`
    c .= d

end

horizontal_size(grid) = (grid.Nx, grid.Ny)
extent(grid) = (grid.Lx, grid.Ly, grid.Lz)

# Set two fields to one another... some shenanigans
#
_set_similar_fields!(c::AbstractField{Ac, G}, d::AbstractField{Ad, G}) where {Ac, Ad, G} =
    c.data .= convert(typeof(c.data), d.data)

function interp_and_set!(c1::AbstractField{A1, G1}, c2::AbstractField{A2, G2}) where {A1, A2, G1, G2}

    grid1 = c1.grid
    grid2 = c2.grid

    @assert extent(grid1) == extent(grid2) "Physical domains differ between the two fields."

    for j in 1:grid1.Ny, i in 1:grid1.Nx
        for k in 1:grid1.Nz
            @inbounds c1[i, j, k] = integral(c2[i,j,:], grid2, grid1.zF[k], grid1.zF[k+1]) / Δz(grid1, i)
        end
    end

    return nothing
end

"""
    set!(c::AbstractField{Ac, G}, d::AbstractField{Ad, G}) where {Ac, Ad, G}

Set the data of field `c` to the data of field `d`, adjusted to field `c`'s grid.

The columns are assumed to be independent and thus the fields must have the same
horizontal resolution. This implementation does not accommodate 3D grids with
dependent columns.
"""
function set!(c::AbstractField{Ac, G}, d::AbstractField{Ad, G}) where {Ac, Ad, G}

    s1 = horizontal_size(c.grid)
    s2 = horizontal_size(d.grid)
    @assert s1 == s2 "Field grids have a different number of columns."

    if s1 != (1, 1)
        @assert c.grid isa OneDimensionalEnsembleGrid && d.grid isa OneDimensionalEnsembleGrid "Field has dependent columns."
    end

    if Lz(c) == Lz(d) && Nz(c) == Nz(d)
        return _set_similar_fields!(c, d)
    else
        return interp_and_set!(c, d)
    end

end
=#
