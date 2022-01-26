using KernelAbstractions: @kernel, @index
using KernelAbstractions.Extras.LoopInfo: @unroll

using Oceananigans.Architectures: arch_array, architecture
using Oceananigans.Operators: Δzᵃᵃᶜ

const SingleColumnGrid = AbstractGrid{<:AbstractFloat, <:Flat, <:Flat, <:Bounded}

"""
    regrid!(a, b)

Regrid field `b` onto the grid of field `a`. 

!!! warning "Functionality limitation"
    Currently `regrid!` only regrids in the vertical ``z`` direction and works only on
    grids for which ``x`` and ``y`` dimensions are `Flat`.

Example
=======

Generate a tracer field on a vertically stretched grid and regrid it on a regular grid.

```jldoctest
using Oceananigans

Nz, Lz = 2, 1.0
topology = (Flat, Flat, Bounded)

input_grid = RectilinearGrid(size=Nz, z = [0, Lz/3, Lz], topology=topology)
input_field = CenterField(input_grid)
input_field[1, 1, 1:Nz] = [2, 3]

output_grid = RectilinearGrid(size=Nz, z=(0, Lz), topology=topology)
output_field = CenterField(output_grid)

regrid!(output_field, input_field)

output_field[1, 1, :]

# output
4-element OffsetArray(::Vector{Float64}, 0:3) with eltype Float64 with indices 0:3:
 0.0
 2.333333333333334
 3.0
 0.0
```
"""
regrid!(a, b) = regrid!(a, a.grid, b.grid, b)

function we_can_regrid(a, target_grid, source_grid, b)
    # Only 1D regridding in the vertical is supported, so check that
    #   1. source and target grid are in the same "class" and
    #   2. source and target Field have same horizontal size
    typeof(source_grid).name.wrapper === typeof(target_grid).name.wrapper &&
        size(a)[1:2] === size(b)[1:2] && return true

    return false
end

function regrid!(a, target_grid, source_grid, b)
    if we_can_regrid(a, target_grid, source_grid, b)
        arch = architecture(a)
        source_z_faces = znodes(Face, source_grid)

        event = launch!(arch, target_grid, :xy, _regrid!, a, b, target_grid, source_grid, source_z_faces)
        wait(device(arch), event)

        return a
    else
        msg = """Regridding
                 $(summary(b)) on $(summary(source_grid))
                 to $(summary(a)) on $(summary(target_grid))
                 is not supported."""

        return throw(ArgumentError(msg))
    end
end

#####
##### Regridding for single column grids
#####

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

