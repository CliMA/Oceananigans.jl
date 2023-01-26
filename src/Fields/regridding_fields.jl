using KernelAbstractions: @kernel, @index
using KernelAbstractions.Extras.LoopInfo: @unroll

using Oceananigans.Architectures: arch_array, architecture
using Oceananigans.Operators: Δzᶜᶜᶜ, Δyᶜᶜᶜ

const SingleColumnGrid = AbstractGrid{<:AbstractFloat, <:Flat, <:Flat, <:Bounded}

"""
    regrid!(a, b)

Regrid field `b` onto the grid of field `a`. 

!!! warning "Functionality limitation"
    Currently `regrid!` only regrids in the vertical ``z`` direction and works only on
    fields that have data only in ``z`` direction.

Example
=======

Generate a tracer field on a vertically stretched grid and regrid it on a regular grid.

```jldoctest
using Oceananigans

Nz, Lz = 2, 1.0
topology = (Flat, Flat, Bounded)

input_grid = RectilinearGrid(size=Nz, z = [0, Lz/3, Lz], topology=topology, halo=1)
input_field = CenterField(input_grid)
input_field[1, 1, 1:Nz] = [2, 3]

output_grid = RectilinearGrid(size=Nz, z=(0, Lz), topology=topology, halo=1)
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

function we_can_regrid_in_z(a, target_grid, source_grid, b)
    # Check that
    #   1. source and target grid are in the same "class" and
    #   2. source and target Field have same horizontal size
    typeof(source_grid).name.wrapper === typeof(target_grid).name.wrapper &&
        size(a)[1:2] === size(b)[1:2] && return true

    return false
end

function we_can_regrid_in_y(a, target_grid, source_grid, b)
    # Check that
    #   1. source and target grid are in the same "class" and
    #   2. source and target Field have same xz size
    typeof(source_grid).name.wrapper === typeof(target_grid).name.wrapper &&
        size(a)[[1, 3]] === size(b)[[1, 3]] && return true

    return false
end


function regrid!(a, target_grid, source_grid, b)
    arch = architecture(a)

    if we_can_regrid_in_z(a, target_grid, source_grid, b)
        source_z_faces = znodes(Face, source_grid)
        event = launch!(arch, target_grid, :xy, _regrid_in_z!, a, b, target_grid, source_grid, source_z_faces)
        wait(device(arch), event)
        return a
    elseif we_can_regrid_in_y(a, target_grid, source_grid, b)
        source_y_faces = ynodes(Face, source_grid)
        event = launch!(arch, target_grid, :xz, _regrid_in_y!, a, b, target_grid, source_grid, source_y_faces)
        wait(device(arch), event)
        return a
    elseif we_can_regrid_in_x(a, target_grid, source_grid, b)
        source_x_faces = xnodes(Face, source_grid)
        event = launch!(arch, target_grid, :yz, _regrid_in_x!, a, b, target_grid, source_grid, source_x_faces)
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

@kernel function _regrid_in_z!(target_field, source_field, target_grid, source_grid, source_z_faces)
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
            @inbounds target_field[i, j, k] += source_field[i_src, j_src, k_src] * Δzᶜᶜᶜ(i_src, j_src, k_src, source_grid)
        end

        zk₋_src = znode(Center(), Center(), Face(), i_src, j_src, k₋_src, source_grid)
        zk₊_src = znode(Center(), Center(), Face(), i_src, j_src, k₊_src+1, source_grid)

        # Add contribution to integral from fractional bottom part,
        # if that region is a part of the grid.
        @inbounds target_field[i, j, k] += source_field[i_src, j_src, max(1, k₋_src - 1)] * (zk₋_src - z₋)

        # Add contribution to integral from fractional top part
        @inbounds target_field[i, j, k] += source_field[i_src, j_src, min(source_grid.Nz, k₊_src)] * (z₊ - zk₊_src)

        @inbounds target_field[i, j, k] /= Δzᶜᶜᶜ(i, j, k, target_grid)
    end
end

@kernel function _regrid_in_y!(target_field, source_field, target_grid, source_grid, source_y_faces)
    i, k = @index(Global, NTuple)

    Nx_target, Ny_target, Nz_target = size(target_grid)
    Nx_source, Ny_source, Nz_source = size(source_grid)
    i_src = ifelse(Nx_target == Nx_source, i, 1)
    k_src = ifelse(Nz_target == Nz_source, k, 1)

    @unroll for j = 1:target_grid.Ny
        @inbounds target_field[i, j, k] = 0

        y₋ = ynode(Center(), Face(), Center(), i, j,   k, target_grid)
        y₊ = ynode(Center(), Face(), Center(), i, j+1, k, target_grid)

        # Integrate source field from y₋ to y₊
        j₋_src = searchsortedfirst(source_y_faces, y₋)
        j₊_src = searchsortedfirst(source_y_faces, y₊) - 1

        # Add contribution from all full cells in the integration range
        @unroll for j_src = j₋_src:j₊_src
            @inbounds target_field[i, j, k] += source_field[i_src, j_src, k_src] * Vᶜᶜᶜ(i_src, j_src, k_src, source_grid)
        end

        yj₋_src = ynode(Center(), Face(), Center(), i_src, j₋_src,   k_src, source_grid)
        yj₊_src = ynode(Center(), Face(), Center(), i_src, j₊_src+1, k_src, source_grid)

        # Add contribution to integral from fractional left part,
        # if that region is a part of the grid.
        # We approximate the volume of the fractional part by linearly interpolating the cell volume.
        j_left = max(1, j₋_src - 1)
        ϵ_left = (yj₋_src - y₋) / Δyᶜᶜᶜ(i_src, j_left, k_src) 
        @inbounds target_field[i, j, k] += source_field[i_src, j_left, k_src] * (yj₋_src - y₋) * ϵ_left * Vᶜᶜᶜ(i_src, j_left, k_src, source_grid)

        # Similar to above, add contribution to integral from fractional right part.
        j_right = min(source_grid.Ny, j₊_src)
        ϵ_right = (y₊ - yj₊_src) / Δyᶜᶜᶜ(i_src, j_right, k_src) 
        @inbounds target_field[i, j, k] += source_field[i_src, j_right, k_src] * ϵ_right * Vᶜᶜᶜ(i_src, j_right, k_src, source_grid)

        @inbounds target_field[i, j, k] /= Vᶜᶜᶜ(i, j, k, target_grid)
    end
end

@kernel function _regrid_in_x!(target_field, source_field, target_grid, source_grid, source_y_faces)
    j, k = @index(Global, NTuple)

    Nx_target, Ny_target, Nz_target = size(target_grid)
    Nx_source, Ny_source, Nz_source = size(source_grid)
    j_src = ifelse(Ny_target == Ny_source, j, 1)
    k_src = ifelse(Nz_target == Nz_source, k, 1)

    @unroll for i = 1:target_grid.Nx
        @inbounds target_field[i, j, k] = 0

        x₋ = ynode(Face(), Center(), Center(), i,   j, k, target_grid)
        x₊ = ynode(Face(), Center(), Center(), i+1, j, k, target_grid)

        # Integrate source field from x₋ to x₊
        i₋_src = searchsortedfirst(source_x_faces, x₋)
        i₊_src = searchsortedfirst(source_x_faces, x₊) - 1

        # Add contribution from all full cells in the integration range
        @unroll for i_src = i₋_src:i₊_src
            @inbounds target_field[i, j, k] += source_field[i_src, j_src, k_src] * Vᶜᶜᶜ(i_src, j_src, k_src, source_grid)
        end

        xi₋_src = ynode(Face(), Center(), Center(), i₋_src,   j_src, k_src, source_grid)
        xi₊_src = ynode(Face(), Center(), Center(), i₊_src+1, j_src, k_src, source_grid)

        # Add contribution to integral from fractional left part,
        # if that region is a part of the grid.
        # We approximate the volume of the fractional part by linearly interpolating the cell volume.
        i_left = max(1, i₋_src - 1)
        ϵ_left = (xi₋_src - x₋) / Δxᶜᶜᶜ(i_left, j_src, k_src) 
        @inbounds target_field[i, j, k] += source_field[i_left, j_src, k_src] * (xi₋_src - xi) * ϵ_left * Vᶜᶜᶜ(i_left, j_src, k_src, source_grid)

        # Similar to above, add contribution to integral from fractional right part.
        i_right = min(source_grid.Nx, i₊_src)
        ϵ_right = (x₊ - xi₊_src) / Δxᶜᶜᶜ(i_right, j_src, k_src) 
        @inbounds target_field[i, j, k] += source_field[i_right, j_src, k_src] * ϵ_right * Vᶜᶜᶜ(i_right, j_src, k_src, source_grid)

        @inbounds target_field[i, j, k] /= Vᶜᶜᶜ(i, j, k, target_grid)
    end
end

