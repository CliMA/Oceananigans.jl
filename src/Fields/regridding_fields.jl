using KernelAbstractions: @kernel, @index
using KernelAbstractions.Extras.LoopInfo: @unroll

using Oceananigans.Architectures: arch_array, architecture
using Oceananigans.Operators: Δzᶜᶜᶜ, Δyᶜᶜᶜ, Δxᶜᶜᶜ, Azᶜᶜᶜ

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

function we_can_regrid_in_x(a, target_grid, source_grid, b)
    # Check that
    #   1. source and target grid are in the same "class" and
    #   2. source and target Field have same yz size
    typeof(source_grid).name.wrapper === typeof(target_grid).name.wrapper &&
        size(a)[[2, 3]] === size(b)[[2, 3]] && return true

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

        if false #k₊_src <= k₋_src
            # If the "last" face on the source grid is equal to or left
            # of the "first" face on the source grid, the target cell
            # lies entirely within the source cell j₊_src (ie, we are _refining_
            # rather than coarse graining). In this case our job is easy:
            # the target cell concentration is equal to the source concentration.
            @inbounds target_field[i, j, k] = source_field[i_src, j_src, k₊_src]
        else
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

        if j₊_src <= j₋_src
            # If the "last" face on the source grid is equal to or left
            # of the "first" face on the source grid, the target cell
            # lies entirely within the source cell j₊_src (ie, we are _refining_
            # rather than coarse graining). In this case our job is easy:
            # the target cell concentration is equal to the source concentration.
            @inbounds target_field[i, j, k] = source_field[i_src, j₊_src, k_src]
        else
            # Add contribution from all full cells in the integration range
            @unroll for j_src = j₋_src:j₊_src
                @inbounds target_field[i, j, k] += source_field[i_src, j_src, k_src] * Azᶜᶜᶜ(i_src, j_src, k_src, source_grid)
            end

            yj₋_src = ynode(Center(), Face(), Center(), i_src, j₋_src,   k_src, source_grid)
            yj₊_src = ynode(Center(), Face(), Center(), i_src, j₊_src+1, k_src, source_grid)

            # Add contribution to integral from fractional left part,
            # if that region is a part of the grid.
            # We approximate the volume of the fractional part by linearly interpolating the cell volume.
            j_left = max(1, j₋_src - 1)
            ϵ_left = (yj₋_src - y₋) / Δyᶜᶜᶜ(i_src, j_left, k_src, source_grid) 
            @inbounds target_field[i, j, k] += source_field[i_src, j_left, k_src] * ϵ_left * Azᶜᶜᶜ(i_src, j_left, k_src, source_grid)

            # Similar to above, add contribution to integral from fractional right part.
            j_right = min(source_grid.Ny, j₊_src)
            ϵ_right = (y₊ - yj₊_src) / Δyᶜᶜᶜ(i_src, j_right, k_src, source_grid) 
            @inbounds target_field[i, j, k] += source_field[i_src, j_right, k_src] * ϵ_right * Azᶜᶜᶜ(i_src, j_right, k_src, source_grid)

            @inbounds target_field[i, j, k] /= Azᶜᶜᶜ(i, j, k, target_grid)
        end
    end
end

@kernel function _regrid_in_x!(target_field, source_field, target_grid, source_grid, source_x_faces)
    j, k = @index(Global, NTuple)

    Nx_target, Ny_target, Nz_target = size(target_grid)
    Nx_source, Ny_source, Nz_source = size(source_grid)
    j_src = ifelse(Ny_target == Ny_source, j, 1)
    k_src = ifelse(Nz_target == Nz_source, k, 1)

    @unroll for i = 1:target_grid.Nx
        @inbounds target_field[i, j, k] = 0

        # Integrate source field from x₋ to x₊
        x₋ = xnode(Face(), Center(), Center(), i,   j, k, target_grid)
        x₊ = xnode(Face(), Center(), Center(), i+1, j, k, target_grid)

        # The first face on the source grid that appears inside the target cell
        i₋_src = searchsortedfirst(source_x_faces, x₋)

        # The last face on the source grid that appears inside the target cell
        i₊_src = searchsortedfirst(source_x_faces, x₊) - 1

        if i₊_src <= i₋_src
            # If the "last" face on the source grid is equal to or left
            # of the "first" face on the source grid, the target cell
            # lies entirely within the source cell i₊_src (ie, we are _refining_
            # rather than coarse graining). In this case our job is easy:
            # the target cell concentration is equal to the source concentration.
            @inbounds target_field[i, j, k] = source_field[i₊_src, j_src, k_src]
        else
            # Otherwise, our job is a little bit harder and we have to carefully, conservatively
            # sum up all the contributions from the source field to the target cell.
            
            # First we add up all the contributions from all source cells that lie entirely within the target cell.
            @unroll for i_src = i₋_src:i₊_src
                @inbounds target_field[i, j, k] += source_field[i_src, j_src, k_src] * Azᶜᶜᶜ(i_src, j_src, k_src, source_grid)
            end
    
            # Next, we add contributions from the "fractional" source cells on the right
            # and left of the target cell.
            xi₋_src = xnode(Face(), Center(), Center(), i₋_src,   j_src, k_src, source_grid)
            xi₊_src = xnode(Face(), Center(), Center(), i₊_src+1, j_src, k_src, source_grid)
    
            # Add contribution to integral from fractional left part,
            # if that region is a part of the grid.
            # We approximate the volume of the fractional part by linearly interpolating the cell volume.
            i_left = max(1, i₋_src - 1)
            ϵ_left = (xi₋_src - x₋) / Δxᶜᶜᶜ(i_left, j_src, k_src, target_grid) 
            ϵ_left = max(zero(source_grid), ϵ_left)
            @inbounds target_field[i, j, k] += source_field[i_left, j_src, k_src] * ϵ_left * Azᶜᶜᶜ(i_left, j_src, k_src, source_grid)
    
            # Similar to above, add contribution to integral from fractional right part.
            i_right = min(source_grid.Nx, i₊_src)
            ϵ_right = (x₊ - xi₊_src) / Δxᶜᶜᶜ(i_right, j_src, k_src, target_grid) 
            ϵ_right = max(zero(source_grid), ϵ_right)
            @inbounds target_field[i, j, k] += source_field[i_right, j_src, k_src] * ϵ_right * Azᶜᶜᶜ(i_right, j_src, k_src, source_grid)
    
            @inbounds target_field[i, j, k] /= Azᶜᶜᶜ(i, j, k, target_grid)
        end
    end
end

