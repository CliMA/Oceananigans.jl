using KernelAbstractions: @kernel, @index
using KernelAbstractions.Extras.LoopInfo: @unroll

using Oceananigans.Architectures: arch_array, architecture
using Oceananigans.Operators: Δzᶜᶜᶜ, Δyᶜᶜᶜ, Δxᶜᶜᶜ, Azᶜᶜᶜ
using Oceananigans.Grids: hack_sind, ξnode, ηnode, rnode

using Base: ForwardOrdering

const f = Face()
const c = Center()

"""
    regrid!(a, b)

Regrid field `b` onto the grid of field `a`. 

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
 2.333333333333333
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

function regrid_in_z!(a, target_grid, source_grid, b)
    location(a, 3) == Center || throw(ArgumentError("Can only regrid fields in z with Center z-locations."))
    arch = architecture(a)
    source_z_faces = znodes(source_grid, f)
    launch!(arch, target_grid, :xy, _regrid_in_z!, a, b, target_grid, source_grid, source_z_faces)
    
    return a
end

function regrid_in_y!(a, target_grid, source_grid, b)
    location(a, 2) == Center || throw(ArgumentError("Can only regrid fields in y with Center y-locations."))
    arch = architecture(a)
    source_y_faces = nodes(source_grid, c, f, c)[2]
    Nx_source_faces = size(source_grid, (Face, Center, Center), 1)
    launch!(arch, target_grid, :xz, _regrid_in_y!, a, b, target_grid, source_grid, source_y_faces, Nx_source_faces)
    return a
end

function regrid_in_x!(a, target_grid, source_grid, b)
    location(a, 1) == Center || throw(ArgumentError("Can only regrid fields in x with Center x-locations."))
    arch = architecture(a)
    source_x_faces = nodes(source_grid, f, c, c)[1]
    Ny_source_faces = size(source_grid, (Center, Face, Center), 2)
    launch!(arch, target_grid, :yz, _regrid_in_x!, a, b, target_grid, source_grid, source_x_faces, Ny_source_faces)
    return a
end

regrid_in_x!(a, b) = regrid_in_x!(a, a.grid, b.grid, b)
regrid_in_y!(a, b) = regrid_in_y!(a, a.grid, b.grid, b)
regrid_in_z!(a, b) = regrid_in_z!(a, a.grid, b.grid, b)

function regrid!(a, target_grid, source_grid, b)
    arch = architecture(a)

    if we_can_regrid_in_z(a, target_grid, source_grid, b)
        return regrid_in_z!(a, target_grid, source_grid, b)
    elseif we_can_regrid_in_y(a, target_grid, source_grid, b)
        return regrid_in_y!(a, target_grid, source_grid, b)
    elseif we_can_regrid_in_x(a, target_grid, source_grid, b)
        return regrid_in_x!(a, target_grid, source_grid, b)
    else
        msg = """Regridding
                 $(summary(b)) on $(summary(source_grid))
                 to $(summary(a)) on $(summary(target_grid))
                 is not supported."""

        return throw(ArgumentError(msg))
    end
end

#####
##### Regridding for all grids
#####

@kernel function _regrid_in_z!(target_field, source_field, target_grid, source_grid, source_z_faces)
    i, j = @index(Global, NTuple)

    Nx_target, Ny_target, Nz_target = size(target_grid)
    Nx_source, Ny_source, Nz_source = size(source_grid)
    i_src = ifelse(Nx_target == Nx_source, i, 1)
    j_src = ifelse(Ny_target == Ny_source, j, 1)

    fo = ForwardOrdering()

    @inbounds @unroll for k = 1:target_grid.Nz
        target_field[i, j, k] = 0

        z₋ = znode(i, j, k,   target_grid, c, c, f)
        z₊ = znode(i, j, k+1, target_grid, c, c, f)

        # Integrate source field from z₋ to z₊
        k₋_src = searchsortedfirst(source_z_faces, z₋, 1, Nz_source+1, fo)
        k₊_src = searchsortedfirst(source_z_faces, z₊, 1, Nz_source+1, fo) - 1

        if k₊_src < k₋_src
            # If the "last" face on the source grid is equal to or left
            # of the "first" face on the source grid, the target cell
            # lies entirely within the source cell j₊_src (ie, we are _refining_
            # rather than coarse graining). In this case our job is easy:
            # the target cell concentration is equal to the source concentration.
            target_field[i, j, k] = source_field[i_src, j_src, k₊_src]
        else
            # Add contribution from all full cells in the integration range
            @unroll for k_src = k₋_src:k₊_src-1
                target_field[i, j, k] += source_field[i_src, j_src, k_src] * Δzᶜᶜᶜ(i_src, j_src, k_src, source_grid)
            end

            zk₋_src = znode(i_src, j_src, k₋_src, source_grid, c, c, f)
            zk₊_src = znode(i_src, j_src, k₊_src, source_grid, c, c, f) 

            # Add contribution to integral from fractional left part of the source field,
            # if that region is a part of the grid.
            if k₋_src > 1
                target_field[i, j, k] += source_field[i_src, j_src, k₋_src - 1] * (zk₋_src - z₋)
            end

            # Add contribution to integral from fractional right part of the source field, if that
            # region is part of the grid.
            if k₊_src < source_grid.Nz+1
                target_field[i, j, k] += source_field[i_src, j_src, k₊_src] * (z₊ - zk₊_src)
            end

            target_field[i, j, k] /= Δzᶜᶜᶜ(i, j, k, target_grid)
        end
    end
end

@kernel function _regrid_in_y!(target_field, source_field, target_grid, source_grid, source_y_faces, Nx_source_faces)
    i, k = @index(Global, NTuple)

    Nx_target, Ny_target, Nz_target = size(target_grid)
    Nx_source, Ny_source, Nz_source = size(source_grid)
    i_src = ifelse(Nx_target == Nx_source, i, 1)
    k_src = ifelse(Nz_target == Nz_source, k, 1)

    i⁺_src = min(Nx_source_faces, i_src + 1)

    fo = ForwardOrdering()

    @inbounds @unroll for j = 1:target_grid.Ny
        target_field[i, j, k] = 0

        y₋ = ηnode(i, j,   k, target_grid, c, f, c)
        y₊ = ηnode(i, j+1, k, target_grid, c, f, c)

        # Integrate source field from y₋ to y₊
        j₋_src = searchsortedfirst(source_y_faces, y₋, 1, Ny_source+1, fo)
        j₊_src = searchsortedfirst(source_y_faces, y₊, 1, Ny_source+1, fo) - 1

        if j₊_src < j₋_src
            # If the "last" face on the source grid is equal to or left
            # of the "first" face on the source grid, the target cell
            # lies entirely within the source cell j₊_src (ie, we are _refining_
            # rather than coarse graining). In this case our job is easy:
            # the target cell concentration is equal to the source concentration.
            target_field[i, j, k] = source_field[i_src, j₊_src, k_src]
        else
            # Add contribution from all full cells in the integration range
            @unroll for j_src = j₋_src:j₊_src-1
                target_field[i, j, k] += source_field[i_src, j_src, k_src] * Azᶜᶜᶜ(i_src, j_src, k_src, source_grid)
            end

            yj₋_src = ηnode(i_src, j₋_src, k_src, source_grid, c, f, c)
            yj₊_src = ηnode(i_src, j₊_src, k_src, source_grid, c, f, c)

            # Add contribution to integral from fractional left part,
            # if that region is a part of the grid.
            # We approximate the volume of the fractional part by linearly interpolating the cell volume.
            if j₋_src > 1
                j_left = j₋_src - 1

                ξ₁ = ξnode(i_src,  j_left, k_src, source_grid, f, c, c)
                ξ₂ = ξnode(i⁺_src, j_left, k_src, source_grid, f, c, c)
                Az_left = fractional_horizontal_area(source_grid, ξ₁, ξ₂, y₋, yj₋_src)

                target_field[i, j, k] += source_field[i_src, j_left, k_src] * Az_left
            end

            # Similar to above, add contribution to integral from fractional right part.
            if j₊_src < source_grid.Ny+1
                j_right = j₊_src

                ξ₁ = ξnode(i_src,  j_right, k_src, source_grid, f, c, c)
                ξ₂ = ξnode(i⁺_src, j_right, k_src, source_grid, f, c, c)
                Az_right = fractional_horizontal_area(source_grid, ξ₁, ξ₂, yj₊_src, y₊)

                target_field[i, j, k] += source_field[i_src, j_right, k_src] * Az_right
            end

            target_field[i, j, k] /= Azᶜᶜᶜ(i, j, k, target_grid)
        end
    end
end

@kernel function _regrid_in_x!(target_field, source_field, target_grid, source_grid, source_x_faces, Ny_source_faces)
    j, k = @index(Global, NTuple)

    Nx_target, Ny_target, Nz_target = size(target_grid)
    Nx_source, Ny_source, Nz_source = size(source_grid)
    j_src = ifelse(Ny_target == Ny_source, j, 1)
    k_src = ifelse(Nz_target == Nz_source, k, 1)

    j⁺_src = min(Ny_source_faces, j_src + 1)

    fo = ForwardOrdering()

    @inbounds @unroll for i = 1:target_grid.Nx
        target_field[i, j, k] = 0

        # Integrate source field from ξ₋ to ξ₊
        ξ₋ = ξnode(i,   j, k, target_grid, f, c, c)
        ξ₊ = ξnode(i+1, j, k, target_grid, f, c, c)

        # The first face on the source grid that appears inside the target cell
        i₋_src = searchsortedfirst(source_x_faces, ξ₋, 1, Nx_source+1, fo)

        # The last face on the source grid that appears inside the target cell
        i₊_src = searchsortedfirst(source_x_faces, ξ₊, 1, Nx_source+1, fo) - 1

        if i₊_src < i₋_src
            # If the "last" face on the source grid is equal to or left
            # of the "first" face on the source grid, the target cell
            # lies entirely within the source cell i₊_src (ie, we are _refining_
            # rather than coarse graining). In this case our job is easy:
            # the target cell concentration is equal to the source concentration.
            target_field[i, j, k] = source_field[i₊_src, j_src, k_src]
        else
            # Otherwise, our job is a little bit harder and we have to carefully, conservatively
            # sum up all the contributions from the source field to the target cell.
            
            # First we add up all the contributions from all source cells that lie entirely within the target cell.
            @unroll for i_src = i₋_src:i₊_src-1
                target_field[i, j, k] += source_field[i_src, j_src, k_src] * Azᶜᶜᶜ(i_src, j_src, k_src, source_grid)
            end
    
            # Next, we add contributions from the "fractional" source cells on the right
            # and left of the target cell.
            ξi₋_src = ξnode(i₋_src, j_src, k_src, source_grid, f, c, c)
            ξi₊_src = ξnode(i₊_src, j_src, k_src, source_grid, f, c, c)
    
            # Add contribution to integral from fractional left part,
            # if that region is a part of the grid.
            # We approximate the volume of the fractional part by linearly interpolating the cell volume.
            if i₋_src > 1
                i_left = i₋_src - 1
                
                η₁ = ηnode(i_left, j_src,  k_src, source_grid, c, f, c)
                η₂ = ηnode(i_left, j⁺_src, k_src, source_grid, c, f, c)
                Az_left = fractional_horizontal_area(source_grid, ξ₋, ξi₋_src, η₁, η₂)

                target_field[i, j, k] += source_field[i_left, j_src, k_src] * Az_left
            end

                
            # Similar to above, add contribution to integral from fractional right part.
            if i₊_src < source_grid.Nx+1
                i_right = i₊_src

                η₁ = ηnode(i_right, j_src,  k_src, source_grid, c, f, c)
                η₂ = ηnode(i_right, j⁺_src, k_src, source_grid, c, f, c)
                Az_right = fractional_horizontal_area(source_grid, ξi₊_src, ξ₊, η₁, η₂)

                target_field[i, j, k] += source_field[i_right, j_src, k_src] * Az_right
            end
    
            target_field[i, j, k] /= Azᶜᶜᶜ(i, j, k, target_grid)
        end
    end
end

@inline fractional_horizontal_area(grid::RectilinearGrid, x₁, x₂, y₁, y₂) = (x₂ - x₁) * (y₂ - y₁)
@inline fractional_horizontal_area(grid::RectilinearGrid{<:Any, <:Flat}, x₁, x₂, y₁, y₂) = y₂ - y₁
@inline fractional_horizontal_area(grid::RectilinearGrid{<:Any, <:Any, <:Flat}, x₁, x₂, y₁, y₂) = (x₂ - x₁)

@inline function fractional_horizontal_area(grid::LatitudeLongitudeGrid, λ₁, λ₂, φ₁, φ₂)
    Δλ = λ₂ - λ₁
    return grid.radius^2 * deg2rad(Δλ) * (hack_sind(φ₂) - hack_sind(φ₁))
end

@inline fractional_horizontal_area(grid::LatitudeLongitudeGrid{<:Any, <:Flat}, λ₁, λ₂, φ₁, φ₂) = grid.radius^2 * (hack_sind(φ₂) - hack_sind(φ₁))
@inline fractional_horizontal_area(grid::LatitudeLongitudeGrid{<:Any, <:Any, <:Flat}, λ₁, λ₂, φ₁, φ₂) = grid.radius^2 * deg2rad(λ₂ - λ₁)

