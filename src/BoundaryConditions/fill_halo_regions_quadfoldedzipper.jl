using Oceananigans.Grids: Center, Face, octahealpix_halo_source_ring_index

const QuadFoldedScalarLikeLocation = Union{Tuple{<:Center, <:Center, <:Any},
                                           Tuple{<:Face, <:Face, <:Any}}
const QuadFoldedVectorLocation = Union{Tuple{<:Face, <:Center, <:Any},
                                       Tuple{<:Center, <:Face, <:Any}}

@inline function fill_quadfolded_west_halos!(j, k, grid, sign, c)
    for i in 1:grid.Hx
        source_ring = octahealpix_halo_source_ring_index(1 - i, j, grid.Nx, grid.Ny, grid.connectivity)
        source_i = grid.connectivity.ring_to_i[source_ring]
        source_j = grid.connectivity.ring_to_j[source_ring]
        @inbounds c[1 - i, j, k] = sign * c[source_i, source_j, k]
    end

    return nothing
end

@inline function fill_quadfolded_east_halos!(j, k, grid, sign, c)
    for i in 1:grid.Hx
        source_ring = octahealpix_halo_source_ring_index(grid.Nx + i, j, grid.Nx, grid.Ny, grid.connectivity)
        source_i = grid.connectivity.ring_to_i[source_ring]
        source_j = grid.connectivity.ring_to_j[source_ring]
        @inbounds c[grid.Nx + i, j, k] = sign * c[source_i, source_j, k]
    end

    return nothing
end

@inline function fill_quadfolded_south_halos!(i, k, grid, sign, c)
    for j in 1:grid.Hy
        source_ring = octahealpix_halo_source_ring_index(i, 1 - j, grid.Nx, grid.Ny, grid.connectivity)
        source_i = grid.connectivity.ring_to_i[source_ring]
        source_j = grid.connectivity.ring_to_j[source_ring]
        @inbounds c[i, 1 - j, k] = sign * c[source_i, source_j, k]
    end

    return nothing
end

@inline function fill_quadfolded_north_halos!(i, k, grid, sign, c)
    for j in 1:grid.Hy
        source_ring = octahealpix_halo_source_ring_index(i, grid.Ny + j, grid.Nx, grid.Ny, grid.connectivity)
        source_i = grid.connectivity.ring_to_i[source_ring]
        source_j = grid.connectivity.ring_to_j[source_ring]
        @inbounds c[i, grid.Ny + j, k] = sign * c[source_i, source_j, k]
    end

    return nothing
end

@inline _fill_west_halo!(j, k, grid, c, bc::QZBC, ::QuadFoldedScalarLikeLocation, args...) =
    fill_quadfolded_west_halos!(j, k, grid, bc.condition, c)

@inline _fill_east_halo!(j, k, grid, c, bc::QZBC, ::QuadFoldedScalarLikeLocation, args...) =
    fill_quadfolded_east_halos!(j, k, grid, bc.condition, c)

@inline _fill_south_halo!(i, k, grid, c, bc::QZBC, ::QuadFoldedScalarLikeLocation, args...) =
    fill_quadfolded_south_halos!(i, k, grid, bc.condition, c)

@inline _fill_north_halo!(i, k, grid, c, bc::QZBC, ::QuadFoldedScalarLikeLocation, args...) =
    fill_quadfolded_north_halos!(i, k, grid, bc.condition, c)

for BC in (:QCovZBC, :QConZBC)
    @eval begin
        @inline _fill_west_halo!(j, k, grid, c, bc::$BC, ::QuadFoldedVectorLocation, args...) = nothing
        @inline _fill_east_halo!(j, k, grid, c, bc::$BC, ::QuadFoldedVectorLocation, args...) = nothing
        @inline _fill_south_halo!(i, k, grid, c, bc::$BC, ::QuadFoldedVectorLocation, args...) = nothing
        @inline _fill_north_halo!(i, k, grid, c, bc::$BC, ::QuadFoldedVectorLocation, args...) = nothing
    end
end
