using OrderedCollections: OrderedDict

abstract type AbstractSphericalShellMapping end

struct EquiangularGnomonicCubedSpherePanel{FT} <: AbstractSphericalShellMapping
    α :: Tuple{FT, FT}
    β :: Tuple{FT, FT}
end

EquiangularGnomonicCubedSpherePanel(FT::DataType = Oceananigans.defaults.FloatType;
                                    α = (-π/4, π/4),
                                    β = (-π/4, π/4)) =
    EquiangularGnomonicCubedSpherePanel{FT}((FT(α[1]), FT(α[2])), (FT(β[1]), FT(β[2])))

struct OctaHEALPixMapping{I} <: AbstractSphericalShellMapping
    N :: I
end

function OctaHEALPixMapping(N::Integer)
    N > 0 || throw(ArgumentError("OctaHEALPixMapping requires N > 0."))
    return OctaHEALPixMapping{typeof(N)}(N)
end

@inline octahealpix_number_of_cells(mapping::OctaHEALPixMapping) = 4 * mapping.N^2
@inline octahealpix_matrix_size(mapping::OctaHEALPixMapping) = (2 * mapping.N, 2 * mapping.N)
@inline octahealpix_number_of_latitude_rings(mapping::OctaHEALPixMapping) = 2 * mapping.N - 1
@inline octahealpix_solid_angle(mapping::OctaHEALPixMapping) = 4π / octahealpix_number_of_cells(mapping)
@inline octahealpix_nlon_per_ring(mapping::OctaHEALPixMapping, j) = min(4j, 8 * mapping.N - 4j)

@inline function octahealpix_latitude(mapping::OctaHEALPixMapping, j)
    N = mapping.N
    FT = float(typeof(N))
    z = ifelse(j <= N,
                one(FT) - FT(j)^2 / FT(N)^2,
               -one(FT) + FT(2N - j)^2 / FT(N)^2)
    return asind(z)
end

@inline octahealpix_latitude(::Type{FT}, mapping::OctaHEALPixMapping, j) where FT =
    convert(FT, octahealpix_latitude(mapping, j))

@inline function octahealpix_longitude(mapping::OctaHEALPixMapping, j, i)
    nλ = octahealpix_nlon_per_ring(mapping, j)
    return (i - convert(float(typeof(i)), 1//2)) * 360 / nλ
end

@inline octahealpix_longitude(::Type{FT}, mapping::OctaHEALPixMapping, j, i) where FT =
    convert(FT, octahealpix_longitude(mapping, j, i))

function octahealpix_index_range_in_ring(mapping::OctaHEALPixMapping, j)
    first_index = 1
    for r in 1:j-1
        first_index += octahealpix_nlon_per_ring(mapping, r)
    end

    last_index = first_index + octahealpix_nlon_per_ring(mapping, j) - 1
    return first_index:last_index
end

function octahealpix_ring2rcq(ring_index, mapping::OctaHEALPixMapping)
    for r in 1:octahealpix_number_of_latitude_rings(mapping)
        range = octahealpix_index_range_in_ring(mapping, r)
        if ring_index in range
            index_in_ring = ring_index - first(range) + 1
            q_width = max(1, octahealpix_nlon_per_ring(mapping, r) ÷ 4)
            q = (index_in_ring - 1) ÷ q_width + 1
            c = index_in_ring - (q - 1) * q_width
            return r, c, q
        end
    end

    throw(ArgumentError("Ring index $ring_index is outside $(summary(mapping))."))
end

function octahealpix_rcq2ring(r, c, q, mapping::OctaHEALPixMapping)
    range = octahealpix_index_range_in_ring(mapping, r)
    q_width = max(1, length(range) ÷ 4)
    return first(range) + (q - 1) * q_width + c - 1
end

@inline function octahealpix_ring2matrix(ring_index, mapping::OctaHEALPixMapping)
    Nx, Ny = octahealpix_matrix_size(mapping)
    i = mod(ring_index - 1, Nx) + 1
    j = (ring_index - 1) ÷ Nx + 1
    return i, j
end

@inline octahealpix_wrapped_index(i, N) = mod(i - 1, N) + 1

@inline function octahealpix_wrapped_ring_index(i, j, Nx, Ny, connectivity)
    source_i = clamp(i, 1, Nx)
    source_j = clamp(j, 1, Ny)
    source_ring = @inbounds connectivity.matrix_to_ring[source_i, source_j]

    if i < 1
        for _ in 1:(1 - i)
            source_ring = @inbounds connectivity.ring_to_minus_i_neighbor[source_ring]
        end
    elseif i > Nx
        for _ in 1:(i - Nx)
            source_ring = @inbounds connectivity.ring_to_plus_i_neighbor[source_ring]
        end
    end

    if j < 1
        for _ in 1:(1 - j)
            source_ring = @inbounds connectivity.ring_to_minus_j_neighbor[source_ring]
        end
    elseif j > Ny
        for _ in 1:(j - Ny)
            source_ring = @inbounds connectivity.ring_to_plus_j_neighbor[source_ring]
        end
    end

    return source_ring
end

@inline function octahealpix_wrapped_center_coordinates(i, j, Nx, Ny, connectivity)
    ring = octahealpix_wrapped_ring_index(i, j, Nx, Ny, connectivity)
    return @inbounds connectivity.ring_to_i[ring], connectivity.ring_to_j[ring]
end

@inline octahealpix_center_folded_index(i, N) =
    ifelse(i < 1, 1 - i, ifelse(i > N, 2N + 1 - i, i))

@inline octahealpix_crosses_polar_fold(j, Ny) =
    (j < 1) | (j > Ny)

@inline function octahealpix_folded_halo_source_indices(i, j, Nx, Ny)
    crosses_polar_fold = octahealpix_crosses_polar_fold(j, Ny)
    polar_i_shift = ifelse(crosses_polar_fold, Nx ÷ 2, 0)
    source_i = octahealpix_wrapped_index(i + polar_i_shift, Nx)
    source_j = octahealpix_center_folded_index(j, Ny)
    return source_i, source_j, crosses_polar_fold
end

@inline function octahealpix_folded_halo_source_ring_index(i, j, Nx, Ny, connectivity)
    source_i, source_j, _ = octahealpix_folded_halo_source_indices(i, j, Nx, Ny)
    return @inbounds connectivity.matrix_to_ring[source_i, source_j]
end

@inline function octahealpix_halo_source_ring_index_and_rotation(i, j, Nx, Ny, connectivity)
    start_ring = @inbounds connectivity.matrix_to_ring[clamp(i, 1, Nx), clamp(j, 1, Ny)]
    source_ring = octahealpix_folded_halo_source_ring_index(i, j, Nx, Ny, connectivity)
    start_rotation = @inbounds octahealpix_quadrant_rotation(connectivity.ring_to_q[start_ring])
    source_rotation = @inbounds octahealpix_quadrant_rotation(connectivity.ring_to_q[source_ring])
    total_rotation = mod(start_rotation - source_rotation, 4)
    return source_ring, start_ring, total_rotation
end

@inline octahealpix_halo_source_ring_index(i, j, Nx, Ny, connectivity) =
    octahealpix_folded_halo_source_ring_index(i, j, Nx, Ny, connectivity)

@inline function octahealpix_block_to_quadrant(block_i, block_j)
    return ifelse(block_j == 1,
                  ifelse(block_i == 1, 1, 2),
                  ifelse(block_i == 2, 3, 4))
end

@inline octahealpix_block_to_quadrant(q) = q

@inline function octahealpix_quadrant_block(q)
    block_i = ifelse((q == 1) | (q == 4), 1, 2)
    block_j = ifelse((q == 1) | (q == 2), 1, 2)
    return block_i, block_j
end

@inline octahealpix_quadrant_rotation(q) = mod(q - 1, 4)

@inline function octahealpix_matrix_quadrant(i, j, N)
    block_i = ifelse(i <= N, 1, 2)
    block_j = ifelse(j <= N, 1, 2)
    return octahealpix_block_to_quadrant(block_i, block_j)
end

@inline function rotate_octahealpix_indices(r, c, N, rotation)
    rotation′ = mod(rotation, 4)
    r′ = ifelse(rotation′ == 0, r,
         ifelse(rotation′ == 1, c,
         ifelse(rotation′ == 2, N + 1 - r, N + 1 - c)))
    c′ = ifelse(rotation′ == 0, c,
         ifelse(rotation′ == 1, N + 1 - r,
         ifelse(rotation′ == 2, N + 1 - c, r)))
    return r′, c′
end

@inline function octahealpix_rotate_step(di, dj, rotation)
    rotation′ = mod(rotation, 4)
    di′ = ifelse(rotation′ == 0, di,
          ifelse(rotation′ == 1, dj,
          ifelse(rotation′ == 2, -di, -dj)))
    dj′ = ifelse(rotation′ == 0, dj,
          ifelse(rotation′ == 1, -di,
          ifelse(rotation′ == 2, -dj, di)))
    return di′, dj′
end

@inline octahealpix_matrix_step_from_local_step(dr, dc, rotation) =
    octahealpix_rotate_step(dr, dc, rotation)

@inline octahealpix_local_step_from_matrix_step(di, dj, rotation) =
    octahealpix_rotate_step(di, dj, -rotation)

@inline _octahealpix_matrix_step_from_local_step(dr, dc, rotation) =
    octahealpix_matrix_step_from_local_step(dr, dc, rotation)

@inline _octahealpix_local_step_from_matrix_step(di, dj, rotation) =
    octahealpix_local_step_from_matrix_step(di, dj, rotation)

@inline function octahealpix_matrix_to_local(i, j, N, q)
    block_i, block_j = octahealpix_quadrant_block(q)
    r = i - (block_i - 1) * N
    c = j - (block_j - 1) * N
    return rotate_octahealpix_indices(r, c, N, -octahealpix_quadrant_rotation(q))
end

@inline function octahealpix_local_to_matrix(r, c, q, N)
    r′, c′ = rotate_octahealpix_indices(r, c, N, octahealpix_quadrant_rotation(q))
    block_i, block_j = octahealpix_quadrant_block(q)
    i = r′ + (block_i - 1) * N
    j = c′ + (block_j - 1) * N
    return i, j
end

@inline function octahealpix_destination_quadrant(i, j, di, dj, N)
    i′ = octahealpix_wrapped_index(i + di, 2N)
    j′ = octahealpix_wrapped_index(j + dj, 2N)
    return octahealpix_matrix_quadrant(i′, j′, N)
end

@inline _octahealpix_destination_quadrant(i, j, di, dj, N) =
    octahealpix_destination_quadrant(i, j, di, dj, N)

@inline function octahealpix_step_stays_in_matrix_block(i, j, di, dj, N)
    i′ = i + di
    j′ = j + dj
    inside_matrix = (1 <= i′ <= 2N) & (1 <= j′ <= 2N)
    q = octahealpix_matrix_quadrant(i, j, N)
    q′ = octahealpix_matrix_quadrant(clamp(i′, 1, 2N), clamp(j′, 1, 2N), N)
    return inside_matrix & (q == q′)
end

@inline _octahealpix_step_stays_in_matrix_block(i, j, di, dj, N) =
    octahealpix_step_stays_in_matrix_block(i, j, di, dj, N)

@inline function octahealpix_rotated_matrix_neighbor(i, j, di, dj, N,
                                                     destination_quadrant,
                                                     use_stepped_local_r,
                                                     use_stepped_local_c)
    q = octahealpix_matrix_quadrant(i, j, N)
    q_rotation = octahealpix_quadrant_rotation(q)
    destination_rotation = octahealpix_quadrant_rotation(destination_quadrant)
    r, c = octahealpix_matrix_to_local(i, j, N, q)
    dr, dc = octahealpix_local_step_from_matrix_step(di, dj, q_rotation)
    r_neighbor = r + dr
    c_neighbor = c + dc
    wrapped_r = ifelse(use_stepped_local_r, octahealpix_wrapped_index(r_neighbor, N), r_neighbor)
    wrapped_c = ifelse(use_stepped_local_c, octahealpix_wrapped_index(c_neighbor, N), c_neighbor)
    destination_r, destination_c =
        rotate_octahealpix_indices(wrapped_r, wrapped_c, N, q_rotation - destination_rotation)
    destination_i, destination_j =
        octahealpix_local_to_matrix(destination_r, destination_c, destination_quadrant, N)
    return octahealpix_wrapped_index(destination_i, 2N),
           octahealpix_wrapped_index(destination_j, 2N)
end

@inline function octahealpix_neighbor_alignment(i, j, destination_i, destination_j, mapping::OctaHEALPixMapping)
    λ, φ = octahealpix_horizontal_longitude_latitude(mapping, i, j, Center(), Center())
    destination_λ, destination_φ =
        octahealpix_horizontal_longitude_latitude(mapping, destination_i, destination_j, Center(), Center())
    x, y, z = spherical_shell_unit_vector(λ, φ)
    destination_x, destination_y, destination_z = spherical_shell_unit_vector(destination_λ, destination_φ)
    return x * destination_x + y * destination_y + z * destination_z
end

@inline function octahealpix_wrap_matrix_neighbor(i, j, di, dj, mapping::OctaHEALPixMapping)
    Nx, Ny = octahealpix_matrix_size(mapping)
    return octahealpix_wrapped_index(i + di, Nx), octahealpix_wrapped_index(j + dj, Ny)
end

@inline octahealpix_rotated_matrix_neighbor(i, j, di, dj, mapping::OctaHEALPixMapping) =
    octahealpix_wrap_matrix_neighbor(i, j, di, dj, mapping)


struct OctaHEALPixConnectivity{M, V}
    matrix_to_ring :: M
    ring_to_i :: V
    ring_to_j :: V
    ring_to_r :: V
    ring_to_c :: V
    ring_to_q :: V
    ring_to_latitude_ring :: V
    ring_to_index_in_ring :: V
    ring_to_minus_i_neighbor :: V
    ring_to_plus_i_neighbor :: V
    ring_to_minus_j_neighbor :: V
    ring_to_plus_j_neighbor :: V
    ring_to_minus_i_halo_source :: V
    ring_to_plus_i_halo_source :: V
    ring_to_minus_j_halo_source :: V
    ring_to_plus_j_halo_source :: V
    ring_to_minus_i_halo_source_rotation :: V
    ring_to_plus_i_halo_source_rotation :: V
    ring_to_minus_j_halo_source_rotation :: V
    ring_to_plus_j_halo_source_rotation :: V
    ring_to_minus_i_covariant_x_halo_map :: V
    ring_to_minus_i_covariant_y_halo_map :: V
    ring_to_plus_i_covariant_x_halo_map :: V
    ring_to_plus_i_covariant_y_halo_map :: V
    ring_to_minus_j_covariant_x_halo_map :: V
    ring_to_minus_j_covariant_y_halo_map :: V
    ring_to_plus_j_covariant_x_halo_map :: V
    ring_to_plus_j_covariant_y_halo_map :: V
    ring_to_minus_i_contravariant_x_halo_map :: V
    ring_to_minus_i_contravariant_y_halo_map :: V
    ring_to_plus_i_contravariant_x_halo_map :: V
    ring_to_plus_i_contravariant_y_halo_map :: V
    ring_to_minus_j_contravariant_x_halo_map :: V
    ring_to_minus_j_contravariant_y_halo_map :: V
    ring_to_plus_j_contravariant_x_halo_map :: V
    ring_to_plus_j_contravariant_y_halo_map :: V
end

function OctaHEALPixConnectivity(mapping::OctaHEALPixMapping)
    Nx, Ny = octahealpix_matrix_size(mapping)
    number_of_cells = octahealpix_number_of_cells(mapping)

    matrix_to_ring = Array{Int}(undef, Nx, Ny)
    ring_to_i = Vector{Int}(undef, number_of_cells)
    ring_to_j = Vector{Int}(undef, number_of_cells)
    ring_to_r = Vector{Int}(undef, number_of_cells)
    ring_to_c = Vector{Int}(undef, number_of_cells)
    ring_to_q = Vector{Int}(undef, number_of_cells)
    ring_to_latitude_ring = Vector{Int}(undef, number_of_cells)
    ring_to_index_in_ring = Vector{Int}(undef, number_of_cells)
    ring_to_minus_i_neighbor = Vector{Int}(undef, number_of_cells)
    ring_to_plus_i_neighbor = Vector{Int}(undef, number_of_cells)
    ring_to_minus_j_neighbor = Vector{Int}(undef, number_of_cells)
    ring_to_plus_j_neighbor = Vector{Int}(undef, number_of_cells)
    ring_to_minus_i_halo_source = Vector{Int}(undef, number_of_cells)
    ring_to_plus_i_halo_source = Vector{Int}(undef, number_of_cells)
    ring_to_minus_j_halo_source = Vector{Int}(undef, number_of_cells)
    ring_to_plus_j_halo_source = Vector{Int}(undef, number_of_cells)
    ring_to_minus_i_halo_source_rotation = Vector{Int}(undef, number_of_cells)
    ring_to_plus_i_halo_source_rotation = Vector{Int}(undef, number_of_cells)
    ring_to_minus_j_halo_source_rotation = Vector{Int}(undef, number_of_cells)
    ring_to_plus_j_halo_source_rotation = Vector{Int}(undef, number_of_cells)
    ring_to_minus_i_covariant_x_halo_map = Vector{Int}(undef, number_of_cells)
    ring_to_minus_i_covariant_y_halo_map = Vector{Int}(undef, number_of_cells)
    ring_to_plus_i_covariant_x_halo_map = Vector{Int}(undef, number_of_cells)
    ring_to_plus_i_covariant_y_halo_map = Vector{Int}(undef, number_of_cells)
    ring_to_minus_j_covariant_x_halo_map = Vector{Int}(undef, number_of_cells)
    ring_to_minus_j_covariant_y_halo_map = Vector{Int}(undef, number_of_cells)
    ring_to_plus_j_covariant_x_halo_map = Vector{Int}(undef, number_of_cells)
    ring_to_plus_j_covariant_y_halo_map = Vector{Int}(undef, number_of_cells)
    ring_to_minus_i_contravariant_x_halo_map = Vector{Int}(undef, number_of_cells)
    ring_to_minus_i_contravariant_y_halo_map = Vector{Int}(undef, number_of_cells)
    ring_to_plus_i_contravariant_x_halo_map = Vector{Int}(undef, number_of_cells)
    ring_to_plus_i_contravariant_y_halo_map = Vector{Int}(undef, number_of_cells)
    ring_to_minus_j_contravariant_x_halo_map = Vector{Int}(undef, number_of_cells)
    ring_to_minus_j_contravariant_y_halo_map = Vector{Int}(undef, number_of_cells)
    ring_to_plus_j_contravariant_x_halo_map = Vector{Int}(undef, number_of_cells)
    ring_to_plus_j_contravariant_y_halo_map = Vector{Int}(undef, number_of_cells)

    for ring in 1:number_of_cells
        i, j = octahealpix_ring2matrix(ring, mapping)
        matrix_to_ring[i, j] = ring
        ring_to_i[ring] = i
        ring_to_j[ring] = j

        r, c, q = octahealpix_ring2rcq(ring, mapping)
        ring_to_r[ring] = r
        ring_to_c[ring] = c
        ring_to_q[ring] = q
        ring_to_latitude_ring[ring] = r
        ring_to_index_in_ring[ring] = (q - 1) * max(1, octahealpix_nlon_per_ring(mapping, r) ÷ 4) + c
    end

    for ring in 1:number_of_cells
        i = ring_to_i[ring]
        j = ring_to_j[ring]
        im, jm = octahealpix_wrap_matrix_neighbor(i, j, -1,  0, mapping)
        ip, jp = octahealpix_wrap_matrix_neighbor(i, j,  1,  0, mapping)
        is, js = octahealpix_wrap_matrix_neighbor(i, j,  0, -1, mapping)
        in, jn = octahealpix_wrap_matrix_neighbor(i, j,  0,  1, mapping)
        ring_to_minus_i_neighbor[ring] = matrix_to_ring[im, jm]
        ring_to_plus_i_neighbor[ring]  = matrix_to_ring[ip, jp]
        ring_to_minus_j_neighbor[ring] = matrix_to_ring[is, js]
        ring_to_plus_j_neighbor[ring]  = matrix_to_ring[in, jn]
    end

    @inline function halo_source_matrix_neighbor(source_i, source_j, di, dj)
        q = octahealpix_matrix_quadrant(source_i, source_j, mapping.N)
        q_rotation = octahealpix_quadrant_rotation(q)
        r, c = octahealpix_matrix_to_local(source_i, source_j, mapping.N, q)
        dr, dc = octahealpix_local_step_from_matrix_step(di, dj, q_rotation)
        r_neighbor = r + dr
        c_neighbor = c + dc
        same_block_step = octahealpix_step_stays_in_matrix_block(source_i, source_j, di, dj, mapping.N)

        if same_block_step && 1 <= r_neighbor <= mapping.N && 1 <= c_neighbor <= mapping.N
            return source_i + di, source_j + dj
        end

        seam_class_destination_quadrant =
            mod1(q + ifelse(!(1 <= source_i + di <= 2 * mapping.N &&
                              1 <= source_j + dj <= 2 * mapping.N), -1, +1), 4)

        i, j = octahealpix_rotated_matrix_neighbor(source_i,
                                                   source_j,
                                                   di,
                                                   dj,
                                                   mapping.N,
                                                   seam_class_destination_quadrant,
                                                   false,
                                                   false)

        return clamp(i, 1, Nx), clamp(j, 1, Ny)
    end

    @inline function halo_source_ring_and_rotation(i, j)
        source_i = clamp(i, 1, Nx)
        source_j = clamp(j, 1, Ny)
        start_ring = matrix_to_ring[source_i, source_j]
        total_rotation = 0

        if i < 1
            for _ in 1:(1 - i)
                current_ring = matrix_to_ring[source_i, source_j]
                source_i, source_j = halo_source_matrix_neighbor(source_i, source_j, -1, 0)
                source_ring = matrix_to_ring[source_i, source_j]
                current_rotation = octahealpix_quadrant_rotation(ring_to_q[current_ring])
                source_rotation = octahealpix_quadrant_rotation(ring_to_q[source_ring])
                total_rotation = mod(total_rotation + current_rotation - source_rotation, 4)
            end
        elseif i > Nx
            for _ in 1:(i - Nx)
                current_ring = matrix_to_ring[source_i, source_j]
                source_i, source_j = halo_source_matrix_neighbor(source_i, source_j, +1, 0)
                source_ring = matrix_to_ring[source_i, source_j]
                current_rotation = octahealpix_quadrant_rotation(ring_to_q[current_ring])
                source_rotation = octahealpix_quadrant_rotation(ring_to_q[source_ring])
                total_rotation = mod(total_rotation + current_rotation - source_rotation, 4)
            end
        end

        if j < 1
            for _ in 1:(1 - j)
                current_ring = matrix_to_ring[source_i, source_j]
                source_i, source_j = halo_source_matrix_neighbor(source_i, source_j, 0, -1)
                source_ring = matrix_to_ring[source_i, source_j]
                current_rotation = octahealpix_quadrant_rotation(ring_to_q[current_ring])
                source_rotation = octahealpix_quadrant_rotation(ring_to_q[source_ring])
                total_rotation = mod(total_rotation + current_rotation - source_rotation, 4)
            end
        elseif j > Ny
            for _ in 1:(j - Ny)
                current_ring = matrix_to_ring[source_i, source_j]
                source_i, source_j = halo_source_matrix_neighbor(source_i, source_j, 0, +1)
                source_ring = matrix_to_ring[source_i, source_j]
                current_rotation = octahealpix_quadrant_rotation(ring_to_q[current_ring])
                source_rotation = octahealpix_quadrant_rotation(ring_to_q[source_ring])
                total_rotation = mod(total_rotation + current_rotation - source_rotation, 4)
            end
        end

        return matrix_to_ring[source_i, source_j], start_ring, total_rotation
    end

    @inline encode_vector_halo_map(source_kind, sign) = sign * source_kind

    for ring in 1:number_of_cells
        i = ring_to_i[ring]
        j = ring_to_j[ring]

        minus_i, _, minus_i_rotation = halo_source_ring_and_rotation(i - 1, j)
        plus_i,  _, plus_i_rotation  = halo_source_ring_and_rotation(i + 1, j)
        minus_j, _, minus_j_rotation = halo_source_ring_and_rotation(i, j - 1)
        plus_j,  _, plus_j_rotation  = halo_source_ring_and_rotation(i, j + 1)

        ring_to_minus_i_halo_source[ring] = minus_i
        ring_to_plus_i_halo_source[ring] = plus_i
        ring_to_minus_j_halo_source[ring] = minus_j
        ring_to_plus_j_halo_source[ring] = plus_j
        ring_to_minus_i_halo_source_rotation[ring] = minus_i_rotation
        ring_to_plus_i_halo_source_rotation[ring] = plus_i_rotation
        ring_to_minus_j_halo_source_rotation[ring] = minus_j_rotation
        ring_to_plus_j_halo_source_rotation[ring] = plus_j_rotation

        x_kind, x_sign = octahealpix_covariant_vector_halo_transform(minus_i_rotation, 1)
        y_kind, y_sign = octahealpix_covariant_vector_halo_transform(minus_i_rotation, 2)
        ring_to_minus_i_covariant_x_halo_map[ring] = encode_vector_halo_map(x_kind, x_sign)
        ring_to_minus_i_covariant_y_halo_map[ring] = encode_vector_halo_map(y_kind, y_sign)

        x_kind, x_sign = octahealpix_covariant_vector_halo_transform(plus_i_rotation, 1)
        y_kind, y_sign = octahealpix_covariant_vector_halo_transform(plus_i_rotation, 2)
        ring_to_plus_i_covariant_x_halo_map[ring] = encode_vector_halo_map(x_kind, x_sign)
        ring_to_plus_i_covariant_y_halo_map[ring] = encode_vector_halo_map(y_kind, y_sign)

        x_kind, x_sign = octahealpix_covariant_vector_halo_transform(minus_j_rotation, 1)
        y_kind, y_sign = octahealpix_covariant_vector_halo_transform(minus_j_rotation, 2)
        ring_to_minus_j_covariant_x_halo_map[ring] = encode_vector_halo_map(x_kind, x_sign)
        ring_to_minus_j_covariant_y_halo_map[ring] = encode_vector_halo_map(y_kind, y_sign)

        x_kind, x_sign = octahealpix_covariant_vector_halo_transform(plus_j_rotation, 1)
        y_kind, y_sign = octahealpix_covariant_vector_halo_transform(plus_j_rotation, 2)
        ring_to_plus_j_covariant_x_halo_map[ring] = encode_vector_halo_map(x_kind, x_sign)
        ring_to_plus_j_covariant_y_halo_map[ring] = encode_vector_halo_map(y_kind, y_sign)

        x_kind, x_sign = octahealpix_contravariant_vector_halo_transform(minus_i_rotation, 1)
        y_kind, y_sign = octahealpix_contravariant_vector_halo_transform(minus_i_rotation, 2)
        ring_to_minus_i_contravariant_x_halo_map[ring] = encode_vector_halo_map(x_kind, x_sign)
        ring_to_minus_i_contravariant_y_halo_map[ring] = encode_vector_halo_map(y_kind, y_sign)

        x_kind, x_sign = octahealpix_contravariant_vector_halo_transform(plus_i_rotation, 1)
        y_kind, y_sign = octahealpix_contravariant_vector_halo_transform(plus_i_rotation, 2)
        ring_to_plus_i_contravariant_x_halo_map[ring] = encode_vector_halo_map(x_kind, x_sign)
        ring_to_plus_i_contravariant_y_halo_map[ring] = encode_vector_halo_map(y_kind, y_sign)

        x_kind, x_sign = octahealpix_contravariant_vector_halo_transform(minus_j_rotation, 1)
        y_kind, y_sign = octahealpix_contravariant_vector_halo_transform(minus_j_rotation, 2)
        ring_to_minus_j_contravariant_x_halo_map[ring] = encode_vector_halo_map(x_kind, x_sign)
        ring_to_minus_j_contravariant_y_halo_map[ring] = encode_vector_halo_map(y_kind, y_sign)

        x_kind, x_sign = octahealpix_contravariant_vector_halo_transform(plus_j_rotation, 1)
        y_kind, y_sign = octahealpix_contravariant_vector_halo_transform(plus_j_rotation, 2)
        ring_to_plus_j_contravariant_x_halo_map[ring] = encode_vector_halo_map(x_kind, x_sign)
        ring_to_plus_j_contravariant_y_halo_map[ring] = encode_vector_halo_map(y_kind, y_sign)
    end

    return OctaHEALPixConnectivity(matrix_to_ring,
                                   ring_to_i,
                                   ring_to_j,
                                   ring_to_r,
                                   ring_to_c,
                                   ring_to_q,
                                   ring_to_latitude_ring,
                                   ring_to_index_in_ring,
                                   ring_to_minus_i_neighbor,
                                   ring_to_plus_i_neighbor,
                                   ring_to_minus_j_neighbor,
                                   ring_to_plus_j_neighbor,
                                   ring_to_minus_i_halo_source,
                                   ring_to_plus_i_halo_source,
                                   ring_to_minus_j_halo_source,
                                   ring_to_plus_j_halo_source,
                                   ring_to_minus_i_halo_source_rotation,
                                   ring_to_plus_i_halo_source_rotation,
                                   ring_to_minus_j_halo_source_rotation,
                                   ring_to_plus_j_halo_source_rotation,
                                   ring_to_minus_i_covariant_x_halo_map,
                                   ring_to_minus_i_covariant_y_halo_map,
                                   ring_to_plus_i_covariant_x_halo_map,
                                   ring_to_plus_i_covariant_y_halo_map,
                                   ring_to_minus_j_covariant_x_halo_map,
                                   ring_to_minus_j_covariant_y_halo_map,
                                   ring_to_plus_j_covariant_x_halo_map,
                                   ring_to_plus_j_covariant_y_halo_map,
                                   ring_to_minus_i_contravariant_x_halo_map,
                                   ring_to_minus_i_contravariant_y_halo_map,
                                   ring_to_plus_i_contravariant_x_halo_map,
                                   ring_to_plus_i_contravariant_y_halo_map,
                                   ring_to_minus_j_contravariant_x_halo_map,
                                   ring_to_minus_j_contravariant_y_halo_map,
                                   ring_to_plus_j_contravariant_x_halo_map,
                                   ring_to_plus_j_contravariant_y_halo_map)
end

function Adapt.adapt_structure(to, connectivity::OctaHEALPixConnectivity)
    return OctaHEALPixConnectivity((adapt(to, getproperty(connectivity, name))
                                    for name in fieldnames(typeof(connectivity)))...)
end

Architectures.on_architecture(arch::AbstractSerialArchitecture, connectivity::OctaHEALPixConnectivity) =
    connectivity

@inline function octahealpix_connectivity_matrix_neighbor(i::Integer,
                                                          j::Integer,
                                                          di::Integer,
                                                          dj::Integer,
                                                          mapping::OctaHEALPixMapping)
    return octahealpix_wrap_matrix_neighbor(i, j, di, dj, mapping)
end

@inline octahealpix_face_halo_source(i, j, connectivity, Nx, Ny) =
    octahealpix_halo_source_ring_index(i, j, Nx, Ny, connectivity)

@inline function _octahealpix_scalar_halo_source(i, j, Nx, Ny, connectivity)
    ring = octahealpix_halo_source_ring_index(i, j, Nx, Ny, connectivity)
    return @inbounds connectivity.ring_to_i[ring], connectivity.ring_to_j[ring]
end

@inline seam_source_indices(i, j, Nx, Ny, connectivity::OctaHEALPixConnectivity) =
    _octahealpix_scalar_halo_source(i, j, Nx, Ny, connectivity)

@inline function seam_vector_source_indices_and_sign(i, j, Nx, Ny, connectivity::OctaHEALPixConnectivity)
    source_i, source_j, crosses_polar_fold = octahealpix_folded_halo_source_indices(i, j, Nx, Ny)
    sign = ifelse(crosses_polar_fold, -1, 1)
    return source_i, source_j, sign
end

@inline octahealpix_component_index(::Val{:u}) = 1
@inline octahealpix_component_index(::Val{:v}) = 2
@inline octahealpix_component_index(component::Integer) = component

@inline function octahealpix_vector_halo_source(i, j, Nx, Ny, connectivity, component, transform=Val(:covariant))
    source_ring, _, rotation = octahealpix_halo_source_ring_index_and_rotation(i, j, Nx, Ny, connectivity)
    source_i = @inbounds connectivity.ring_to_i[source_ring]
    source_j = @inbounds connectivity.ring_to_j[source_ring]
    source_component, sign = octahealpix_vector_halo_transform(rotation, octahealpix_component_index(component), transform)
    return source_component, source_i, source_j, sign
end

@inline octahealpix_xface_vector_halo_source(i, j, Nx, Ny, connectivity, transform=Val(:covariant)) =
    octahealpix_vector_halo_source(i, j, Nx, Ny, connectivity, Val(:u), transform)
@inline octahealpix_yface_vector_halo_source(i, j, Nx, Ny, connectivity, transform=Val(:covariant)) =
    octahealpix_vector_halo_source(i, j, Nx, Ny, connectivity, Val(:v), transform)

@inline octahealpix_covariant_xface_halo_source(i, j, Nx, Ny, connectivity) =
    octahealpix_xface_vector_halo_source(i, j, Nx, Ny, connectivity, Val(:covariant))
@inline octahealpix_covariant_yface_halo_source(i, j, Nx, Ny, connectivity) =
    octahealpix_yface_vector_halo_source(i, j, Nx, Ny, connectivity, Val(:covariant))
@inline octahealpix_contravariant_xface_halo_source(i, j, Nx, Ny, connectivity) =
    octahealpix_xface_vector_halo_source(i, j, Nx, Ny, connectivity, Val(:contravariant))
@inline octahealpix_contravariant_yface_halo_source(i, j, Nx, Ny, connectivity) =
    octahealpix_yface_vector_halo_source(i, j, Nx, Ny, connectivity, Val(:contravariant))

@inline function octahealpix_vector_halo_source_pair(i, j, Nx, Ny, connectivity, transform=Val(:covariant))
    ux, ix, jx, sx = octahealpix_xface_vector_halo_source(i, j, Nx, Ny, connectivity, transform)
    vy, iy, jy, sy = octahealpix_yface_vector_halo_source(i, j, Nx, Ny, connectivity, transform)
    return ux, ix, jx, sx, vy, iy, jy, sy
end

@inline octahealpix_vector_halo_source_pair_from_maps(i, j, Nx, Ny, connectivity, transform=Val(:covariant)) =
    octahealpix_vector_halo_source_pair(i, j, Nx, Ny, connectivity, transform)

@inline function octahealpix_covariant_vector_halo_transform(rotation, component)
    rotation′ = mod(rotation, 4)

    source_component =
        ifelse(component == 1,
               ifelse((rotation′ == 0) | (rotation′ == 2), 1, 2),
               ifelse((rotation′ == 1) | (rotation′ == 3), 1, 2))

    sign =
        ifelse(component == 1,
               ifelse((rotation′ == 0) | (rotation′ == 1), 1, -1),
               ifelse((rotation′ == 0) | (rotation′ == 3), 1, -1))

    return source_component, sign
end

@inline function octahealpix_contravariant_vector_halo_transform(rotation, component)
    rotation′ = mod(rotation, 4)

    source_component =
        ifelse(component == 1,
               ifelse((rotation′ == 0) | (rotation′ == 2), 1, 2),
               ifelse((rotation′ == 1) | (rotation′ == 3), 1, 2))

    sign =
        ifelse(component == 1,
               ifelse((rotation′ == 0) | (rotation′ == 3), 1, -1),
               ifelse((rotation′ == 0) | (rotation′ == 1), 1, -1))

    return source_component, sign
end

@inline octahealpix_vector_halo_transform(rotation, component, ::Val{:covariant}) =
    octahealpix_covariant_vector_halo_transform(rotation, component)

@inline octahealpix_vector_halo_transform(rotation, component, ::Val{:contravariant}) =
    octahealpix_contravariant_vector_halo_transform(rotation, component)

@inline octahealpix_vector_halo_transform(rotation, component, ::CovariantVectorSeamTransform) =
    octahealpix_covariant_vector_halo_transform(rotation, component)

@inline octahealpix_vector_halo_transform(rotation, component, ::ContravariantVectorSeamTransform) =
    octahealpix_contravariant_vector_halo_transform(rotation, component)

@inline octahealpix_vector_halo_transform(rotation, component) =
    octahealpix_covariant_vector_halo_transform(rotation, component)

@inline function octahealpix_transport_face_halo_source(i, j, connectivity, Nx, Ny, N, component)
    if component === Val(:u) || component == 1 || component === :u
        return octahealpix_contravariant_xface_halo_source(i, j, Nx, Ny, connectivity)
    else
        return octahealpix_contravariant_yface_halo_source(i, j, Nx, Ny, connectivity)
    end
end

@inline octahealpix_average_longitude_latitude(λ₁, φ₁, λ₂, φ₂) = ((λ₁ + λ₂) / 2, (φ₁ + φ₂) / 2)
@inline octahealpix_average_longitude_latitude(λ₁, φ₁, λ₂, φ₂, λ₃, φ₃, λ₄, φ₄) =
    ((λ₁ + λ₂ + λ₃ + λ₄) / 4, (φ₁ + φ₂ + φ₃ + φ₄) / 4)
@inline octahealpix_corner_longitude_latitude(mapping, i, j) = octahealpix_horizontal_longitude_latitude(mapping, i, j, Face(), Face())
@inline octahealpix_edge_longitude_latitude(mapping, i, j, LX, LY) = octahealpix_horizontal_longitude_latitude(mapping, i, j, LX, LY)

struct SphericalShellMetrics{A}
    xᶜᶜᵃ :: A
    yᶜᶜᵃ :: A
    zᶜᶜᵃ :: A
    Jᶜᶜᵃ :: A
    Jᶠᶜᵃ :: A
    Jᶜᶠᵃ :: A
    g₁₁ᶜᶜᵃ :: A
    g₁₂ᶜᶜᵃ :: A
    g₂₂ᶜᶜᵃ :: A
    g¹¹ᶜᶜᵃ :: A
    g¹²ᶜᶜᵃ :: A
    g²²ᶜᶜᵃ :: A
    g¹¹ᶠᶜᵃ :: A
    g¹²ᶠᶜᵃ :: A
    g²¹ᶜᶠᵃ :: A
    g²²ᶜᶠᵃ :: A
    G¹¹ᶜᶜᵃ :: A
    G¹²ᶜᶜᵃ :: A
    G²²ᶜᶜᵃ :: A
    G¹¹ᶠᶜᵃ :: A
    G¹²ᶠᶜᵃ :: A
    G²¹ᶜᶠᵃ :: A
    G²²ᶜᶠᵃ :: A
end

function Adapt.adapt_structure(to, metrics::SphericalShellMetrics)
    return SphericalShellMetrics((adapt(to, getproperty(metrics, name)) for name in fieldnames(typeof(metrics)))...)
end

struct SphericalShellGrid{FT, TX, TY, TZ, Z, C, M, Metrics,
                          CC, FC, CF, FF, Arch, I} <: AbstractHorizontallyCurvilinearGrid{FT, TX, TY, TZ, Z, Arch}
    architecture :: Arch
    Nx :: I
    Ny :: I
    Nz :: I
    Hx :: I
    Hy :: I
    Hz :: I
    Lx :: FT
    Ly :: FT
    Lz :: FT
    z :: Z
    radius :: FT
    connectivity :: C
    mapping :: M
    metrics :: Metrics
    λᶜᶜᵃ :: CC
    λᶠᶜᵃ :: FC
    λᶜᶠᵃ :: CF
    λᶠᶠᵃ :: FF
    φᶜᶜᵃ :: CC
    φᶠᶜᵃ :: FC
    φᶜᶠᵃ :: CF
    φᶠᶠᵃ :: FF
    Δxᶜᶜᵃ :: CC
    Δxᶠᶜᵃ :: FC
    Δxᶜᶠᵃ :: CF
    Δxᶠᶠᵃ :: FF
    Δyᶜᶜᵃ :: CC
    Δyᶠᶜᵃ :: FC
    Δyᶜᶠᵃ :: CF
    Δyᶠᶠᵃ :: FF
    Azᶜᶜᵃ :: CC
    Azᶠᶜᵃ :: FC
    Azᶜᶠᵃ :: CF
    Azᶠᶠᵃ :: FF
end

const ZRegSphericalShellGrid = SphericalShellGrid{<:Any, <:Any, <:Any, <:Any, <:RegularVerticalCoordinate}

@inline octahealpix_corner_longitude_latitude(grid::SphericalShellGrid, i, j) =
    octahealpix_corner_longitude_latitude(grid.mapping, i, j)

@inline octahealpix_edge_longitude_latitude(grid::SphericalShellGrid, i, j, LX, LY) =
    octahealpix_edge_longitude_latitude(grid.mapping, i, j, LX, LY)

@inline octahealpix_edge_longitude_latitude(grid::SphericalShellGrid, i, j, location) =
    octahealpix_edge_longitude_latitude(grid, i, j, location...)

@inline metrics_precomputed(::SphericalShellGrid) = true
regular_dimensions(::ZRegSphericalShellGrid) = tuple(3)

@inline _horizontal_node_fraction(i, N, ::Center, ::Bounded) = (i - 1//2) / N
@inline _horizontal_node_fraction(i, N, ::Face,   ::Bounded) = (i - 1) / N
@inline _horizontal_node_fraction(i, N, ::Center, topo) = (i - 1//2) / N
@inline _horizontal_node_fraction(i, N, ::Face,   topo) = (i - 1) / N

@inline function spherical_shell_unit_vector(λ, φ)
    cosφ = hack_cosd(φ)
    return cosφ * cosd(λ), cosφ * sind(λ), hack_sind(φ)
end

@inline function spherical_shell_tangent_basis(λ, φ)
    sinλ = sind(λ)
    cosλ = cosd(λ)
    sinφ = hack_sind(φ)
    cosφ = hack_cosd(φ)
    eλ = (-sinλ, cosλ, zero(cosλ))
    eφ = (-cosλ * sinφ, -sinλ * sinφ, cosφ)
    er = (cosφ * cosλ, cosφ * sinλ, sinφ)
    return eλ, eφ, er
end

@inline function equiangular_gnomonic_coordinate(i, N, ::Center, bounds)
    lower, upper = bounds
    return lower + (i - convert(typeof(lower), 1//2)) * (upper - lower) / N
end

@inline function equiangular_gnomonic_coordinate(i, N, ::Face, bounds)
    lower, upper = bounds
    return lower + (i - one(typeof(lower))) * (upper - lower) / N
end

@inline function equiangular_gnomonic_panel_unit_vector(α, β)
    x = one(α)
    y = tan(α)
    z = tan(β)
    inverse_radius = inv(sqrt(x^2 + y^2 + z^2))
    return x * inverse_radius, y * inverse_radius, z * inverse_radius
end

@inline function equiangular_gnomonic_panel_cartesian_node(α, β, radius)
    x, y, z = equiangular_gnomonic_panel_unit_vector(α, β)
    return radius * x, radius * y, radius * z
end

@inline function _equiangular_gnomonic_panel_solid_angle_primitive(α, β)
    x = tan(α)
    y = tan(β)
    return atan(x * y, sqrt(one(x) + x^2 + y^2))
end

@inline function equiangular_gnomonic_panel_solid_angle(α₁, α₂, β₁, β₂)
    return  _equiangular_gnomonic_panel_solid_angle_primitive(α₂, β₂) -
            _equiangular_gnomonic_panel_solid_angle_primitive(α₁, β₂) -
            _equiangular_gnomonic_panel_solid_angle_primitive(α₂, β₁) +
            _equiangular_gnomonic_panel_solid_angle_primitive(α₁, β₁)
end

@inline function equiangular_gnomonic_panel_solid_angle(α, β)
    x = tan(α)
    y = tan(β)
    A = one(x) + x^2
    B = one(y) + y^2
    s = one(x) + x^2 + y^2
    return A * B / sqrt(s^3)
end

@inline function equiangular_gnomonic_panel_metric_tensor(α, β, radius)
    x = tan(α)
    y = tan(β)
    A = one(x) + x^2
    B = one(y) + y^2
    s = one(x) + x^2 + y^2
    radius² = radius^2
    inverse_s² = inv(s^2)

    g₁₁ = radius² * A^2 * B * inverse_s²
    g₁₂ = - radius² * A * B * x * y * inverse_s²
    g₂₂ = radius² * A * B^2 * inverse_s²

    determinant = g₁₁ * g₂₂ - g₁₂^2
    J = sqrt(determinant)
    determinant⁻¹ = inv(determinant)

    g¹¹ = g₂₂ * determinant⁻¹
    g¹² = - g₁₂ * determinant⁻¹
    g²² = g₁₁ * determinant⁻¹

    G¹¹ = J * g¹¹
    G¹² = J * g¹²
    G²² = J * g²²

    return J, g₁₁, g₁₂, g₂₂, g¹¹, g¹², g²², G¹¹, G¹², G²²
end

@inline function _normalize_longitude(λ)
    λ′ = mod(λ + 180, 360) - 180
    return ifelse(λ′ == -180, convert(typeof(λ′), 180), λ′)
end

@inline function _mapping_longitude_latitude(mapping::EquiangularGnomonicCubedSpherePanel, ξ, η)
    α = mapping.α[1] + ξ * (mapping.α[2] - mapping.α[1])
    β = mapping.β[1] + η * (mapping.β[2] - mapping.β[1])
    x, y, z = equiangular_gnomonic_panel_unit_vector(α, β)
    return rad2deg(atan(y, x)), rad2deg(asin(z))
end

@inline function _mapping_longitude_latitude(mapping::OctaHEALPixMapping, ξ, η)
    η₀ = zero(η)
    η₁ = one(η)
    η₂ = η₁ + η₁
    south_polar_fold = η < η₀
    north_polar_fold = η > η₁
    crosses_polar_fold = south_polar_fold | north_polar_fold
    η′ = ifelse(south_polar_fold, -η, ifelse(north_polar_fold, η₂ - η, η))
    ξ′ = ifelse(crosses_polar_fold, ξ + convert(typeof(ξ), 1//2), ξ)
    λ = -180 + 360 * mod(ξ′, one(ξ′))
    # OctaHEALPix cells have uniform solid angle, so latitude is uniform
    # in sin(φ), not in φ.
    z = -one(η′) + η₂ * η′
    φ = asind(z)
    return _normalize_longitude(λ), φ
end

@inline function octahealpix_horizontal_longitude_latitude(mapping::OctaHEALPixMapping, i, j, LX, LY)
    N = 2 * mapping.N
    ξ = _horizontal_node_fraction(i, N, LX, QuadFolded())
    η = _horizontal_node_fraction(j, N, LY, QuadFolded())
    return _mapping_longitude_latitude(mapping, ξ, η)
end

@inline function octahealpix_horizontal_longitude_latitude(i, j, grid::SphericalShellGrid, LX, LY)
    return octahealpix_horizontal_longitude_latitude(grid.mapping, i, j, LX, LY)
end

function _new_horizontal_data(FT, loc, topo, size, halo)
    return new_data(FT, CPU(), loc, topo, size, halo)
end

function _fill_horizontal_coordinates!(λ, φ, mapping, LX, LY, topo, size)
    Nx, Ny = size
    TX, TY = topo

    for j in axes(λ, 2), i in axes(λ, 1)
        ξ = _horizontal_node_fraction(i, Nx, LX(), TX())
        η = _horizontal_node_fraction(j, Ny, LY(), TY())
        λᵢⱼ, φᵢⱼ = _mapping_longitude_latitude(mapping, ξ, η)
        λ[i, j] = λᵢⱼ
        φ[i, j] = φᵢⱼ
    end

    return nothing
end

@inline function _cartesian_from_lonlat(radius, λ, φ)
    x̂, ŷ, ẑ = spherical_shell_unit_vector(λ, φ)
    return radius * x̂, radius * ŷ, radius * ẑ
end

@inline _clamp_axis_index(i, A, dim) = clamp(i, first(axes(A, dim)), last(axes(A, dim)))

function _metric_tensor_from_coordinates(i, j, radius, λ, φ)
    im = _clamp_axis_index(i - 1, λ, 1)
    ip = _clamp_axis_index(i + 1, λ, 1)
    jm = _clamp_axis_index(j - 1, λ, 2)
    jp = _clamp_axis_index(j + 1, λ, 2)

    xip, yip, zip = _cartesian_from_lonlat(radius, λ[ip, j], φ[ip, j])
    xim, yim, zim = _cartesian_from_lonlat(radius, λ[im, j], φ[im, j])
    xjp, yjp, zjp = _cartesian_from_lonlat(radius, λ[i, jp], φ[i, jp])
    xjm, yjm, zjm = _cartesian_from_lonlat(radius, λ[i, jm], φ[i, jm])

    half = convert(eltype(λ), 1//2)
    a₁x = half * (xip - xim)
    a₁y = half * (yip - yim)
    a₁z = half * (zip - zim)
    a₂x = half * (xjp - xjm)
    a₂y = half * (yjp - yjm)
    a₂z = half * (zjp - zjm)

    g₁₁ = a₁x^2 + a₁y^2 + a₁z^2
    g₁₂ = a₁x * a₂x + a₁y * a₂y + a₁z * a₂z
    g₂₂ = a₂x^2 + a₂y^2 + a₂z^2
    detg = g₁₁ * g₂₂ - g₁₂^2
    safe_detg = max(detg, eps(eltype(λ)))
    J = sqrt(safe_detg)
    g¹¹ = g₂₂ / safe_detg
    g¹² = -g₁₂ / safe_detg
    g²² = g₁₁ / safe_detg
    G¹¹ = J * g¹¹
    G¹² = J * g¹²
    G²² = J * g²²

    return J, g₁₁, g₁₂, g₂₂, g¹¹, g¹², g²², G¹¹, G¹², G²²
end

@inline function _metric_tensor_from_mapping(i, j, radius, mapping::EquiangularGnomonicCubedSpherePanel, LX, LY, size, λ, φ)
    Nx, Ny = size[1:2]
    α = equiangular_gnomonic_coordinate(i, Nx, LX, mapping.α)
    β = equiangular_gnomonic_coordinate(j, Ny, LY, mapping.β)
    return equiangular_gnomonic_panel_metric_tensor(α, β, radius)
end

@inline function _metric_tensor_from_mapping(i, j, radius, mapping::OctaHEALPixMapping, LX, LY, size, λ, φ)
    Nx, Ny = size[1:2]
    half = convert(eltype(λ), 1//2)

    λ⁺ᶦ, φ⁺ᶦ = octahealpix_horizontal_longitude_latitude(mapping, i + 1, j, LX, LY)
    λ⁻ᶦ, φ⁻ᶦ = octahealpix_horizontal_longitude_latitude(mapping, i - 1, j, LX, LY)
    λ⁺ʲ, φ⁺ʲ = octahealpix_horizontal_longitude_latitude(mapping, i, j + 1, LX, LY)
    λ⁻ʲ, φ⁻ʲ = octahealpix_horizontal_longitude_latitude(mapping, i, j - 1, LX, LY)

    x⁺ᶦ, y⁺ᶦ, z⁺ᶦ = _cartesian_from_lonlat(radius, λ⁺ᶦ, φ⁺ᶦ)
    x⁻ᶦ, y⁻ᶦ, z⁻ᶦ = _cartesian_from_lonlat(radius, λ⁻ᶦ, φ⁻ᶦ)
    x⁺ʲ, y⁺ʲ, z⁺ʲ = _cartesian_from_lonlat(radius, λ⁺ʲ, φ⁺ʲ)
    x⁻ʲ, y⁻ʲ, z⁻ʲ = _cartesian_from_lonlat(radius, λ⁻ʲ, φ⁻ʲ)

    a₁x = half * (x⁺ᶦ - x⁻ᶦ)
    a₁y = half * (y⁺ᶦ - y⁻ᶦ)
    a₁z = half * (z⁺ᶦ - z⁻ᶦ)
    a₂x = half * (x⁺ʲ - x⁻ʲ)
    a₂y = half * (y⁺ʲ - y⁻ʲ)
    a₂z = half * (z⁺ʲ - z⁻ʲ)

    g₁₁ = a₁x^2 + a₁y^2 + a₁z^2
    g₁₂ = a₁x * a₂x + a₁y * a₂y + a₁z * a₂z
    g₂₂ = a₂x^2 + a₂y^2 + a₂z^2

    determinant = g₁₁ * g₂₂ - g₁₂^2
    J = sqrt(determinant)
    determinant⁻¹ = inv(determinant)
    g¹¹ = g₂₂ * determinant⁻¹
    g¹² = - g₁₂ * determinant⁻¹
    g²² = g₁₁ * determinant⁻¹
    G¹¹ = J * g¹¹
    G¹² = J * g¹²
    G²² = J * g²²

    return J, g₁₁, g₁₂, g₂₂, g¹¹, g¹², g²², G¹¹, G¹², G²²
end

@inline _metric_tensor_from_mapping(i, j, radius, mapping, LX, LY, size, λ, φ) =
    _metric_tensor_from_coordinates(i, j, radius, λ, φ)

function _fill_center_metric_location!(J, g₁₁, g₁₂, g₂₂, g¹¹, g¹², g²², G¹¹, G¹², G²²,
                                       radius, λ, φ, mapping, LX, LY, size)
    for j in axes(J, 2), i in axes(J, 1)
        Jᵢⱼ, g₁₁ᵢⱼ, g₁₂ᵢⱼ, g₂₂ᵢⱼ, g¹¹ᵢⱼ, g¹²ᵢⱼ, g²²ᵢⱼ, G¹¹ᵢⱼ, G¹²ᵢⱼ, G²²ᵢⱼ =
            _metric_tensor_from_mapping(i, j, radius, mapping, LX, LY, size, λ, φ)
        J[i, j] = Jᵢⱼ
        g₁₁[i, j] = g₁₁ᵢⱼ
        g₁₂[i, j] = g₁₂ᵢⱼ
        g₂₂[i, j] = g₂₂ᵢⱼ
        g¹¹[i, j] = g¹¹ᵢⱼ
        g¹²[i, j] = g¹²ᵢⱼ
        g²²[i, j] = g²²ᵢⱼ
        G¹¹[i, j] = G¹¹ᵢⱼ
        G¹²[i, j] = G¹²ᵢⱼ
        G²²[i, j] = G²²ᵢⱼ
    end

    return nothing
end

function _fill_xface_metric_location!(J, g¹¹, g¹², G¹¹, G¹²,
                                      radius, λ, φ, mapping, LX, LY, size)
    for j in axes(J, 2), i in axes(J, 1)
        Jᵢⱼ, _, _, _, g¹¹ᵢⱼ, g¹²ᵢⱼ, _, G¹¹ᵢⱼ, G¹²ᵢⱼ, _ =
            _metric_tensor_from_mapping(i, j, radius, mapping, LX, LY, size, λ, φ)
        J[i, j] = Jᵢⱼ
        g¹¹[i, j] = g¹¹ᵢⱼ
        g¹²[i, j] = g¹²ᵢⱼ
        G¹¹[i, j] = G¹¹ᵢⱼ
        G¹²[i, j] = G¹²ᵢⱼ
    end

    return nothing
end

function _fill_yface_metric_location!(J, g²¹, g²², G²¹, G²²,
                                      radius, λ, φ, mapping, LX, LY, size)
    for j in axes(J, 2), i in axes(J, 1)
        Jᵢⱼ, _, _, _, _, g¹²ᵢⱼ, g²²ᵢⱼ, _, G¹²ᵢⱼ, G²²ᵢⱼ =
            _metric_tensor_from_mapping(i, j, radius, mapping, LX, LY, size, λ, φ)
        J[i, j] = Jᵢⱼ
        g²¹[i, j] = g¹²ᵢⱼ
        g²²[i, j] = g²²ᵢⱼ
        G²¹[i, j] = G¹²ᵢⱼ
        G²²[i, j] = G²²ᵢⱼ
    end

    return nothing
end

@inline _fill_octahealpix_cross_metrics!(mapping, args...) = nothing

function _fill_octahealpix_cross_metrics!(::OctaHEALPixMapping,
                                          g¹²ᶠᶜᵃ, G¹²ᶠᶜᵃ,
                                          g²¹ᶜᶠᵃ, G²¹ᶜᶠᵃ,
                                          g¹²ᶜᶜᵃ, G¹²ᶜᶜᵃ)
    half = convert(eltype(g¹²ᶜᶜᵃ), 1//2)
    center_i = axes(g¹²ᶜᶜᵃ, 1)
    center_j = axes(g¹²ᶜᶜᵃ, 2)

    for j in axes(g¹²ᶠᶜᵃ, 2), i in axes(g¹²ᶠᶜᵃ, 1)
        if (i - 1 in center_i) && (i in center_i)
            g¹²ᶠᶜᵃ[i, j] = half * (g¹²ᶜᶜᵃ[i-1, j] + g¹²ᶜᶜᵃ[i, j])
            G¹²ᶠᶜᵃ[i, j] = half * (G¹²ᶜᶜᵃ[i-1, j] + G¹²ᶜᶜᵃ[i, j])
        end
    end

    for j in axes(g²¹ᶜᶠᵃ, 2), i in axes(g²¹ᶜᶠᵃ, 1)
        if (j - 1 in center_j) && (j in center_j)
            g²¹ᶜᶠᵃ[i, j] = half * (g¹²ᶜᶜᵃ[i, j-1] + g¹²ᶜᶜᵃ[i, j])
            G²¹ᶜᶠᵃ[i, j] = half * (G¹²ᶜᶜᵃ[i, j-1] + G¹²ᶜᶜᵃ[i, j])
        end
    end

    return nothing
end

@inline function _spherical_shell_area(radius, mapping::EquiangularGnomonicCubedSpherePanel, i, j, LX, LY, size)
    Nx, Ny = size[1:2]
    α₁ = equiangular_gnomonic_coordinate(i,     Nx, Face(), mapping.α)
    α₂ = equiangular_gnomonic_coordinate(i + 1, Nx, Face(), mapping.α)
    β₁ = equiangular_gnomonic_coordinate(j,     Ny, Face(), mapping.β)
    β₂ = equiangular_gnomonic_coordinate(j + 1, Ny, Face(), mapping.β)
    return radius^2 * equiangular_gnomonic_panel_solid_angle(α₁, α₂, β₁, β₂)
end

@inline _spherical_shell_area(radius, mapping::OctaHEALPixMapping, i, j, LX, LY, size) =
    radius^2 * octahealpix_solid_angle(mapping)

@inline function _spherical_shell_area(radius, mapping, i, j, LX, LY, size)
    return _metric_tensor_from_coordinates(i, j, radius, λ, φ)[1]
end

function _fill_spacings_and_areas!(Δx, Δy, Az, radius, λ, φ, mapping, LX, LY, size)
    for j in axes(Az, 2), i in axes(Az, 1)
        _, g₁₁, _, g₂₂, _, _, _, _, _, _ = _metric_tensor_from_mapping(i, j, radius, mapping, LX, LY, size, λ, φ)
        Δx[i, j] = sqrt(max(g₁₁, eps(eltype(λ))))
        Δy[i, j] = sqrt(max(g₂₂, eps(eltype(λ))))
        Az[i, j] = _spherical_shell_area(radius, mapping, i, j, LX, LY, size)
    end

    return nothing
end

function _build_spherical_shell_metrics(FT, radius, mapping,
                                        λᶜᶜᵃ, φᶜᶜᵃ,
                                        λᶠᶜᵃ, φᶠᶜᵃ,
                                        λᶜᶠᵃ, φᶜᶠᵃ,
                                        topo, size, halo)
    h_size = size[1:2]
    h_halo = halo[1:2]

    xᶜᶜᵃ = _new_horizontal_data(FT, (Center, Center), topo, h_size, h_halo)
    yᶜᶜᵃ = _new_horizontal_data(FT, (Center, Center), topo, h_size, h_halo)
    zᶜᶜᵃ = _new_horizontal_data(FT, (Center, Center), topo, h_size, h_halo)

    Jᶜᶜᵃ = _new_horizontal_data(FT, (Center, Center), topo, h_size, h_halo)
    Jᶠᶜᵃ = _new_horizontal_data(FT, (Face, Center), topo, h_size, h_halo)
    Jᶜᶠᵃ = _new_horizontal_data(FT, (Center, Face), topo, h_size, h_halo)

    g₁₁ᶜᶜᵃ = _new_horizontal_data(FT, (Center, Center), topo, h_size, h_halo)
    g₁₂ᶜᶜᵃ = _new_horizontal_data(FT, (Center, Center), topo, h_size, h_halo)
    g₂₂ᶜᶜᵃ = _new_horizontal_data(FT, (Center, Center), topo, h_size, h_halo)

    g¹¹ᶜᶜᵃ = _new_horizontal_data(FT, (Center, Center), topo, h_size, h_halo)
    g¹²ᶜᶜᵃ = _new_horizontal_data(FT, (Center, Center), topo, h_size, h_halo)
    g²¹ᶜᶜᵃ = _new_horizontal_data(FT, (Center, Center), topo, h_size, h_halo)
    g²²ᶜᶜᵃ = _new_horizontal_data(FT, (Center, Center), topo, h_size, h_halo)

    g¹¹ᶠᶜᵃ = _new_horizontal_data(FT, (Face, Center), topo, h_size, h_halo)
    g¹²ᶠᶜᵃ = _new_horizontal_data(FT, (Face, Center), topo, h_size, h_halo)
    g²¹ᶜᶠᵃ = _new_horizontal_data(FT, (Center, Face), topo, h_size, h_halo)
    g²²ᶜᶠᵃ = _new_horizontal_data(FT, (Center, Face), topo, h_size, h_halo)

    G¹¹ᶜᶜᵃ = _new_horizontal_data(FT, (Center, Center), topo, h_size, h_halo)
    G¹²ᶜᶜᵃ = _new_horizontal_data(FT, (Center, Center), topo, h_size, h_halo)
    G²¹ᶜᶜᵃ = _new_horizontal_data(FT, (Center, Center), topo, h_size, h_halo)
    G²²ᶜᶜᵃ = _new_horizontal_data(FT, (Center, Center), topo, h_size, h_halo)

    G¹¹ᶠᶜᵃ = _new_horizontal_data(FT, (Face, Center), topo, h_size, h_halo)
    G¹²ᶠᶜᵃ = _new_horizontal_data(FT, (Face, Center), topo, h_size, h_halo)
    G²¹ᶜᶠᵃ = _new_horizontal_data(FT, (Center, Face), topo, h_size, h_halo)
    G²²ᶜᶠᵃ = _new_horizontal_data(FT, (Center, Face), topo, h_size, h_halo)

    for j in axes(xᶜᶜᵃ, 2), i in axes(xᶜᶜᵃ, 1)
        x, y, z = _cartesian_from_lonlat(radius, λᶜᶜᵃ[i, j], φᶜᶜᵃ[i, j])
        xᶜᶜᵃ[i, j] = x
        yᶜᶜᵃ[i, j] = y
        zᶜᶜᵃ[i, j] = z
    end

    _fill_center_metric_location!(Jᶜᶜᵃ, g₁₁ᶜᶜᵃ, g₁₂ᶜᶜᵃ, g₂₂ᶜᶜᵃ,
                                  g¹¹ᶜᶜᵃ, g¹²ᶜᶜᵃ, g²²ᶜᶜᵃ,
                                  G¹¹ᶜᶜᵃ, G¹²ᶜᶜᵃ, G²²ᶜᶜᵃ,
                                  radius, λᶜᶜᵃ, φᶜᶜᵃ, mapping, Center(), Center(), size)

    _fill_xface_metric_location!(Jᶠᶜᵃ, g¹¹ᶠᶜᵃ, g¹²ᶠᶜᵃ, G¹¹ᶠᶜᵃ, G¹²ᶠᶜᵃ,
                                 radius, λᶠᶜᵃ, φᶠᶜᵃ, mapping, Face(), Center(), size)

    _fill_yface_metric_location!(Jᶜᶠᵃ, g²¹ᶜᶠᵃ, g²²ᶜᶠᵃ, G²¹ᶜᶠᵃ, G²²ᶜᶠᵃ,
                                 radius, λᶜᶠᵃ, φᶜᶠᵃ, mapping, Center(), Face(), size)

    _fill_octahealpix_cross_metrics!(mapping,
                                     g¹²ᶠᶜᵃ, G¹²ᶠᶜᵃ,
                                     g²¹ᶜᶠᵃ, G²¹ᶜᶠᵃ,
                                     g¹²ᶜᶜᵃ, G¹²ᶜᶜᵃ)

    return SphericalShellMetrics(xᶜᶜᵃ, yᶜᶜᵃ, zᶜᶜᵃ,
                                 Jᶜᶜᵃ, Jᶠᶜᵃ, Jᶜᶠᵃ,
                                 g₁₁ᶜᶜᵃ, g₁₂ᶜᶜᵃ, g₂₂ᶜᶜᵃ,
                                 g¹¹ᶜᶜᵃ, g¹²ᶜᶜᵃ, g²²ᶜᶜᵃ,
                                 g¹¹ᶠᶜᵃ, g¹²ᶠᶜᵃ, g²¹ᶜᶠᵃ, g²²ᶜᶠᵃ,
                                 G¹¹ᶜᶜᵃ, G¹²ᶜᶜᵃ, G²²ᶜᶜᵃ,
                                 G¹¹ᶠᶜᵃ, G¹²ᶠᶜᵃ, G²¹ᶜᶠᵃ, G²²ᶜᶠᵃ)
end

function _default_spherical_shell_topology(mapping)
    return mapping isa OctaHEALPixMapping ? (QuadFolded, QuadFolded, Bounded) : (Bounded, Bounded, Bounded)
end

function _default_spherical_shell_size(mapping, size)
    if isnothing(size)
        if mapping isa OctaHEALPixMapping
            N = 2 * mapping.N
            return (N, N, 1)
        else
            throw(ArgumentError("SphericalShellGrid requires `size` unless `mapping isa OctaHEALPixMapping`."))
        end
    end

    return length(size) == 2 ? (size..., 1) : size
end

function _deflate_flat_spherical_shell_tuple(TX, TY, TZ, tuple)
    if length(tuple) == 3 && topological_tuple_length(TX, TY, TZ) != 3
        return deflate_tuple(TX, TY, TZ, tuple)
    end

    return tuple
end

function SphericalShellGrid(arch::AbstractArchitecture = CPU(),
                            FT::DataType = Oceananigans.defaults.FloatType;
                            mapping,
                            size = nothing,
                            z = nothing,
                            radius = Oceananigans.defaults.planet_radius,
                            halo = (3, 3, 3),
                            topology = _default_spherical_shell_topology(mapping),
                            connectivity = mapping isa OctaHEALPixMapping ? OctaHEALPixConnectivity(mapping) : mapping)
    TX, TY, TZ = validate_topology(topology)
    size = _default_spherical_shell_size(mapping, size)
    size = _deflate_flat_spherical_shell_tuple(TX, TY, TZ, size)
    halo = _deflate_flat_spherical_shell_tuple(TX, TY, TZ, halo)
    size = validate_size(TX, TY, TZ, size)
    z = validate_dimension_specification(TZ, z, :z, size[3], FT)
    halo = validate_halo(TX, TY, TZ, size, halo)

    Nx, Ny, Nz = size
    Hx, Hy, Hz = halo
    h_size = (Nx, Ny)
    h_halo = (Hx, Hy)
    h_topo = (TX, TY)

    Lz, z = generate_coordinate(FT, (TX, TY, TZ), size, halo, z, :z, 3, CPU())

    λᶜᶜᵃ = _new_horizontal_data(FT, (Center, Center), h_topo, h_size, h_halo)
    λᶠᶜᵃ = _new_horizontal_data(FT, (Face, Center), h_topo, h_size, h_halo)
    λᶜᶠᵃ = _new_horizontal_data(FT, (Center, Face), h_topo, h_size, h_halo)
    λᶠᶠᵃ = _new_horizontal_data(FT, (Face, Face), h_topo, h_size, h_halo)

    φᶜᶜᵃ = _new_horizontal_data(FT, (Center, Center), h_topo, h_size, h_halo)
    φᶠᶜᵃ = _new_horizontal_data(FT, (Face, Center), h_topo, h_size, h_halo)
    φᶜᶠᵃ = _new_horizontal_data(FT, (Center, Face), h_topo, h_size, h_halo)
    φᶠᶠᵃ = _new_horizontal_data(FT, (Face, Face), h_topo, h_size, h_halo)

    _fill_horizontal_coordinates!(λᶜᶜᵃ, φᶜᶜᵃ, mapping, Center, Center, h_topo, h_size)
    _fill_horizontal_coordinates!(λᶠᶜᵃ, φᶠᶜᵃ, mapping, Face, Center, h_topo, h_size)
    _fill_horizontal_coordinates!(λᶜᶠᵃ, φᶜᶠᵃ, mapping, Center, Face, h_topo, h_size)
    _fill_horizontal_coordinates!(λᶠᶠᵃ, φᶠᶠᵃ, mapping, Face, Face, h_topo, h_size)

    Δxᶜᶜᵃ = _new_horizontal_data(FT, (Center, Center), h_topo, h_size, h_halo)
    Δxᶠᶜᵃ = _new_horizontal_data(FT, (Face, Center), h_topo, h_size, h_halo)
    Δxᶜᶠᵃ = _new_horizontal_data(FT, (Center, Face), h_topo, h_size, h_halo)
    Δxᶠᶠᵃ = _new_horizontal_data(FT, (Face, Face), h_topo, h_size, h_halo)
    Δyᶜᶜᵃ = _new_horizontal_data(FT, (Center, Center), h_topo, h_size, h_halo)
    Δyᶠᶜᵃ = _new_horizontal_data(FT, (Face, Center), h_topo, h_size, h_halo)
    Δyᶜᶠᵃ = _new_horizontal_data(FT, (Center, Face), h_topo, h_size, h_halo)
    Δyᶠᶠᵃ = _new_horizontal_data(FT, (Face, Face), h_topo, h_size, h_halo)
    Azᶜᶜᵃ = _new_horizontal_data(FT, (Center, Center), h_topo, h_size, h_halo)
    Azᶠᶜᵃ = _new_horizontal_data(FT, (Face, Center), h_topo, h_size, h_halo)
    Azᶜᶠᵃ = _new_horizontal_data(FT, (Center, Face), h_topo, h_size, h_halo)
    Azᶠᶠᵃ = _new_horizontal_data(FT, (Face, Face), h_topo, h_size, h_halo)

    radius = FT(radius)
    _fill_spacings_and_areas!(Δxᶜᶜᵃ, Δyᶜᶜᵃ, Azᶜᶜᵃ, radius, λᶜᶜᵃ, φᶜᶜᵃ, mapping, Center(), Center(), size)
    _fill_spacings_and_areas!(Δxᶠᶜᵃ, Δyᶠᶜᵃ, Azᶠᶜᵃ, radius, λᶠᶜᵃ, φᶠᶜᵃ, mapping, Face(), Center(), size)
    _fill_spacings_and_areas!(Δxᶜᶠᵃ, Δyᶜᶠᵃ, Azᶜᶠᵃ, radius, λᶜᶠᵃ, φᶜᶠᵃ, mapping, Center(), Face(), size)
    _fill_spacings_and_areas!(Δxᶠᶠᵃ, Δyᶠᶠᵃ, Azᶠᶠᵃ, radius, λᶠᶠᵃ, φᶠᶠᵃ, mapping, Face(), Face(), size)

    metrics = _build_spherical_shell_metrics(FT, radius, mapping,
                                             λᶜᶜᵃ, φᶜᶜᵃ,
                                             λᶠᶜᵃ, φᶠᶜᵃ,
                                             λᶜᶠᵃ, φᶜᶠᵃ,
                                             h_topo, size, halo)

    grid = SphericalShellGrid{FT, TX, TY, TZ, typeof(z), typeof(connectivity), typeof(mapping), typeof(metrics),
                              typeof(λᶜᶜᵃ), typeof(λᶠᶜᵃ), typeof(λᶜᶠᵃ), typeof(λᶠᶠᵃ), CPU, typeof(Nx)}(
                                  CPU(), Nx, Ny, Nz, Hx, Hy, Hz,
                                  FT(Nx), FT(Ny), FT(Lz), z, radius,
                                  connectivity, mapping, metrics,
                                  λᶜᶜᵃ, λᶠᶜᵃ, λᶜᶠᵃ, λᶠᶠᵃ,
                                  φᶜᶜᵃ, φᶠᶜᵃ, φᶜᶠᵃ, φᶠᶠᵃ,
                                  Δxᶜᶜᵃ, Δxᶠᶜᵃ, Δxᶜᶠᵃ, Δxᶠᶠᵃ,
                                  Δyᶜᶜᵃ, Δyᶠᶜᵃ, Δyᶜᶠᵃ, Δyᶠᶠᵃ,
                                  Azᶜᶜᵃ, Azᶠᶜᵃ, Azᶜᶠᵃ, Azᶠᶠᵃ)

    return arch isa CPU ? grid : Architectures.on_architecture(arch, grid)
end

SphericalShellGrid(FT::DataType; kwargs...) = SphericalShellGrid(CPU(), FT; kwargs...)

function _spherical_shell_grid_reconstruct(grid, arch, connectivity, metrics,
                                           z,
                                           λᶜᶜᵃ, λᶠᶜᵃ, λᶜᶠᵃ, λᶠᶠᵃ,
                                           φᶜᶜᵃ, φᶠᶜᵃ, φᶜᶠᵃ, φᶠᶠᵃ,
                                           Δxᶜᶜᵃ, Δxᶠᶜᵃ, Δxᶜᶠᵃ, Δxᶠᶠᵃ,
                                           Δyᶜᶜᵃ, Δyᶠᶜᵃ, Δyᶜᶠᵃ, Δyᶠᶠᵃ,
                                           Azᶜᶜᵃ, Azᶠᶜᵃ, Azᶜᶠᵃ, Azᶠᶠᵃ)
    TX, TY, TZ = topology(grid)
    FT = eltype(grid)

    return SphericalShellGrid{FT, TX, TY, TZ, typeof(z), typeof(connectivity), typeof(grid.mapping), typeof(metrics),
                              typeof(λᶜᶜᵃ), typeof(λᶠᶜᵃ), typeof(λᶜᶠᵃ), typeof(λᶠᶠᵃ), typeof(arch), typeof(grid.Nx)}(
                                  arch, grid.Nx, grid.Ny, grid.Nz,
                                  grid.Hx, grid.Hy, grid.Hz,
                                  grid.Lx, grid.Ly, grid.Lz,
                                  z, grid.radius, connectivity, grid.mapping, metrics,
                                  λᶜᶜᵃ, λᶠᶜᵃ, λᶜᶠᵃ, λᶠᶠᵃ,
                                  φᶜᶜᵃ, φᶠᶜᵃ, φᶜᶠᵃ, φᶠᶠᵃ,
                                  Δxᶜᶜᵃ, Δxᶠᶜᵃ, Δxᶜᶠᵃ, Δxᶠᶠᵃ,
                                  Δyᶜᶜᵃ, Δyᶠᶜᵃ, Δyᶜᶠᵃ, Δyᶠᶠᵃ,
                                  Azᶜᶜᵃ, Azᶠᶜᵃ, Azᶜᶠᵃ, Azᶠᶠᵃ)
end

function Adapt.adapt_structure(to, grid::SphericalShellGrid)
    return _spherical_shell_grid_reconstruct(grid, nothing,
                                             adapt(to, grid.connectivity),
                                             adapt(to, grid.metrics),
                                             adapt(to, grid.z),
                                             adapt(to, grid.λᶜᶜᵃ), adapt(to, grid.λᶠᶜᵃ), adapt(to, grid.λᶜᶠᵃ), adapt(to, grid.λᶠᶠᵃ),
                                             adapt(to, grid.φᶜᶜᵃ), adapt(to, grid.φᶠᶜᵃ), adapt(to, grid.φᶜᶠᵃ), adapt(to, grid.φᶠᶠᵃ),
                                             adapt(to, grid.Δxᶜᶜᵃ), adapt(to, grid.Δxᶠᶜᵃ), adapt(to, grid.Δxᶜᶠᵃ), adapt(to, grid.Δxᶠᶠᵃ),
                                             adapt(to, grid.Δyᶜᶜᵃ), adapt(to, grid.Δyᶠᶜᵃ), adapt(to, grid.Δyᶜᶠᵃ), adapt(to, grid.Δyᶠᶠᵃ),
                                             adapt(to, grid.Azᶜᶜᵃ), adapt(to, grid.Azᶠᶜᵃ), adapt(to, grid.Azᶜᶠᵃ), adapt(to, grid.Azᶠᶠᵃ))
end

function Architectures.on_architecture(arch::AbstractSerialArchitecture, grid::SphericalShellGrid)
    arch == architecture(grid) && return grid

    return _spherical_shell_grid_reconstruct(grid, arch,
                                             on_architecture(arch, grid.connectivity),
                                             on_architecture(arch, grid.metrics),
                                             on_architecture(arch, grid.z),
                                             on_architecture(arch, grid.λᶜᶜᵃ), on_architecture(arch, grid.λᶠᶜᵃ), on_architecture(arch, grid.λᶜᶠᵃ), on_architecture(arch, grid.λᶠᶠᵃ),
                                             on_architecture(arch, grid.φᶜᶜᵃ), on_architecture(arch, grid.φᶠᶜᵃ), on_architecture(arch, grid.φᶜᶠᵃ), on_architecture(arch, grid.φᶠᶠᵃ),
                                             on_architecture(arch, grid.Δxᶜᶜᵃ), on_architecture(arch, grid.Δxᶠᶜᵃ), on_architecture(arch, grid.Δxᶜᶠᵃ), on_architecture(arch, grid.Δxᶠᶠᵃ),
                                             on_architecture(arch, grid.Δyᶜᶜᵃ), on_architecture(arch, grid.Δyᶠᶜᵃ), on_architecture(arch, grid.Δyᶜᶠᵃ), on_architecture(arch, grid.Δyᶠᶠᵃ),
                                             on_architecture(arch, grid.Azᶜᶜᵃ), on_architecture(arch, grid.Azᶠᶜᵃ), on_architecture(arch, grid.Azᶜᶠᵃ), on_architecture(arch, grid.Azᶠᶠᵃ))
end

function Base.summary(grid::SphericalShellGrid)
    FT = eltype(grid)
    TX, TY, TZ = topology(grid)
    return string(size_summary(grid),
                  " SphericalShellGrid{$FT, $TX, $TY, $TZ} on ", summary(architecture(grid)),
                  " with ", size_summary(halo_size(grid)), " halo")
end

function Base.show(io::IO, grid::SphericalShellGrid, withsummary=true)
    if withsummary
        print(io, summary(grid), "\n")
    end

    return print(io,
                 "├── mapping: ", summary(grid.mapping), "\n",
                 "├── radius:  ", prettysummary(grid.radius), "\n",
                 "└── z:       ", summary(grid.z))
end

function constructor_arguments(grid::SphericalShellGrid)
    args = OrderedDict(:architecture => architecture(grid), :number_type => eltype(grid))
    kwargs = Dict(:mapping => grid.mapping,
                  :size => size(grid),
                  :halo => halo_size(grid),
                  :z => cpu_face_constructor_z(grid),
                  :radius => grid.radius,
                  :topology => topology(grid),
                  :connectivity => grid.connectivity)
    return args, kwargs
end

function Base.similar(grid::SphericalShellGrid)
    args, kwargs = constructor_arguments(grid)
    return SphericalShellGrid(args[:architecture], args[:number_type]; kwargs...)
end

function with_number_type(FT, grid::SphericalShellGrid)
    args, kwargs = constructor_arguments(grid)
    return SphericalShellGrid(args[:architecture], FT; kwargs...)
end

function with_halo(halo, grid::SphericalShellGrid)
    args, kwargs = constructor_arguments(grid)
    kwargs[:halo] = halo
    return SphericalShellGrid(args[:architecture], args[:number_type]; kwargs...)
end

function nodes(grid::SphericalShellGrid, ℓx, ℓy, ℓz; reshape=false, with_halos=false, indices=(Colon(), Colon(), Colon()))
    λ = λnodes(grid, ℓx, ℓy, ℓz; with_halos, indices=indices[1:2])
    φ = φnodes(grid, ℓx, ℓy, ℓz; with_halos, indices=indices[1:2])
    z = rnodes(grid, ℓz; with_halos, indices=indices[3])

    if reshape
        λ = Base.reshape(λ, size(λ, 1), size(λ, 2), 1)
        φ = Base.reshape(φ, size(φ, 1), size(φ, 2), 1)
        z = Base.reshape(z, 1, 1, length(z))
    end

    return λ, φ, z
end

@inline _λ_array(grid, ::Center, ::Center) = grid.λᶜᶜᵃ
@inline _λ_array(grid, ::Face,   ::Center) = grid.λᶠᶜᵃ
@inline _λ_array(grid, ::Center, ::Face)   = grid.λᶜᶠᵃ
@inline _λ_array(grid, ::Face,   ::Face)   = grid.λᶠᶠᵃ
@inline _φ_array(grid, ::Center, ::Center) = grid.φᶜᶜᵃ
@inline _φ_array(grid, ::Face,   ::Center) = grid.φᶠᶜᵃ
@inline _φ_array(grid, ::Center, ::Face)   = grid.φᶜᶠᵃ
@inline _φ_array(grid, ::Face,   ::Face)   = grid.φᶠᶠᵃ

@inline λnodes(grid::SphericalShellGrid, ℓx, ℓy, ℓz; with_halos=false, indices=(Colon(), Colon())) =
    view(_property(_λ_array(grid, ℓx, ℓy), ℓx, ℓy, topology(grid, 1), topology(grid, 2), grid.Nx, grid.Ny, grid.Hx, grid.Hy, with_halos), indices...)
@inline φnodes(grid::SphericalShellGrid, ℓx, ℓy, ℓz; with_halos=false, indices=(Colon(), Colon())) =
    view(_property(_φ_array(grid, ℓx, ℓy), ℓx, ℓy, topology(grid, 1), topology(grid, 2), grid.Nx, grid.Ny, grid.Hx, grid.Hy, with_halos), indices...)

@inline λnodes(grid::SphericalShellGrid, ℓx, ℓy; kwargs...) = λnodes(grid, ℓx, ℓy, Center(); kwargs...)
@inline φnodes(grid::SphericalShellGrid, ℓx, ℓy; kwargs...) = φnodes(grid, ℓx, ℓy, Center(); kwargs...)

@inline λnode(i, j, grid::SphericalShellGrid, ℓx, ℓy) = @inbounds _λ_array(grid, ℓx, ℓy)[i, j]
@inline φnode(i, j, grid::SphericalShellGrid, ℓx, ℓy) = @inbounds _φ_array(grid, ℓx, ℓy)[i, j]
@inline λnode(i, j, k, grid::SphericalShellGrid, ℓx, ℓy, ℓz) = λnode(i, j, grid, ℓx, ℓy)
@inline φnode(i, j, k, grid::SphericalShellGrid, ℓx, ℓy, ℓz) = φnode(i, j, grid, ℓx, ℓy)
@inline ξnode(i, j, k, grid::SphericalShellGrid, ℓx, ℓy, ℓz) = λnode(i, j, grid, ℓx, ℓy)
@inline ηnode(i, j, k, grid::SphericalShellGrid, ℓx, ℓy, ℓz) = φnode(i, j, grid, ℓx, ℓy)
@inline rnode(i, j, k, grid::SphericalShellGrid, ℓx, ℓy, ℓz) = rnode(k, grid, ℓz)

ξname(::SphericalShellGrid) = :λ
ηname(::SphericalShellGrid) = :φ
rname(::SphericalShellGrid) = :z

@inline xnode(i, j, grid::SphericalShellGrid, ℓx, ℓy) = grid.radius * deg2rad(λnode(i, j, grid, ℓx, ℓy)) * hack_cosd(φnode(i, j, grid, ℓx, ℓy))
@inline ynode(i, j, grid::SphericalShellGrid, ℓx, ℓy) = grid.radius * deg2rad(φnode(i, j, grid, ℓx, ℓy))

@inline function spherical_shell_tangent_basis(i, j, k, grid::SphericalShellGrid, LX, LY, LZ)
    return spherical_shell_tangent_basis(λnode(i, j, grid, LX, LY), φnode(i, j, grid, LX, LY))
end

@inline _spherical_shell_radial_offset(radius, z) = z
@inline _spherical_shell_radial_offset(radius, ::Nothing) = zero(radius)

@inline function spherical_shell_cartesian_node(i, j, k, grid::SphericalShellGrid, LX, LY, LZ)
    λ = λnode(i, j, grid, LX, LY)
    φ = φnode(i, j, grid, LX, LY)
    z = rnode(k, grid, LZ)
    r = grid.radius + _spherical_shell_radial_offset(grid.radius, z)
    return _cartesian_from_lonlat(r, λ, φ)
end

@inline function horizontal_spherical_shell_metric_tensor(i, j, grid::SphericalShellGrid, LX, LY)
    λ = _λ_array(grid, LX, LY)
    φ = _φ_array(grid, LX, LY)
    return _metric_tensor_from_mapping(i, j, grid.radius, grid.mapping, LX, LY, size(grid), λ, φ)
end

@inline horizontal_spherical_shell_metric_tensor(i, j, k, grid::SphericalShellGrid, LX, LY, LZ) =
    horizontal_spherical_shell_metric_tensor(i, j, grid, LX, LY)
@inline spherical_shell_metric_tensor(args...) = horizontal_spherical_shell_metric_tensor(args...)
@inline octahealpix_metric_tensor(args...) = horizontal_spherical_shell_metric_tensor(args...)

for LX in (:ᶜ, :ᶠ), LY in (:ᶜ, :ᶠ)
    Δx = Symbol(:Δx, LX, LY, :ᵃ)
    Δy = Symbol(:Δy, LX, LY, :ᵃ)
    Az = Symbol(:Az, LX, LY, :ᵃ)
    Δx_field = Symbol(:Δx, LX, LY, :ᵃ)
    Δy_field = Symbol(:Δy, LX, LY, :ᵃ)
    Az_field = Symbol(:Az, LX, LY, :ᵃ)

    @eval begin
        @inline $Δx(i, j, k, grid::SphericalShellGrid) = @inbounds grid.$Δx_field[i, j]
        @inline $Δy(i, j, k, grid::SphericalShellGrid) = @inbounds grid.$Δy_field[i, j]
        @inline $Az(i, j, k, grid::SphericalShellGrid) = @inbounds grid.$Az_field[i, j]
    end
end

@inline xspacings(grid::SphericalShellGrid, loc...; kwargs...) = grid.Δxᶜᶜᵃ
@inline yspacings(grid::SphericalShellGrid, loc...; kwargs...) = grid.Δyᶜᶜᵃ
@inline λspacings(grid::SphericalShellGrid, loc...; kwargs...) = grid.Δxᶜᶜᵃ ./ grid.radius
@inline φspacings(grid::SphericalShellGrid, loc...; kwargs...) = grid.Δyᶜᶜᵃ ./ grid.radius
