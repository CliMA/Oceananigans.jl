using Oceananigans.BoundaryConditions
using Oceananigans.BoundaryConditions: fill_halo_size, fill_halo_offset
using Oceananigans.Fields: reduced_dimensions
using Oceananigans.MultiRegion: number_of_regions

import Oceananigans.BoundaryConditions: fill_halo_regions!

function fill_halo_regions!(field::CubedSphereField{<:Center, <:Center}; kwargs...)
    grid = field.grid

    multiregion_field = Reference(field.data.regional_objects)
    regions = Iterate(1:6)

    @apply_regionally fill_cubed_sphere_field_halo_event!(grid, field, multiregion_field, regions,
        grid.connectivity.connections, WestAndEast(),
        _fill_cubed_sphere_center_center_field_east_west_halo_regions!)
    @apply_regionally fill_cubed_sphere_field_halo_event!(grid, field, multiregion_field, regions,
        grid.connectivity.connections, SouthAndNorth(),
        _fill_cubed_sphere_center_center_field_north_south_halo_regions!)

    return nothing
end

@inline function fill_cubed_sphere_field_halo_event!(grid, field, multiregion_field, region, connections,
                                                     side, _fill_halo_kernel!)

    sz = fill_halo_size(field.data, side, field.indices, FullyConnected, location(field), grid)
    of = fill_halo_offset(sz, side, field.indices)
    kernel_parameters = KernelParameters(sz, of)
    reduced_dims = reduced_dimensions(field)

    return launch!(grid.architecture, grid, kernel_parameters, _fill_halo_kernel!, field, multiregion_field, region,
                   connections, grid.Nx, grid.Hx; reduced_dimensions = reduced_dims)
end

@kernel function _fill_cubed_sphere_center_center_field_east_west_halo_regions!(field, multiregion_field, region,
                                                                                connections, Nc, Hc)
    j, k = @index(Global, NTuple)
    region_E = connections.east.from_rank
    region_W = connections.west.from_rank

    # The commented blocks below show the equivalent non-GPU vectorized implementation, which can be useful for visually
    # verifying halo filling against schematics or physical cubed sphere models.

    #- E + W Halo for field:
    for i in 1:Hc
        if isodd(region)
            @inbounds begin
                #=
                field[region][Nc+1:Nc+Hc, 1:Nc, k] .=         field[region_E][1:Hc, 1:Nc, k]
                field[region][1-Hc:0, 1:Nc, k]     .= reverse(field[region_W][1:Nc, Nc+1-Hc:Nc, k], dims=1)'
                =#
                field[Nc+i, j, k] = multiregion_field[region_E][i, j, k]
                field[i-Hc, j, k] = multiregion_field[region_W][Nc+1-j, Nc+i-Hc, k]
            end
        elseif iseven(region)
            @inbounds begin
                #=
                field[region][Nc+1:Nc+Hc, 1:Nc, k] .= reverse(field[region_E][1:Nc, 1:Hc, k], dims=1)'
                field[region][1-Hc:0, 1:Nc, k]     .=         field[region_W][Nc+1-Hc:Nc, 1:Nc, k]
                =#
                field[Nc+i, j, k] = multiregion_field[region_E][Nc+1-j, i, k]
                field[i-Hc, j, k] = multiregion_field[region_W][Nc+i-Hc, j, k]
            end
        end
    end
end

@kernel function _fill_cubed_sphere_center_center_field_north_south_halo_regions!(field, multiregion_field, region,
                                                                                  connections, Nc, Hc)
    i, k = @index(Global, NTuple)
    region_N = connections.north.from_rank
    region_S = connections.south.from_rank

    # The commented blocks below show the equivalent non-GPU vectorized implementation, which can be useful for visually
    # verifying halo filling against schematics or physical cubed sphere models.

    #- N + S Halo for field:
    for j in 1:Hc
        if isodd(region)
            @inbounds begin
                #=
                field[region][1:Nc, Nc+1:Nc+Hc, k] .= reverse(field[region_N][1:Hc, 1:Nc, k], dims=2)'
                field[region][1:Nc, 1-Hc:0, k]     .=         field[region_S][1:Nc, Nc+1-Hc:Nc, k]
                =#
                field[i, Nc+j, k] = multiregion_field[region_N][j, Nc+1-i, k]
                field[i, j-Hc, k] = multiregion_field[region_S][i, Nc+j-Hc, k]
            end
        elseif iseven(region)
            @inbounds begin
                #=
                field[region][1:Nc, Nc+1:Nc+Hc, k] .=         field[region_N][1:Nc, 1:Hc, k]
                field[region][1:Nc, 1-Hc:0, k]     .= reverse(field[region_S][Nc+1-Hc:Nc, 1:Nc, k], dims=2)'
                =#
                field[i, Nc+j, k] = multiregion_field[region_N][i, j, k]
                field[i, j-Hc, k] = multiregion_field[region_S][Nc+j-Hc, Nc+1-i, k]
            end
        end
    end
end

function fill_halo_regions!(field::CubedSphereField{<:Face, <:Face}; kwargs...)
    grid = field.grid

    multiregion_field = Reference(field.data.regional_objects)
    regions = Iterate(1:6)

    @apply_regionally fill_cubed_sphere_field_halo_event!(grid, field, multiregion_field, regions,
        grid.connectivity.connections, WestAndEast(),
        _fill_cubed_sphere_face_face_field_east_west_halo_regions!)
    @apply_regionally fill_cubed_sphere_field_halo_event!(grid, field, multiregion_field, regions,
        grid.connectivity.connections, SouthAndNorth(),
        _fill_cubed_sphere_face_face_field_north_south_halo_regions!)

    return nothing
end

@kernel function _fill_cubed_sphere_face_face_field_east_west_halo_regions!(field, multiregion_field, region,
                                                                            connections, Nc, Hc)
    j, k = @index(Global, NTuple)
    region_E = connections.east.from_rank
    region_W = connections.west.from_rank
    region_S = connections.south.from_rank

    # The commented blocks below show the equivalent non-GPU vectorized implementation, which can be useful for visually
    # verifying halo filling against schematics or physical cubed sphere models.

    #- E + W Halo for field:
    for i in 1:Hc
        if isodd(region)
            @inbounds begin
                #=
                field[region][Nc+1:Nc+Hc, 1:Nc, k]   .=         field[region_E][1:Hc, 1:Nc, k]
                field[region][1-Hc:0, 2:Nc+1, k]     .= reverse(field[region_W][1:Nc, Nc+1-Hc:Nc, k], dims=1)'
                field[region][1-Hc:0, 1, k]          .=         field[region_S][1, Nc+1-Hc:Nc, k]
                =#
                field[Nc+i, j, k]   = multiregion_field[region_E][i, j, k]
                field[i-Hc, j+1, k] = multiregion_field[region_W][Nc+1-j, Nc+i-Hc, k]
                field[i-Hc, 1, k]   = multiregion_field[region_S][1, Nc+i-Hc, k]
            end
        elseif iseven(region)
            @inbounds begin
                #=
                field[region][Nc+1:Nc+Hc, 2:Nc, k]   .= reverse(field[region_E][2:Nc, 1:Hc, k], dims=1)'
                if Hc > 1
                    field[region][Nc+2:Nc+Hc, 1, k]  .= reverse(field[region_S][Nc+2-Hc:Nc, 1, k])
                end
                Note that the halo corresponding to the "missing" south-east corner of even panels, specifically
                field[region][Nc+1, 1, k], remains unfilled.
                =#
                j > 1 && (field[Nc+i, j, k] = multiregion_field[region_E][Nc+2-j, i, k])
                (Hc > 1 && i > 1) && (field[Nc+i, 1, k] = multiregion_field[region_S][Nc+2-i, 1, k])
                #=
                field[region][1-Hc:0, 1:Nc, k]       .=         field[region_W][Nc+1-Hc:Nc, 1:Nc, k]
                =#
                field[i-Hc, j, k] = multiregion_field[region_W][Nc+i-Hc, j, k]
            end
        end
    end
end

@kernel function _fill_cubed_sphere_face_face_field_north_south_halo_regions!(field, multiregion_field, region,
                                                                              connections, Nc, Hc)
    i, k = @index(Global, NTuple)
    region_E = connections.east.from_rank
    region_N = connections.north.from_rank
    region_W = connections.west.from_rank
    region_S = connections.south.from_rank
    
    # The commented blocks below show the equivalent non-GPU vectorized implementation, which can be useful for visually
    # verifying halo filling against schematics or physical cubed sphere models.

    #- N + S Halo for field:
    for j in 1:Hc
        if isodd(region)
            @inbounds begin
                #=
                field[region][2:Nc+1, Nc+1:Nc+Hc, k] .= reverse(field[region_N][1:Hc, 1:Nc, k], dims=2)'
                if Hc > 1
                    field[region][1, Nc+2:Nc+Hc, k]  .= reverse(field[region_W][1, Nc+2-Hc:Nc, k])'
                end
                Note that the halo corresponding to the "missing" north-west corner of odd panels, specifically
                field[region][1, Nc+1, k], remains unfilled.
                =#
                field[i+1, Nc+j, k] = multiregion_field[region_N][j, Nc+1-i, k]
                (Hc > 1 && j > 1) && (field[1, Nc+j, k] = multiregion_field[region_W][1, Nc+2-j, k])
                #=
                field[region][1:Nc, 1-Hc:0, k]       .=         field[region_S][1:Nc, Nc+1-Hc:Nc, k]
                field[region][Nc+1, 1-Hc:0, k]       .= reverse(field[region_E][2:Hc+1, 1, k])'
                =#
                field[i, j-Hc, k] = multiregion_field[region_S][i, Nc+j-Hc, k]
                field[Nc+1, j-Hc, k] = multiregion_field[region_E][Hc+2-j, 1, k]
            end
        elseif iseven(region)
            @inbounds begin
                #=
                field[region][1:Nc, Nc+1:Nc+Hc, k]   .=         field[region_N][1:Nc, 1:Hc, k]
                field[region][Nc+1, Nc+1:Nc+Hc, k]   .=         field[region_E][1, 1:Hc, k]'
                field[region][2:Nc+1, 1-Hc:0, k]     .= reverse(field[region_S][Nc+1-Hc:Nc, 1:Nc, k], dims=2)'
                field[region][1, 1-Hc:0, k]          .=         field[region_W][Nc+1-Hc:Nc, 1, k]'
                =#
                field[i, Nc+j, k] = multiregion_field[region_N][i, j, k]
                field[Nc+1, Nc+j, k] = multiregion_field[region_E][1, j, k]
                field[i+1, j-Hc, k] = multiregion_field[region_S][Nc+j-Hc, Nc+1-i, k]
                field[1, j-Hc, k] = multiregion_field[region_W][Nc+j-Hc, 1, k]
            end
        end
    end
end

fill_halo_regions!(fields::Tuple{CubedSphereField,CubedSphereField}; signed = true, kwargs...) = fill_halo_regions!(fields...; signed, kwargs...)

function fill_halo_regions!(field_1::CubedSphereField{<:Center, <:Center},
                            field_2::CubedSphereField{<:Center, <:Center}; signed = true, kwargs...)
    grid = field_1.grid

    signed ? plmn = -1 : plmn = 1

    multiregion_field_1 = Reference(field_1.data.regional_objects)
    multiregion_field_2 = Reference(field_2.data.regional_objects)
    regions = Iterate(1:6)

    @apply_regionally fill_cubed_sphere_field_pairs_halo_event!(grid, field_1, multiregion_field_1, field_2,
        multiregion_field_2, regions, grid.connectivity.connections, plmn, WestAndEast(),
        _fill_cubed_sphere_center_center_center_center_field_pairs_east_west_halo_regions!)
    @apply_regionally fill_cubed_sphere_field_pairs_halo_event!(grid, field_1, multiregion_field_1, field_2,
        multiregion_field_2, regions, grid.connectivity.connections, plmn, SouthAndNorth(),
        _fill_cubed_sphere_center_center_center_center_field_pairs_north_south_halo_regions!)

    return nothing
end

@inline function fill_cubed_sphere_field_pairs_halo_event!(grid, field_1, multiregion_field_1, field_2,
                                                           multiregion_field_2, region, connections, plmn,
                                                           side, _fill_halo_kernel!)
    
    sz = fill_halo_size(field_1.data, side, field_1.indices, FullyConnected, location(field_1), grid)
    of = fill_halo_offset(sz, side, field_1.indices)
    kernel_parameters = KernelParameters(sz, of)
    reduced_dims = reduced_dimensions(field_1)

    return launch!(grid.architecture, grid, kernel_parameters, _fill_halo_kernel!, field_1, multiregion_field_1,
                   field_2, multiregion_field_2, region, connections, grid.Nx, grid.Hx, plmn;
                   reduced_dimensions = reduced_dims)
end

@kernel function _fill_cubed_sphere_center_center_center_center_field_pairs_east_west_halo_regions!(
field_1, multiregion_field_1, field_2, multiregion_field_2, region, connections, Nc, Hc, plmn)
    j, k = @index(Global, NTuple)
    region_E = connections.east.from_rank
    region_W = connections.west.from_rank

    # The commented blocks below show the equivalent non-GPU vectorized implementation, which can be useful for visually
    # verifying halo filling against schematics or physical cubed sphere models.

    #- E + W Halo for field:
    for i in 1:Hc
        if isodd(region)
            @inbounds begin
                #=
                #- E Halo:
                field_1[region][Nc+1:Nc+Hc, 1:Nc, k] .=         field_1[region_E][1:Hc, 1:Nc, k]
                field_2[region][Nc+1:Nc+Hc, 1:Nc, k] .=         field_2[region_E][1:Hc, 1:Nc, k]
                =#
                field_1[Nc+i, j, k] = multiregion_field_1[region_E][i, j, k]
                field_2[Nc+i, j, k] = multiregion_field_2[region_E][i, j, k]
                #=
                #- W Halo:
                field_1[region][1-Hc:0, 1:Nc, k]     .= reverse(field_2[region_W][1:Nc, Nc+1-Hc:Nc, k], dims=1)'
                field_2[region][1-Hc:0, 1:Nc, k]     .= reverse(field_1[region_W][1:Nc, Nc+1-Hc:Nc, k], dims=1)' * plmn
                =#
                field_1[i-Hc, j, k] = multiregion_field_2[region_W][Nc+1-j, Nc+i-Hc, k]
                field_2[i-Hc, j, k] = multiregion_field_1[region_W][Nc+1-j, Nc+i-Hc, k] * plmn
            end
        elseif iseven(region)
            @inbounds begin
                #=
                #- E Halo:
                field_1[region][Nc+1:Nc+Hc, 1:Nc, k] .= reverse(field_2[region_E][1:Nc, 1:Hc, k], dims=1)'
                field_2[region][Nc+1:Nc+Hc, 1:Nc, k] .= reverse(field_1[region_E][1:Nc, 1:Hc, k], dims=1)' * plmn
                =#
                field_1[Nc+i, j, k] = multiregion_field_2[region_E][Nc+1-j, i, k]
                field_2[Nc+i, j, k] = multiregion_field_1[region_E][Nc+1-j, i, k] * plmn
                #=
                #- W Halo:
                field_1[region][1-Hc:0, 1:Nc, k]     .=         field_1[region_W][Nc+1-Hc:Nc, 1:Nc, k]
                field_2[region][1-Hc:0, 1:Nc, k]     .=         field_2[region_W][Nc+1-Hc:Nc, 1:Nc, k]
                =#
                field_1[i-Hc, j, k] = multiregion_field_1[region_W][Nc+i-Hc, j, k]
                field_2[i-Hc, j, k] = multiregion_field_2[region_W][Nc+i-Hc, j, k]
            end
        end
    end
end

@kernel function _fill_cubed_sphere_center_center_center_center_field_pairs_north_south_halo_regions!(
field_1, multiregion_field_1, field_2, multiregion_field_2, region, connections, Nc, Hc, plmn)
    i, k = @index(Global, NTuple)
    region_N = connections.north.from_rank
    region_S = connections.south.from_rank

    # The commented blocks below show the equivalent non-GPU vectorized implementation, which can be useful for visually
    # verifying halo filling against schematics or physical cubed sphere models.

    #- N + S Halo for field:
    for j in 1:Hc
        if isodd(region)
            @inbounds begin
                #=
                #- N Halo:
                field_1[region][1:Nc, Nc+1:Nc+Hc, k] .= reverse(field_2[region_N][1:Hc, 1:Nc, k], dims=2)' * plmn
                field_2[region][1:Nc, Nc+1:Nc+Hc, k] .= reverse(field_1[region_N][1:Hc, 1:Nc, k], dims=2)'
                =#
                field_1[i, Nc+j, k] = multiregion_field_2[region_N][j, Nc+1-i, k] * plmn
                field_2[i, Nc+j, k] = multiregion_field_1[region_N][j, Nc+1-i, k]
                #=
                #- S Halo:
                field_1[region][1:Nc, 1-Hc:0, k]     .=         field_1[region_S][1:Nc, Nc+1-Hc:Nc, k]
                field_2[region][1:Nc, 1-Hc:0, k]     .=         field_2[region_S][1:Nc, Nc+1-Hc:Nc, k]
                =#
                field_1[i, j-Hc, k] = multiregion_field_1[region_S][i, Nc+j-Hc, k]
                field_2[i, j-Hc, k] = multiregion_field_2[region_S][i, Nc+j-Hc, k]
            end
        elseif iseven(region)
            @inbounds begin
                #=
                #- N Halo:
                field_1[region][1:Nc, Nc+1:Nc+Hc, k] .=         field_1[region_N][1:Nc, 1:Hc, k]
                field_2[region][1:Nc, Nc+1:Nc+Hc, k] .=         field_2[region_N][1:Nc, 1:Hc, k]
                =#
                field_1[i, Nc+j, k] = multiregion_field_1[region_N][i, j, k]
                field_2[i, Nc+j, k] = multiregion_field_2[region_N][i, j, k]
                #=
                #- S Halo:
                field_1[region][1:Nc, 1-Hc:0, k]     .= reverse(field_2[region_S][Nc+1-Hc:Nc, 1:Nc, k], dims=2)' * plmn
                field_2[region][1:Nc, 1-Hc:0, k]     .= reverse(field_1[region_S][Nc+1-Hc:Nc, 1:Nc, k], dims=2)'
                =#
                field_1[i, j-Hc, k] = multiregion_field_2[region_S][Nc+j-Hc, Nc+1-i, k] * plmn
                field_2[i, j-Hc, k] = multiregion_field_1[region_S][Nc+j-Hc, Nc+1-i, k]
            end
        end
    end
end

function fill_halo_regions!(field_1::CubedSphereField{<:Face, <:Center},
                            field_2::CubedSphereField{<:Center, <:Face}; signed = true, kwargs...)
    grid = field_1.grid

    signed ? plmn = -1 : plmn = 1

    multiregion_field_1 = Reference(field_1.data.regional_objects)
    multiregion_field_2 = Reference(field_2.data.regional_objects)
    regions = Iterate(1:6)

    @apply_regionally fill_cubed_sphere_field_pairs_halo_event!(grid, field_1, multiregion_field_1, field_2,
        multiregion_field_2, regions, grid.connectivity.connections, plmn, WestAndEast(),
        _fill_cubed_sphere_face_center_center_face_field_pairs_east_west_halo_regions!)
    @apply_regionally fill_cubed_sphere_field_pairs_halo_event!(grid, field_1, multiregion_field_1, field_2,
        multiregion_field_2, regions, grid.connectivity.connections, plmn, SouthAndNorth(),
        _fill_cubed_sphere_face_center_center_face_field_pairs_north_south_halo_regions!)

    #=
    Add one valid field_1, field_2 value next to the corner, that allows us to compute vorticity on a wider stencil
    (e.g., ζ[0, 1, k] and ζ[1, 0, k]).
    =#

    if grid.Hx > 1
        @apply_regionally fill_cubed_sphere_field_pairs_corner_halo_event!(grid, field_1, field_2, plmn,
            _fill_cubed_sphere_face_center_field_corner_halo_regions!)
        @apply_regionally fill_cubed_sphere_field_pairs_corner_halo_event!(grid, field_1, field_2, plmn,
            _fill_cubed_sphere_center_face_field_corner_halo_regions!)
    end

    return nothing
end

@inline vertical_offset(::Tuple{<:Any, <:Any, <:Colon}) = 0
@inline vertical_offset(indices) = first(indices[3]) - 1

@inline function fill_cubed_sphere_field_pairs_corner_halo_event!(grid, field_1, field_2, plmn, _fill_halo_kernel!)
    Nz = size(field_1, 3)
    of = vertical_offset(field_1.indices)
    kernel_parameters = KernelParameters(Nz, of)
    reduced_dims = reduced_dimensions(field_1)
    
    return launch!(grid.architecture, grid, kernel_parameters, _fill_halo_kernel!, field_1, field_2, grid.Nx, grid.Hx,
                   plmn; reduced_dimensions = reduced_dims)
end

@kernel function _fill_cubed_sphere_face_center_center_face_field_pairs_east_west_halo_regions!(
field_1, multiregion_field_1, field_2, multiregion_field_2, region, connections, Nc, Hc, plmn)
    j, k = @index(Global, NTuple)

    region_E = connections.east.from_rank
    region_N = connections.north.from_rank
    region_W = connections.west.from_rank
    region_S = connections.south.from_rank

    # The commented blocks below show the equivalent non-GPU vectorized implementation, which can be useful for visually
    # verifying halo filling against schematics or physical cubed sphere models.

    #- E + W Halo for field:
    for i in 1:Hc
        if isodd(region)
            @inbounds begin
                #=
                #- E + W Halo for field_1:
                field_1[region][Nc+1:Nc+Hc, 1:Nc, k]   .=         field_1[region_E][1:Hc, 1:Nc, k]
                field_1[region][1-Hc:0, 1:Nc, k]       .= reverse(field_2[region_W][1:Nc, Nc+1-Hc:Nc, k], dims=1)'
                =#
                field_1[Nc+i, j, k] = multiregion_field_1[region_E][i, j, k]
                field_1[i-Hc, j, k] = multiregion_field_2[region_W][Nc+1-j, Nc+i-Hc, k]
                #=
                #- E + W Halo for field_2:
                field_2[region][Nc+1:Nc+Hc, 1:Nc, k]   .=         field_2[region_E][1:Hc, 1:Nc, k]
                field_2[region][Nc+1:Nc+Hc, Nc+1, k]   .=         field_2[region_N][1:Hc, 1, k]
                field_2[region][1-Hc:0, 2:Nc+1, k]     .= reverse(field_1[region_W][1:Nc, Nc+1-Hc:Nc, k], dims=1)' * plmn
                field_2[region][1-Hc:0, 1, k]          .=         field_1[region_S][1, Nc+1-Hc:Nc, k] * plmn
                =#
                field_2[Nc+i, j, k] = multiregion_field_2[region_E][i, j, k]
                field_2[Nc+i, Nc+1, k] = multiregion_field_2[region_N][i, 1, k]
                field_2[i-Hc, j+1, k] = multiregion_field_1[region_W][Nc+1-j, Nc+i-Hc, k] * plmn
                field_2[i-Hc, 1, k] = multiregion_field_1[region_S][1, Nc+i-Hc, k] * plmn
            end
        elseif iseven(region)
            @inbounds begin
                #=
                #- E + W Halo for field_1:
                field_1[region][Nc+1:Nc+Hc, 1:Nc, k]   .= reverse(field_2[region_E][1:Nc, 1:Hc, k], dims=1)'
                field_1[region][1-Hc:0, 1:Nc, k]       .=         field_1[region_W][Nc+1-Hc:Nc, 1:Nc, k]
                =#
                field_1[Nc+i, j, k] = multiregion_field_2[region_E][Nc+1-j, i, k]
                field_1[i-Hc, j, k] = multiregion_field_1[region_W][Nc+i-Hc, j, k]
                #=
                #- E + W Halo for field_2:
                field_2[region][Nc+1:Nc+Hc, 2:Nc+1, k] .= reverse(field_1[region_E][1:Nc, 1:Hc, k], dims=1)' * plmn
                field_2[region][Nc+1:Nc+Hc, 1, k]      .= reverse(field_2[region_S][Nc+1-Hc:Nc, 1, k]) * plmn
                field_2[region][1-Hc:0, 1:Nc, k]       .=         field_2[region_W][Nc+1-Hc:Nc, 1:Nc, k]
                field_2[region][1-Hc:0, Nc+1, k]       .= reverse(field_1[region_N][1, 1:Hc, k])
                =#
                field_2[Nc+i, j+1, k] = multiregion_field_1[region_E][Nc+1-j, i, k] * plmn
                field_2[Nc+i, 1, k] = multiregion_field_2[region_S][Nc+1-i, 1, k] * plmn
                field_2[i-Hc, j, k] = multiregion_field_2[region_W][Nc+i-Hc, j, k]
                field_2[i-Hc, Nc+1, k] = multiregion_field_1[region_N][1, Hc+1-i, k]
            end
        end
    end
end

@kernel function _fill_cubed_sphere_face_center_center_face_field_pairs_north_south_halo_regions!(
field_1, multiregion_field_1, field_2, multiregion_field_2, region, connections, Nc, Hc, plmn)
    i, k = @index(Global, NTuple)

    region_E = connections.east.from_rank
    region_N = connections.north.from_rank
    region_W = connections.west.from_rank
    region_S = connections.south.from_rank

    # The commented blocks below show the equivalent non-GPU vectorized implementation, which can be useful for visually
    # verifying halo filling against schematics or physical cubed sphere models.

    #- N + S Halo for field:
    for j in 1:Hc
        if isodd(region)
            @inbounds begin
                #=
                #- N + S Halo for field_1:
                field_1[region][2:Nc+1, Nc+1:Nc+Hc, k] .= reverse(field_2[region_N][1:Hc, 1:Nc, k], dims=2)' * plmn
                field_1[region][1, Nc+1:Nc+Hc, k]      .= reverse(field_1[region_W][1, Nc+1-Hc:Nc, k])' * plmn
                field_1[region][1:Nc, 1-Hc:0, k]       .=         field_1[region_S][1:Nc, Nc+1-Hc:Nc, k]
                field_1[region][Nc+1, 1-Hc:0, k]       .= reverse(field_2[region_E][1:Hc, 1, k])'
                =#
                field_1[i+1, Nc+j, k] = multiregion_field_2[region_N][j, Nc+1-i, k] * plmn
                field_1[1, Nc+j, k] = multiregion_field_1[region_W][1, Nc+1-j, k] * plmn
                field_1[i, j-Hc, k] = multiregion_field_1[region_S][i, Nc+j-Hc, k]
                field_1[Nc+1, j-Hc, k] = multiregion_field_2[region_E][Hc+1-j, 1, k]
                #=
                #- N + S Halo for field_2:
                field_2[region][1:Nc, Nc+1:Nc+Hc, k]   .= reverse(field_1[region_N][1:Hc, 1:Nc, k], dims=2)'
                field_2[region][1:Nc, 1-Hc:0, k]       .=         field_2[region_S][1:Nc, Nc+1-Hc:Nc, k]
                =#
                field_2[i, Nc+j, k] = multiregion_field_1[region_N][j, Nc+1-i, k]
                field_2[i, j-Hc, k] = multiregion_field_2[region_S][i, Nc+j-Hc, k]
            end
        elseif iseven(region)
            @inbounds begin
                #=
                #- N + S Halo for field_1:
                field_1[region][1:Nc, Nc+1:Nc+Hc, k]   .=         field_1[region_N][1:Nc, 1:Hc, k]
                field_1[region][Nc+1, Nc+1:Nc+Hc, k]   .=         field_1[region_E][1, 1:Hc, k]'
                field_1[region][2:Nc+1, 1-Hc:0, k]     .= reverse(field_2[region_S][Nc+1-Hc:Nc, 1:Nc, k], dims=2)' * plmn
                field_1[region][1, 1-Hc:0, k]          .=         field_2[region_W][Nc+1-Hc:Nc, 1, k]' * plmn
                =#
                field_1[i, Nc+j, k] = multiregion_field_1[region_N][i, j, k]
                field_1[Nc+1, Nc+j, k] = multiregion_field_1[region_E][1, j, k]
                field_1[i+1, j-Hc, k] = multiregion_field_2[region_S][Nc+j-Hc, Nc+1-i, k] * plmn
                field_1[1, j-Hc, k] = multiregion_field_2[region_W][Nc+j-Hc, 1, k] * plmn
                #=
                #- N + S Halo for field_2:
                field_2[region][1:Nc, Nc+1:Nc+Hc, k]   .=         field_2[region_N][1:Nc, 1:Hc, k]
                field_2[region][1:Nc, 1-Hc:0, k]       .= reverse(field_1[region_S][Nc+1-Hc:Nc, 1:Nc, k], dims=2)'
                =#
                field_2[i, Nc+j, k] = multiregion_field_2[region_N][i, j, k]
                field_2[i, j-Hc, k] = multiregion_field_1[region_S][Nc+j-Hc, Nc+1-i, k]
            end
        end
    end
end

@kernel function _fill_cubed_sphere_face_center_field_corner_halo_regions!(field_1, field_2, Nc, Hc, plmn)
    k = @index(Global, Linear)

    # The commented blocks below show the equivalent non-GPU vectorized implementation, which can be useful for visually
    # verifying halo filling against schematics or physical cubed sphere models.

    for i in 1:Hc
        @inbounds begin
            #=
            #- SW corner:
            field_1[region][1-Hc:0, 0, k] .= field_2[region][1, 1-Hc:0, k]
            =#
            field_1[i-Hc, 0, k] = field_2[1, i-Hc, k]
            #=
            #- NW corner:
            field_1[region][2-Hc:0, Nc+1, k] .= reverse(field_2[region][1, Nc+2:Nc+Hc, k]) * plmn
            =#
            (i > 1) && (field_1[i-Hc, Nc+1, k] = field_2[1, Nc+Hc+2-i, k] * plmn)
            #=
            #- SE corner:
            field_1[region][Nc+2:Nc+Hc, 0, k] .= reverse(field_2[region][Nc, 2-Hc:0, k]) * plmn
            =#
            (i > 1) && (field_1[Nc+i, 0, k] = field_2[Nc, 2-i, k] * plmn)
            #=
            #- NE corner:
            field_1[region][Nc+2:Nc+Hc, Nc+1, k] .= field_2[region][Nc, Nc+2:Nc+Hc, k]
            =#
            (i > 1) && (field_1[Nc+i, Nc+1, k] = field_2[Nc, Nc+i, k])
        end
    end
end

@kernel function _fill_cubed_sphere_center_face_field_corner_halo_regions!(field_1, field_2, Nc, Hc, plmn)
    k = @index(Global, Linear)

    # The commented blocks below show the equivalent non-GPU vectorized implementation, which can be useful for visually
    # verifying halo filling against schematics or physical cubed sphere models.

    for j in 1:Hc
        @inbounds begin
            #=
            #- SW corner:
            field_2[region][0, 1-Hc:0, k] .= field_1[region][1-Hc:0, 1, k]'
            =#
            field_2[0, j-Hc, k] = field_1[j-Hc, 1, k]
            #=
            #- NW corner:
            field_2[region][0, Nc+2:Nc+Hc, k] .= reverse(field_1[region][2-Hc:0, Nc, k])' * plmn
            =#
            (j > 1) && (field_2[0, Nc+j, k] = field_1[2-j, Nc, k] * plmn)
            #=
            #- SE corner:
            field_2[region][Nc+1, 2-Hc:0, k] .= reverse(field_1[region][Nc+2:Nc+Hc, 1, k])' * plmn
            =#
            (j > 1) && (field_2[Nc+1, j-Hc, k] = field_1[Nc+Hc+2-j, 1, k] * plmn)
            #=
            #- NE corner:
            field_2[region][Nc+1, Nc+2:Nc+Hc, k] .= field_1[region][Nc+2:Nc+Hc, Nc, k]'
            =#
            (j > 1) && (field_2[Nc+1, Nc+j, k] = field_1[Nc+j, Nc, k])
        end
    end
end

function fill_halo_regions!(field_1::CubedSphereField{<:Face, <:Face},
                            field_2::CubedSphereField{<:Face, <:Face}; signed = true, kwargs...)
    grid = field_1.grid

    signed ? plmn = -1 : plmn = 1

    multiregion_field_1 = Reference(field_1.data.regional_objects)
    multiregion_field_2 = Reference(field_2.data.regional_objects)
    regions = Iterate(1:6)

    @apply_regionally fill_cubed_sphere_field_pairs_halo_event!(grid, field_1, multiregion_field_1, field_2,
        multiregion_field_2, regions, grid.connectivity.connections, plmn, WestAndEast(),
        _fill_cubed_sphere_face_face_face_face_field_pairs_east_west_halo_regions!)
    @apply_regionally fill_cubed_sphere_field_pairs_halo_event!(grid, field_1, multiregion_field_1, field_2,
        multiregion_field_2, regions, grid.connectivity.connections, plmn, SouthAndNorth(),
        _fill_cubed_sphere_face_face_face_face_field_pairs_north_south_halo_regions!)

    return nothing
end

@kernel function _fill_cubed_sphere_face_face_face_face_field_pairs_east_west_halo_regions!(
field_1, multiregion_field_1, field_2, multiregion_field_2, region, connections, Nc, Hc, plmn)
    j, k = @index(Global, NTuple)

    region_E = connections.east.from_rank
    region_N = connections.north.from_rank
    region_W = connections.west.from_rank
    region_S = connections.south.from_rank

    # The commented blocks below show the equivalent non-GPU vectorized implementation, which can be useful for visually
    # verifying halo filling against schematics or physical cubed sphere models.

    #- E + W Halo for field:
    for i in 1:Hc
        if isodd(region)
            @inbounds begin
                #=
                #- E Halo:
                field_1[region][Nc+1:Nc+Hc, 1:Nc, k]   .=         field_1[region_E][1:Hc, 1:Nc, k]
                field_2[region][Nc+1:Nc+Hc, 1:Nc, k]   .=         field_2[region_E][1:Hc, 1:Nc, k]
                field_2[region][Nc+1:Nc+Hc, Nc+1, k]   .=         field_2[region_N][1:Hc, 1, k]
                =#
                field_1[Nc+i, j, k] = multiregion_field_1[region_E][i, j, k]
                field_2[Nc+i, j, k] = multiregion_field_2[region_E][i, j, k]
                field_2[Nc+i, Nc+1, k] = multiregion_field_2[region_N][i, 1, k]
                #=
                #- W Halo:
                field_1[region][1-Hc:0, 2:Nc+1, k]     .= reverse(field_2[region_W][1:Nc, Nc+1-Hc:Nc, k], dims=1)'
                field_2[region][1-Hc:0, 2:Nc+1, k]     .= reverse(field_1[region_W][1:Nc, Nc+1-Hc:Nc, k], dims=1)' * plmn
                field_1[region][1-Hc:0, 1, k]          .=         field_2[region_S][1, Nc+1-Hc:Nc, k]
                field_2[region][1-Hc:0, 1, k]          .=         field_1[region_S][1, Nc+1-Hc:Nc, k] * plmn
                =#
                field_1[i-Hc, j+1, k] = multiregion_field_2[region_W][Nc+1-j, Nc+i-Hc, k]
                field_2[i-Hc, j+1, k] = multiregion_field_1[region_W][Nc+1-j, Nc+i-Hc, k] * plmn
                field_1[i-Hc, 1, k] = multiregion_field_2[region_S][1, Nc+i-Hc, k]
                field_2[i-Hc, 1, k] = multiregion_field_1[region_S][1, Nc+i-Hc, k] * plmn
            end
        elseif iseven(region)
            @inbounds begin
                #=
                #- E Halo:
                field_1[region][Nc+1:Nc+Hc, 2:Nc, k]   .= reverse(field_2[region_E][2:Nc, 1:Hc, k], dims=1)'
                field_2[region][Nc+1:Nc+Hc, 2:Nc+1, k] .= reverse(field_1[region_E][1:Nc, 1:Hc, k], dims=1)' * plmn
                if Hc > 1
                    field_1[region][Nc+2:Nc+Hc, 1, k]  .= reverse(field_1[region_S][Nc+2-Hc:Nc, 1, k]) * plmn
                    field_2[region][Nc+2:Nc+Hc, 1, k]  .= reverse(field_2[region_S][Nc+2-Hc:Nc, 1, k]) * plmn
                end
                Note that the halos corresponding to the "missing" south-east corner of even panels, specifically
                field_1[region][Nc+1, 1, k] and field_2[region][Nc+1, 1, k], remain unfilled.
                =#
                j > 1 && (field_1[Nc+i, j, k] = multiregion_field_2[region_E][Nc+2-j, i, k])
                field_2[Nc+i, j+1, k] = multiregion_field_1[region_E][Nc+1-j, i, k] * plmn
                (Hc > 1 && i > 1) && (field_1[Nc+i, 1, k] = multiregion_field_1[region_S][Nc+2-i, 1, k]) * plmn
                (Hc > 1 && i > 1) && (field_2[Nc+i, 1, k] = multiregion_field_2[region_S][Nc+2-i, 1, k]) * plmn
                #=
                #- W Halo:
                field_1[region][1-Hc:0, 1:Nc, k]       .=         field_1[region_W][Nc+1-Hc:Nc, 1:Nc, k]
                field_2[region][1-Hc:0, 1:Nc, k]       .=         field_2[region_W][Nc+1-Hc:Nc, 1:Nc, k]
                field_2[region][1-Hc:0, Nc+1, k]       .= reverse(field_1[region_N][1, 2:Hc+1, k])
                =#
                field_1[i-Hc, j, k] = multiregion_field_1[region_W][Nc+i-Hc, j, k]
                field_2[i-Hc, j, k] = multiregion_field_2[region_W][Nc+i-Hc, j, k]
                field_2[i-Hc, Nc+1, k] = multiregion_field_1[region_N][1, Hc+2-i, k]
            end
        end
    end
end

@kernel function _fill_cubed_sphere_face_face_face_face_field_pairs_north_south_halo_regions!(
field_1, multiregion_field_1, field_2, multiregion_field_2, region, connections, Nc, Hc, plmn)
    i, k = @index(Global, NTuple)

    region_E = connections.east.from_rank
    region_N = connections.north.from_rank
    region_W = connections.west.from_rank
    region_S = connections.south.from_rank

    # The commented blocks below show the equivalent non-GPU vectorized implementation, which can be useful for visually
    # verifying halo filling against schematics or physical cubed sphere models.

    #- N + S Halo for field:
    for j in 1:Hc
        if isodd(region)
            @inbounds begin
                #=
                #- N Halo:
                field_1[region][2:Nc+1, Nc+1:Nc+Hc, k] .= reverse(field_2[region_N][1:Hc, 1:Nc, k], dims=2)' * plmn
                field_2[region][2:Nc, Nc+1:Nc+Hc, k]   .= reverse(field_1[region_N][1:Hc, 2:Nc, k], dims=2)'
                if Hc > 1
                    field_1[region][1, Nc+2:Nc+Hc, k]  .= reverse(field_1[region_W][1, Nc+2-Hc:Nc, k])' * plmn
                    field_2[region][1, Nc+2:Nc+Hc, k]  .= reverse(field_2[region_W][1, Nc+2-Hc:Nc, k])' * plmn
                end
                # Note that the halos corresponding to the "missing" north-west corner of odd panels, specifically
                # field_1[region][1, Nc+1, k] and field_2[region][1, Nc+1, k], remain unfilled.
                =#
                field_1[i+1, Nc+j, k] = multiregion_field_2[region_N][j, Nc+1-i, k] * plmn
                (i > 1) && (field_2[i, Nc+j, k] = multiregion_field_1[region_N][j, Nc+2-i, k])
                (Hc > 1 && j > 1) && (field_1[1, Nc+j, k] = multiregion_field_1[region_W][1, Nc+2-j, k]) * plmn
                (Hc > 1 && j > 1) && (field_2[1, Nc+j, k] = multiregion_field_2[region_W][1, Nc+2-j, k]) * plmn
                #=
                #- S Halo:
                field_1[region][1:Nc, 1-Hc:0, k]       .=         field_1[region_S][1:Nc, Nc+1-Hc:Nc, k]
                field_2[region][1:Nc, 1-Hc:0, k]       .=         field_2[region_S][1:Nc, Nc+1-Hc:Nc, k]
                field_1[region][Nc+1, 1-Hc:0, k]       .= reverse(field_2[region_E][2:Hc+1, 1, k])'
                =#
                field_1[i, j-Hc, k] = multiregion_field_1[region_S][i, Nc+j-Hc, k]
                field_2[i, j-Hc, k] = multiregion_field_2[region_S][i, Nc+j-Hc, k]
                field_1[Nc+1, j-Hc, k] = multiregion_field_2[region_E][Hc+2-j, 1, k]
            end
        elseif iseven(region)
            @inbounds begin
                #=
                #- N Halo:
                field_1[region][1:Nc, Nc+1:Nc+Hc, k]   .=         field_1[region_N][1:Nc, 1:Hc, k]
                field_2[region][1:Nc, Nc+1:Nc+Hc, k]   .=         field_2[region_N][1:Nc, 1:Hc, k]
                field_1[region][Nc+1, Nc+1:Nc+Hc, k]   .=         field_1[region_E][1, 1:Hc, k]'
                =#
                field_1[i, Nc+j, k] = multiregion_field_1[region_N][i, j, k]
                field_2[i, Nc+j, k] = multiregion_field_2[region_N][i, j, k]
                field_1[Nc+1, Nc+j, k] = multiregion_field_1[region_E][1, j, k]
                #=
                #- S Halo:
                field_1[region][2:Nc+1, 1-Hc:0, k]     .= reverse(field_2[region_S][Nc+1-Hc:Nc, 1:Nc, k], dims=2)' * plmn
                field_2[region][2:Nc, 1-Hc:0, k]       .= reverse(field_1[region_S][Nc+1-Hc:Nc, 2:Nc, k], dims=2)'
                field_1[region][1, 1-Hc:0, k]          .=         field_2[region_W][Nc+1-Hc:Nc, 1, k]' * plmn
                field_2[region][1, 1-Hc:0, k]          .=         field_1[region_W][Nc+1-Hc:Nc, 1, k]'
                =#
                field_1[i+1, j-Hc, k] = multiregion_field_2[region_S][Nc+j-Hc, Nc+1-i, k] * plmn
                (i > 1) && (field_2[i, j-Hc, k] = multiregion_field_1[region_S][Nc+j-Hc, Nc+2-i, k])
                field_1[1, j-Hc, k] = multiregion_field_2[region_W][Nc+j-Hc, 1, k] * plmn
                field_2[1, j-Hc, k] = multiregion_field_1[region_W][Nc+j-Hc, 1, k]
            end
        end
    end
end
