using Oceananigans.MultiRegion: number_of_regions

import Oceananigans.BoundaryConditions: fill_halo_regions!

function find_neighboring_panels(n_regions, region)
    n_regions !== 6 && error("requires cubed sphere grids with 1 region per panel")

    if isodd(region)
        region_E = mod(region + 0, 6) + 1
        region_N = mod(region + 1, 6) + 1
        region_W = mod(region + 3, 6) + 1
        region_S = mod(region + 4, 6) + 1
    elseif iseven(region)
        region_E = mod(region + 1, 6) + 1
        region_N = mod(region + 0, 6) + 1
        region_W = mod(region + 4, 6) + 1
        region_S = mod(region + 3, 6) + 1
    end

    return (; region_E, region_N, region_W, region_S)
end

function fill_halo_regions!(field::CubedSphereField{<:Center, <:Center})
    grid = field.grid
    n_regions = number_of_regions(grid)

    Nx, Ny, Nz = size(grid)
    Hx, Hy, Hz = halo_size(grid)

    Nx == Ny || error("horizontal grid size Nx and Ny must be the same")
    Nc = Nx

    Hx == Hy || error("horizontal halo size Hx and Hy must be the same")
    Hc = Hx

    multiregion_field = Reference(field.data.regional_objects)
    region = Iterate(1:6)

    kernel_parameters = KernelParameters((Hc, Nc, Nz), (0, 0, 0))
    @apply_regionally begin
        launch!(grid.architecture, grid, kernel_parameters, _fill_cubed_sphere_center_center_field_east_west_halo_regions!,
                field, multiregion_field, n_regions, region, Hc, Nc)
    end

    kernel_parameters = KernelParameters((Nc, Hc, Nz), (0, 0, 0))
    @apply_regionally begin
        launch!(grid.architecture, grid, kernel_parameters, _fill_cubed_sphere_center_center_field_north_south_halo_regions!,
                field, multiregion_field, n_regions, region, Nc, Hc)
    end

    return nothing
end

@kernel function _fill_cubed_sphere_center_center_field_east_west_halo_regions!(field, multiregion_field, n_regions,
                                                                                region, Hc, Nc)
    i, j, k = @index(Global, NTuple)
    region_E, region_N, region_W, region_S = find_neighboring_panels(n_regions, region)

    #- E + W Halo for field:
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

@kernel function _fill_cubed_sphere_center_center_field_north_south_halo_regions!(field, multiregion_field, n_regions,
                                                                                  region, Nc, Hc)
    i, j, k = @index(Global, NTuple)
    region_E, region_N, region_W, region_S = find_neighboring_panels(n_regions, region)

    #- N + S Halo for field:
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

function fill_halo_regions!(field::CubedSphereField{<:Face, <:Face})
    grid = field.grid
    n_regions = number_of_regions(grid)

    Nx, Ny, Nz = size(grid)
    Hx, Hy, Hz = halo_size(grid)

    Nx == Ny || error("horizontal grid size Nx and Ny must be the same")
    Nc = Nx

    Hx == Hy || error("horizontal halo size Hx and Hy must be the same")
    Hc = Hx

    multiregion_field = Reference(field.data.regional_objects)
    region = Iterate(1:6)

    kernel_parameters = KernelParameters((Hc, Nc, Nz), (0, 0, 0))
    @apply_regionally begin
        launch!(grid.architecture, grid, kernel_parameters, _fill_cubed_sphere_face_face_field_east_west_halo_regions!,
                field, multiregion_field, n_regions, region, Hc, Nc)
    end

    kernel_parameters = KernelParameters((Nc, Hc, Nz), (0, 0, 0))
    @apply_regionally begin
        launch!(grid.architecture, grid, kernel_parameters, _fill_cubed_sphere_face_face_field_north_south_halo_regions!,
                field, multiregion_field, n_regions, region, Nc, Hc)
    end

    return nothing
end

@kernel function _fill_cubed_sphere_face_face_field_east_west_halo_regions!(field, multiregion_field, n_regions, region,
                                                                            Hc, Nc)
    i, j, k = @index(Global, NTuple)
    region_E, region_N, region_W, region_S = find_neighboring_panels(n_regions, region)

    #- E + W Halo for field:
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
            =#
            j > 1 && (field[Nc+i, j, k] = multiregion_field[region_E][Nc+2-j, i, k])
            (Hc > 1 && i > 1) && (field[Nc+i, 1, k] = multiregion_field[region_S][Nc+2-i, 1, k])
            #=
            Note that the halo corresponding to the "missing" south-east corner of even panels, specifically
            field[region][Nc+1, 1, k], remains unfilled.
            field[region][1-Hc:0, 1:Nc, k]       .=         field[region_W][Nc+1-Hc:Nc, 1:Nc, k]
            =#
            field[i-Hc, j, k] = multiregion_field[region_W][Nc+i-Hc, j, k]
        end
    end
end

@kernel function _fill_cubed_sphere_face_face_field_north_south_halo_regions!(field, multiregion_field, n_regions,
                                                                              region, Nc, Hc)
    i, j, k = @index(Global, NTuple)
    region_E, region_N, region_W, region_S = find_neighboring_panels(n_regions, region)

    #- N + S Halo for field:
    if isodd(region)
        @inbounds begin
            #=
            field[region][2:Nc+1, Nc+1:Nc+Hc, k] .= reverse(field[region_N][1:Hc, 1:Nc, k], dims=2)'
            if Hc > 1
                field[region][1, Nc+2:Nc+Hc, k]  .= reverse(field[region_W][1, Nc+2-Hc:Nc, k])'
            end
            =#
            field[i+1, Nc+j, k] = multiregion_field[region_N][j, Nc+1-i, k]
            (Hc > 1 && j > 1) && (field[1, Nc+j, k] = multiregion_field[region_W][1, Nc+2-j, k])
            #=
            Note that the halo corresponding to the "missing" north-west corner of odd panels, specifically
            field[region][1, Nc+1, k], remains unfilled.
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

fill_halo_regions!(fields::Tuple{CubedSphereField, CubedSphereField}; signed = true) = fill_halo_regions!(fields...; signed)

function fill_halo_regions!(field_1::CubedSphereField{<:Center, <:Center},
                            field_2::CubedSphereField{<:Center, <:Center}; signed = true)

    field_1.grid == field_2.grid || error("fields must be on the same grid")
    grid = field_1.grid
    n_regions = number_of_regions(grid)

    Nx, Ny, Nz = size(grid)
    Hx, Hy, Hz = halo_size(grid)
    signed ? plmn = -1 : plmn = 1

    Nx == Ny || error("horizontal grid size Nx and Ny must be the same")
    Nc = Nx

    Hx == Hy || error("horizontal halo size Hx and Hy must be the same")
    Hc = Hx

    multiregion_field_1 = Reference(field_1.data.regional_objects)
    multiregion_field_2 = Reference(field_2.data.regional_objects)
    region = Iterate(1:6)

    kernel_parameters = KernelParameters((Hc, Nc, Nz), (0, 0, 0))
    @apply_regionally begin
        launch!(grid.architecture, grid, kernel_parameters,
                _fill_cubed_sphere_center_center_center_center_fields_east_west_halo_regions!,
                field_1, multiregion_field_1, field_2, multiregion_field_2, n_regions, region, Hc, Nc, plmn)
    end

    kernel_parameters = KernelParameters((Nc, Hc, Nz), (0, 0, 0))
    @apply_regionally begin
        launch!(grid.architecture, grid, kernel_parameters,
                _fill_cubed_sphere_center_center_center_center_fields_north_south_halo_regions!,
                field_1, multiregion_field_1, field_2, multiregion_field_2, n_regions, region, Nc, Hc, plmn)
    end

    return nothing
end

@kernel function _fill_cubed_sphere_center_center_center_center_fields_east_west_halo_regions!(
field_1, multiregion_field_1, field_2, multiregion_field_2, n_regions, region, Hc, Nc, plmn)
    i, j, k = @index(Global, NTuple)
    region_E, region_N, region_W, region_S = find_neighboring_panels(n_regions, region)

    #- E + W Halo for field:
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

@kernel function _fill_cubed_sphere_center_center_center_center_fields_north_south_halo_regions!(
field_1, multiregion_field_1, field_2, multiregion_field_2, n_regions, region, Nc, Hc, plmn)
    i, j, k = @index(Global, NTuple)
    region_E, region_N, region_W, region_S = find_neighboring_panels(n_regions, region)

    #- N + S Halo for field:
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

function fill_halo_regions!(field_1::CubedSphereField{<:Face, <:Center},
                            field_2::CubedSphereField{<:Center, <:Face}; signed = true)

    field_1.grid == field_2.grid || error("fields must be on the same grid")
    grid = field_1.grid
    n_regions = number_of_regions(grid)

    Nx, Ny, Nz = size(grid)
    Hx, Hy, Hz = halo_size(grid)
    signed ? plmn = -1 : plmn = 1

    Nx == Ny || error("horizontal grid size Nx and Ny must be the same")
    Nc = Nx

    Hx == Hy || error("horizontal halo size Hx and Hy must be the same")
    Hc = Hx

    #-- one pass: only use interior-point values:
    for region in 1:6

        region_E, region_N, region_W, region_S = find_neighboring_panels(n_regions, region)

        if isodd(region)
            #- odd face number (1, 3, 5):
            for k in -Hz+1:Nz+Hz
                #- E + W Halo for field_1:
                field_1[region][Nc+1:Nc+Hc, 1:Nc, k]   .=         field_1[region_E][1:Hc, 1:Nc, k]
                field_1[region][1-Hc:0, 1:Nc, k]       .= reverse(field_2[region_W][1:Nc, Nc+1-Hc:Nc, k], dims=1)'
                #- N + S Halo for field_1:
                field_1[region][2:Nc+1, Nc+1:Nc+Hc, k] .= reverse(field_2[region_N][1:Hc, 1:Nc, k], dims=2)' * plmn
                field_1[region][1, Nc+1:Nc+Hc, k]      .= reverse(field_1[region_W][1, Nc+1-Hc:Nc, k])' * plmn
                field_1[region][1:Nc, 1-Hc:0, k]       .=         field_1[region_S][1:Nc, Nc+1-Hc:Nc, k]
                field_1[region][Nc+1, 1-Hc:0, k]       .= reverse(field_2[region_E][1:Hc, 1, k])'
                #- E + W Halo for field_2:
                field_2[region][Nc+1:Nc+Hc, 1:Nc, k]   .=         field_2[region_E][1:Hc, 1:Nc, k]
                field_2[region][Nc+1:Nc+Hc, Nc+1, k]   .=         field_2[region_N][1:Hc, 1, k]
                field_2[region][1-Hc:0, 2:Nc+1, k]     .= reverse(field_1[region_W][1:Nc, Nc+1-Hc:Nc, k], dims=1)' * plmn
                field_2[region][1-Hc:0, 1, k]          .=         field_1[region_S][1, Nc+1-Hc:Nc, k] * plmn
                #- N + S Halo for field_2:
                field_2[region][1:Nc, Nc+1:Nc+Hc, k]   .= reverse(field_1[region_N][1:Hc, 1:Nc, k], dims=2)'
                field_2[region][1:Nc, 1-Hc:0, k]       .=         field_2[region_S][1:Nc, Nc+1-Hc:Nc, k]
            end
        elseif iseven(region)
            #- even face number (2, 4, 6):
            for k in -Hz+1:Nz+Hz
                #- E + W Halo for field_1:
                field_1[region][Nc+1:Nc+Hc, 1:Nc, k]   .= reverse(field_2[region_E][1:Nc, 1:Hc, k], dims=1)'
                field_1[region][1-Hc:0, 1:Nc, k]       .=         field_1[region_W][Nc+1-Hc:Nc, 1:Nc, k]
                #- N + S Halo for field_1:
                field_1[region][1:Nc, Nc+1:Nc+Hc, k]   .=         field_1[region_N][1:Nc, 1:Hc, k]
                field_1[region][Nc+1, Nc+1:Nc+Hc, k]   .=         field_1[region_E][1, 1:Hc, k]'
                field_1[region][2:Nc+1, 1-Hc:0, k]     .= reverse(field_2[region_S][Nc+1-Hc:Nc, 1:Nc, k], dims=2)' * plmn
                field_1[region][1, 1-Hc:0, k]          .=         field_2[region_W][Nc+1-Hc:Nc, 1, k]' * plmn
                #- E + W Halo for field_2:
                field_2[region][Nc+1:Nc+Hc, 2:Nc+1, k] .= reverse(field_1[region_E][1:Nc, 1:Hc, k], dims=1)' * plmn
                field_2[region][Nc+1:Nc+Hc, 1, k]      .= reverse(field_2[region_S][Nc+1-Hc:Nc, 1, k]) * plmn
                field_2[region][1-Hc:0, 1:Nc, k]       .=         field_2[region_W][Nc+1-Hc:Nc, 1:Nc, k]
                field_2[region][1-Hc:0, Nc+1, k]       .= reverse(field_1[region_N][1, 1:Hc, k])
                #- N + S Halo for field_2:
                field_2[region][1:Nc, Nc+1:Nc+Hc, k]   .=         field_2[region_N][1:Nc, 1:Hc, k]
                field_2[region][1:Nc, 1-Hc:0, k]       .= reverse(field_1[region_S][Nc+1-Hc:Nc, 1:Nc, k], dims=2)'
            end
        end
    end

    #-- Add one valid field_1, field_2 value next to the corner, that allows to compute vorticity on a wider stencil
    # (e.g., vort3(0,1) & (1,0)).
    if Hc > 1
        for region in 1:6
            for k in -Hz+1:Nz+Hz
                #- SW corner:
                field_1[region][1-Hc:0, 0, k] .= field_2[region][1, 1-Hc:0, k]
                field_2[region][0, 1-Hc:0, k] .= field_1[region][1-Hc:0, 1, k]'
                #- NW corner:
                field_1[region][2-Hc:0, Nc+1,  k]    .= reverse(field_2[region][1, Nc+2:Nc+Hc, k]) * plmn
                field_2[region][0, Nc+2:Nc+Hc, k]    .= reverse(field_1[region][2-Hc:0, Nc, k])' * plmn
                #- SE corner:
                field_1[region][Nc+2:Nc+Hc, 0, k]    .= reverse(field_2[region][Nc, 2-Hc:0, k]) * plmn
                field_2[region][Nc+1, 2-Hc:0,  k]    .= reverse(field_1[region][Nc+2:Nc+Hc, 1, k])' * plmn
                #- NE corner:
                field_1[region][Nc+2:Nc+Hc, Nc+1, k] .= field_2[region][Nc, Nc+2:Nc+Hc, k]
                field_2[region][Nc+1, Nc+2:Nc+Hc, k] .= field_1[region][Nc+2:Nc+Hc, Nc, k]'
            end
        end
    end

    return nothing
end

function fill_halo_regions!(field_1::CubedSphereField{<:Face, <:Face},
                            field_2::CubedSphereField{<:Face, <:Face}; signed = true)

    field_1.grid == field_2.grid || error("fields must be on the same grid")
    grid = field_1.grid
    n_regions = number_of_regions(grid)

    Nx, Ny, Nz = size(grid)
    Hx, Hy, Hz = halo_size(grid)
    signed ? plmn = -1 : plmn = 1

    Nx == Ny || error("horizontal grid size Nx and Ny must be the same")
    Nc = Nx

    Hx == Hy || error("horizontal halo size Hx and Hy must be the same")
    Hc = Hx

    #-- one pass: only use interior-point values:
    for region in 1:6

        region_E, region_N, region_W, region_S = find_neighboring_panels(n_regions, region)

        if isodd(region)
            #- odd face number (1, 3, 5):
            for k in -Hz+1:Nz+Hz
                #- E Halo:
                field_1[region][Nc+1:Nc+Hc, 1:Nc, k]   .=         field_1[region_E][1:Hc, 1:Nc, k]
                field_2[region][Nc+1:Nc+Hc, 1:Nc, k]   .=         field_2[region_E][1:Hc, 1:Nc, k]
                field_2[region][Nc+1:Nc+Hc, Nc+1, k]   .=         field_2[region_N][1:Hc, 1, k]
                #- W Halo:
                field_1[region][1-Hc:0, 2:Nc+1, k]     .= reverse(field_2[region_W][1:Nc, Nc+1-Hc:Nc, k], dims=1)'
                field_2[region][1-Hc:0, 2:Nc+1, k]     .= reverse(field_1[region_W][1:Nc, Nc+1-Hc:Nc, k], dims=1)' * plmn
                field_1[region][1-Hc:0, 1, k]          .=         field_2[region_S][1, Nc+1-Hc:Nc, k]
                field_2[region][1-Hc:0, 1, k]          .=         field_1[region_S][1, Nc+1-Hc:Nc, k] * plmn
                #- N Halo:
                field_1[region][2:Nc+1, Nc+1:Nc+Hc, k] .= reverse(field_2[region_N][1:Hc, 1:Nc, k], dims=2)' * plmn
                field_2[region][2:Nc, Nc+1:Nc+Hc, k]   .= reverse(field_1[region_N][1:Hc, 2:Nc, k], dims=2)'
                if Hc > 1
                    field_1[region][1, Nc+2:Nc+Hc, k]  .= reverse(field_1[region_W][1, Nc+2-Hc:Nc, k])' * plmn
                    field_2[region][1, Nc+2:Nc+Hc, k]  .= reverse(field_2[region_W][1, Nc+2-Hc:Nc, k])' * plmn
                end
                # Note that the halos corresponding to the "missing" north-west corner of odd panels, specifically
                # field_1[region][1, Nc+1, k] and field_2[region][1, Nc+1, k], remain unfilled.
                #- S Halo:
                field_1[region][1:Nc, 1-Hc:0, k]       .=         field_1[region_S][1:Nc, Nc+1-Hc:Nc, k]
                field_2[region][1:Nc, 1-Hc:0, k]       .=         field_2[region_S][1:Nc, Nc+1-Hc:Nc, k]
                field_1[region][Nc+1, 1-Hc:0, k]       .= reverse(field_2[region_E][2:Hc+1, 1, k])'
            end
        else
            #- even face number (2, 4, 6):
            for k in -Hz+1:Nz+Hz
                #- E Halo:
                field_1[region][Nc+1:Nc+Hc, 2:Nc, k]   .= reverse(field_2[region_E][2:Nc, 1:Hc, k], dims=1)'
                field_2[region][Nc+1:Nc+Hc, 2:Nc+1, k] .= reverse(field_1[region_E][1:Nc, 1:Hc, k], dims=1)' * plmn
                if Hc > 1
                    field_1[region][Nc+2:Nc+Hc, 1, k]  .= reverse(field_1[region_S][Nc+2-Hc:Nc, 1, k]) * plmn
                    field_2[region][Nc+2:Nc+Hc, 1, k]  .= reverse(field_2[region_S][Nc+2-Hc:Nc, 1, k]) * plmn
                end
                # Note that the halos corresponding to the "missing" south-east corner of even panels, specifically
                # field_1[region][Nc+1, 1, k] and field_2[region][Nc+1, 1, k], remain unfilled.
                #- W Halo:
                field_1[region][1-Hc:0, 1:Nc, k]       .=         field_1[region_W][Nc+1-Hc:Nc, 1:Nc, k]
                field_2[region][1-Hc:0, 1:Nc, k]       .=         field_2[region_W][Nc+1-Hc:Nc, 1:Nc, k]
                field_2[region][1-Hc:0, Nc+1, k]       .= reverse(field_1[region_N][1, 2:Hc+1, k])
                #- N Halo:
                field_1[region][1:Nc, Nc+1:Nc+Hc, k]   .=         field_1[region_N][1:Nc, 1:Hc, k]
                field_2[region][1:Nc, Nc+1:Nc+Hc, k]   .=         field_2[region_N][1:Nc, 1:Hc, k]
                field_1[region][Nc+1, Nc+1:Nc+Hc, k]   .=         field_1[region_E][1, 1:Hc, k]'
                #- S Halo:
                field_1[region][2:Nc+1, 1-Hc:0, k]     .= reverse(field_2[region_S][Nc+1-Hc:Nc, 1:Nc, k], dims=2)' * plmn
                field_2[region][2:Nc, 1-Hc:0, k]       .= reverse(field_1[region_S][Nc+1-Hc:Nc, 2:Nc, k], dims=2)'
                field_1[region][1, 1-Hc:0, k]          .=         field_2[region_W][Nc+1-Hc:Nc, 1, k]' * plmn
                field_2[region][1, 1-Hc:0, k]          .=         field_1[region_W][Nc+1-Hc:Nc, 1, k]'
            end
        end
    end

    return nothing
end
