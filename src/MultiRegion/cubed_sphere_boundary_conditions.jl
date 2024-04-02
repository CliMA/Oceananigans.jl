function find_neighboring_panels(region)
    if mod(region, 2) == 1
        region_E = mod(region + 0, 6) + 1
        region_N = mod(region + 1, 6) + 1
        region_W = mod(region + 3, 6) + 1
        region_S = mod(region + 4, 6) + 1
    elseif mod(region, 2) == 0
        region_E = mod(region + 1, 6) + 1
        region_N = mod(region + 0, 6) + 1
        region_W = mod(region + 4, 6) + 1
        region_S = mod(region + 3, 6) + 1
    end

    return (; region_E, region_N, region_W, region_S)
end

function fill_cubed_sphere_halo_regions!(field::CubedSphereField{<:Center, <:Center})
    grid = field.grid

    Nx, Ny, Nz = size(grid)
    Hx, Hy, Hz = halo_size(grid)

    Nx == Ny || error("horizontal grid size Nx and Ny must be the same")
    Nc = Nx

    Hx == Hy || error("horizontal halo size Hx and Hy must be the same")
    Hc = Hx

    #-- one pass: only use interior-point values:
    for region in 1:6

        region_E, region_N, region_W, region_S = find_neighboring_panels(region)

        if mod(region, 2) == 1
            #- odd face number (1, 3, 5):
            for k in -Hz+1:Nz+Hz
                #- E + W Halo for field:
                field[region][Nc+1:Nc+Hc, 1:Nc, k] .=         field[region_E][1:Hc, 1:Nc, k]
                field[region][1-Hc:0, 1:Nc, k]     .= reverse(field[region_W][1:Nc, Nc+1-Hc:Nc, k], dims=1)'
                #- N + S Halo for field:
                field[region][1:Nc, Nc+1:Nc+Hc, k] .= reverse(field[region_N][1:Hc, 1:Nc, k], dims=2)'
                field[region][1:Nc, 1-Hc:0, k]     .=         field[region_S][1:Nc, Nc+1-Hc:Nc, k]
            end
        else
            #- even face number (2, 4, 6):
            for k in -Hz+1:Nz+Hz
                #- E + W Halo for field:
                field[region][Nc+1:Nc+Hc, 1:Nc, k] .= reverse(field[region_E][1:Nc, 1:Hc, k], dims=1)'
                field[region][1-Hc:0, 1:Nc, k]     .=         field[region_W][Nc+1-Hc:Nc, 1:Nc, k]
                #- N + S Halo for field:
                field[region][1:Nc, Nc+1:Nc+Hc, k] .=         field[region_N][1:Nc, 1:Hc, k]
                field[region][1:Nc, 1-Hc:0, k]     .= reverse(field[region_S][Nc+1-Hc:Nc, 1:Nc, k], dims=2)'
            end
        end
    end

    return nothing
end

function fill_cubed_sphere_halo_regions!(field::CubedSphereField{<:Face, <:Face})
    grid = field.grid

    Nx, Ny, Nz = size(grid)
    Hx, Hy, Hz = halo_size(grid)

    Nx == Ny || error("horizontal grid size Nx and Ny must be the same")
    Nc = Nx

    Hx == Hy || error("horizontal halo size Hx and Hy must be the same")
    Hc = Hx

    #-- one pass: only use interior-point values:
    for region in 1:6

        region_E, region_N, region_W, region_S = find_neighboring_panels(region)

        if mod(region, 2) == 1
            #- odd face number (1, 3, 5):
            for k in -Hz+1:Nz+Hz
                #- E + W Halo for field:
                field[region][Nc+1:Nc+Hc, 1:Nc, k]   .=         field[region_E][1:Hc, 1:Nc, k]
                field[region][1-Hc:0, 2:Nc+1, k]     .= reverse(field[region_W][1:Nc, Nc+1-Hc:Nc, k], dims=1)'
                field[region][1-Hc:0, 1, k]          .=         field[region_S][1, Nc+1-Hc:Nc, k]
                #- N + S Halo for field:
                field[region][2:Nc+1, Nc+1:Nc+Hc, k] .= reverse(field[region_N][1:Hc, 1:Nc, k], dims=2)'
                if Hc > 1
                    field[region][1, Nc+2:Nc+Hc, k]   = reverse(field[region_W][1, Nc+2-Hc:Nc, k])
                end
                # Note that the halo corresponding to the "missing" north-west corner of odd panels, specifically
                # field[region][1, Nc+1, k], remains unfilled.
                field[region][1:Nc, 1-Hc:0, k]       .=         field[region_S][1:Nc, Nc+1-Hc:Nc, k]
                field[region][Nc+1, 1-Hc:0, k]        = reverse(field[region_E][2:Hc+1, 1, k])
            end
        else
            #- even face number (2, 4, 6):
            for k in -Hz+1:Nz+Hz
                #- E + W Halo for field:
                field[region][Nc+1:Nc+Hc, 2:Nc, k]   .= reverse(field[region_E][2:Nc, 1:Hc, k], dims=1)'
                if Hc > 1
                    field[region][Nc+2:Nc+Hc, 1, k]  .= reverse(field[region_S][Nc+2-Hc:Nc, 1, k])
                end
                # Note that the halo corresponding to the "missing" south-east corner of even panels, specifically
                # field[region][Nc+1, 1, k], remains unfilled.
                field[region][1-Hc:0, 1:Nc, k]       .=         field[region_W][Nc+1-Hc:Nc, 1:Nc, k]
                #- N + S Halo for field:
                field[region][1:Nc, Nc+1:Nc+Hc, k]   .=         field[region_N][1:Nc, 1:Hc, k]
                field[region][Nc+1, Nc+1:Nc+Hc, k]    =         field[region_E][1, 1:Hc, k]
                field[region][2:Nc+1, 1-Hc:0, k]     .= reverse(field[region_S][Nc+1-Hc:Nc, 1:Nc, k], dims=2)'
                field[region][1, 1-Hc:0, k]           =         field[region_W][Nc+1-Hc:Nc, 1, k]
            end
        end
    end

    return nothing
end

fill_cubed_sphere_halo_regions!(fields::Tuple{CubedSphereField, CubedSphereField};
                                signed = true) = fill_cubed_sphere_halo_regions!(fields...; signed)

function fill_cubed_sphere_halo_regions!(field_1::CubedSphereField{<:Center, <:Center},
                                         field_2::CubedSphereField{<:Center, <:Center};
                                         signed = true)

    field_1.grid == field_2.grid || error("fields must be on the same grid")
    grid = field_1.grid

    Nx, Ny, Nz = size(grid)
    Hx, Hy, Hz = halo_size(grid)
    signed ? plmn = -1 : plmn = 1

    Nx == Ny || error("horizontal grid size Nx and Ny must be the same")
    Nc = Nx

    Hx == Hy || error("horizontal halo size Hx and Hy must be the same")
    Hc = Hx

    #-- one pass: only use interior-point values:
    for region in 1:6

        region_E, region_N, region_W, region_S = find_neighboring_panels(region)

        if mod(region, 2) == 1
            #- odd face number (1, 3, 5):
            for k in -Hz+1:Nz+Hz
                #- E Halo:
                field_1[region][Nc+1:Nc+Hc, 1:Nc, k] .=         field_1[region_E][1:Hc, 1:Nc, k]
                field_2[region][Nc+1:Nc+Hc, 1:Nc, k] .=         field_2[region_E][1:Hc, 1:Nc, k]
                #- W Halo:
                field_1[region][1-Hc:0, 1:Nc, k]     .= reverse(field_2[region_W][1:Nc, Nc+1-Hc:Nc, k], dims=1)'
                field_2[region][1-Hc:0, 1:Nc, k]     .= reverse(field_1[region_W][1:Nc, Nc+1-Hc:Nc, k], dims=1)' * plmn
                #- N Halo:
                field_1[region][1:Nc, Nc+1:Nc+Hc, k] .= reverse(field_2[region_N][1:Hc, 1:Nc, k], dims=2)' * plmn
                field_2[region][1:Nc, Nc+1:Nc+Hc, k] .= reverse(field_1[region_N][1:Hc, 1:Nc, k], dims=2)'
                #- S Halo:
                field_1[region][1:Nc, 1-Hc:0, k]     .=         field_1[region_S][1:Nc, Nc+1-Hc:Nc, k]
                field_2[region][1:Nc, 1-Hc:0, k]     .=         field_2[region_S][1:Nc, Nc+1-Hc:Nc, k]
            end
        else
            #- even face number (2, 4, 6):
            for k in -Hz+1:Nz+Hz
                #- E Halo:
                field_1[region][Nc+1:Nc+Hc, 1:Nc, k] .= reverse(field_2[region_E][1:Nc, 1:Hc, k], dims=1)'
                field_2[region][Nc+1:Nc+Hc, 1:Nc, k] .= reverse(field_1[region_E][1:Nc, 1:Hc, k], dims=1)' * plmn
                #- W Halo:
                field_1[region][1-Hc:0, 1:Nc, k]     .=         field_1[region_W][Nc+1-Hc:Nc, 1:Nc, k]
                field_2[region][1-Hc:0, 1:Nc, k]     .=         field_2[region_W][Nc+1-Hc:Nc, 1:Nc, k]
                #- N Halo:
                field_1[region][1:Nc, Nc+1:Nc+Hc, k] .=         field_1[region_N][1:Nc, 1:Hc, k]
                field_2[region][1:Nc, Nc+1:Nc+Hc, k] .=         field_2[region_N][1:Nc, 1:Hc, k]
                #- S Halo:
                field_1[region][1:Nc, 1-Hc:0, k]     .= reverse(field_2[region_S][Nc+1-Hc:Nc, 1:Nc, k], dims=2)' * plmn
                field_2[region][1:Nc, 1-Hc:0, k]     .= reverse(field_1[region_S][Nc+1-Hc:Nc, 1:Nc, k], dims=2)'
            end
        end
    end

    return nothing
end

function fill_cubed_sphere_halo_regions!(field_1::CubedSphereField{<:Face, <:Center},
                                         field_2::CubedSphereField{<:Center, <:Face};
                                         signed = true)

    field_1.grid == field_2.grid || error("fields must be on the same grid")
    grid = field_1.grid

    Nx, Ny, Nz = size(grid)
    Hx, Hy, Hz = halo_size(grid)
    signed ? plmn = -1 : plmn = 1

    Nx == Ny || error("horizontal grid size Nx and Ny must be the same")
    Nc = Nx

    Hx == Hy || error("horizontal halo size Hx and Hy must be the same")
    Hc = Hx

    #-- one pass: only use interior-point values:
    for region in 1:6

        region_E, region_N, region_W, region_S = find_neighboring_panels(region)

        if mod(region, 2) == 1
            #- odd face number (1, 3, 5):
            for k in -Hz+1:Nz+Hz
                #- E + W Halo for field_1:
                field_1[region][Nc+1:Nc+Hc, 1:Nc, k]   .=         field_1[region_E][1:Hc, 1:Nc, k]
                field_1[region][1-Hc:0, 1:Nc, k]       .= reverse(field_2[region_W][1:Nc, Nc+1-Hc:Nc, k], dims=1)'
                #- N + S Halo for field_1:
                field_1[region][2:Nc+1, Nc+1:Nc+Hc, k] .= reverse(field_2[region_N][1:Hc, 1:Nc, k], dims=2)' * plmn
                field_1[region][1, Nc+1:Nc+Hc, k]       = reverse(field_1[region_W][1, Nc+1-Hc:Nc, k]) * plmn
                field_1[region][1:Nc, 1-Hc:0, k]       .=         field_1[region_S][1:Nc, Nc+1-Hc:Nc, k]
                field_1[region][Nc+1, 1-Hc:0, k]        = reverse(field_2[region_E][1:Hc, 1, k])
                #- E + W Halo for field_2:
                field_2[region][Nc+1:Nc+Hc, 1:Nc, k]   .=         field_2[region_E][1:Hc, 1:Nc, k]
                field_2[region][Nc+1:Nc+Hc, Nc+1, k]   .=         field_2[region_N][1:Hc, 1, k]
                field_2[region][1-Hc:0, 2:Nc+1, k]     .= reverse(field_1[region_W][1:Nc, Nc+1-Hc:Nc, k], dims=1)' * plmn
                field_2[region][1-Hc:0, 1, k]          .=         field_1[region_S][1, Nc+1-Hc:Nc, k] * plmn
                #- N + S Halo for field_2:
                field_2[region][1:Nc, Nc+1:Nc+Hc, k]   .= reverse(field_1[region_N][1:Hc, 1:Nc, k], dims=2)'
                field_2[region][1:Nc, 1-Hc:0, k]       .=         field_2[region_S][1:Nc, Nc+1-Hc:Nc, k]
            end
        else
            #- even face number (2, 4, 6):
            for k in -Hz+1:Nz+Hz
                #- E + W Halo for field_1:
                field_1[region][Nc+1:Nc+Hc, 1:Nc, k]   .= reverse(field_2[region_E][1:Nc, 1:Hc, k], dims=1)'
                field_1[region][1-Hc:0, 1:Nc, k]       .=         field_1[region_W][Nc+1-Hc:Nc, 1:Nc, k]
                #- N + S Halo for field_1:
                field_1[region][1:Nc, Nc+1:Nc+Hc, k]   .=         field_1[region_N][1:Nc, 1:Hc, k]
                field_1[region][Nc+1, Nc+1:Nc+Hc, k]    =         field_1[region_E][1, 1:Hc, k]
                field_1[region][2:Nc+1, 1-Hc:0, k]     .= reverse(field_2[region_S][Nc+1-Hc:Nc, 1:Nc, k], dims=2)' * plmn
                field_1[region][1, 1-Hc:0, k]           =         field_2[region_W][Nc+1-Hc:Nc, 1, k] * plmn
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
            end
            for k in -Hz+1:Nz+Hz
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

function fill_cubed_sphere_halo_regions!(field_1::CubedSphereField{<:Face, <:Face},
                                         field_2::CubedSphereField{<:Face, <:Face};
                                         signed = true)

    field_1.grid == field_2.grid || error("fields must be on the same grid")
    grid = field_1.grid

    Nx, Ny, Nz = size(grid)
    Hx, Hy, Hz = halo_size(grid)
    signed ? plmn = -1 : plmn = 1

    Nx == Ny || error("horizontal grid size Nx and Ny must be the same")
    Nc = Nx

    Hx == Hy || error("horizontal halo size Hx and Hy must be the same")
    Hc = Hx

    #-- one pass: only use interior-point values:
    for region in 1:6

        region_E, region_N, region_W, region_S = find_neighboring_panels(region)

        if mod(region, 2) == 1
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
                    field_1[region][1, Nc+2:Nc+Hc, k]   = reverse(field_1[region_W][1, Nc+2-Hc:Nc, k]) * plmn
                    field_2[region][1, Nc+2:Nc+Hc, k]   = reverse(field_2[region_W][1, Nc+2-Hc:Nc, k]) * plmn
                end
                # Note that the halos corresponding to the "missing" north-west corner of odd panels, specifically
                # field_1[region][1, Nc+1, k] and field_2[region][1, Nc+1, k], remain unfilled.
                #- S Halo:
                field_1[region][1:Nc, 1-Hc:0, k]       .=         field_1[region_S][1:Nc, Nc+1-Hc:Nc, k]
                field_2[region][1:Nc, 1-Hc:0, k]       .=         field_2[region_S][1:Nc, Nc+1-Hc:Nc, k]
                field_1[region][Nc+1, 1-Hc:0, k]        = reverse(field_2[region_E][2:Hc+1, 1, k])
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
                field_1[region][Nc+1, Nc+1:Nc+Hc, k]    =         field_1[region_E][1, 1:Hc, k]
                #- S Halo:
                field_1[region][2:Nc+1, 1-Hc:0, k]     .= reverse(field_2[region_S][Nc+1-Hc:Nc, 1:Nc, k], dims=2)' * plmn
                field_2[region][2:Nc, 1-Hc:0, k]       .= reverse(field_1[region_S][Nc+1-Hc:Nc, 2:Nc, k], dims=2)'
                field_1[region][1, 1-Hc:0, k]           =         field_2[region_W][Nc+1-Hc:Nc, 1, k] * plmn
                field_2[region][1, 1-Hc:0, k]           =         field_1[region_W][Nc+1-Hc:Nc, 1, k]
            end
        end
    end

    return nothing
end
