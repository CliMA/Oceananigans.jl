# Email from Jean-Michel:
#
# I made a little tar file (-czf) of cs32 grid files (the same as in
# https://github.com/MITgcm/MITgcm/blob/master/verification/tutorial_held_suarez_cs/input
# with a litte matlab script that load them.
#
# Each file (big-endian, 64 bits) contains 18 reccords of 33 x 33 values, with:
# rec  | MITgcm | Description
#   #  |  name  |
#   1  : xC  <-> longitude (in -180:+180) at grid-cell center (cca)
#   2  : yC  <-> latitude  at grid-cell center (cca)
#   3  : dxF <-> dx (in m) at center (cca)
#   4  : dyF <-> dy (in m) at center (cca)
#   5  : rAc <-> Area (in m^2) at center ( zA^cca )
#   6  : xG  <-> longitude (in -180:+180) at grid-cell corner (ffa)
#   7  : yG  <-> latitude  at grid-cell corner (ffa)
#   8  : dxV <-> dx (in m) at grid-cell corner (ffa)
#   9  : dyU <-> dy (in m) at grid-cell corner (ffa)
#  10  : rAz <-> Area (in m^2) at grid-cell corner  ( zA^ffa )
#  11  : dxC <-> dx (in m) at U pt (fca)
#  12  : dyC <-> dy (in m) at V pt (cfa)
#  13  : rAw <-> Area (in m^2) at U pt ( zA^fca )
#  14  : rAs <-> Area (in m^2) at V pt ( zA^cfa )
#  15  : dxG <-> dx (in m) at V pt (cfa)
#  16  : dyG <-> dy (in m) at U pt (fca)
#  17  : cosine of the local angle between the grid orientation (e.g., U(1) velocity) and West-East direction at grid-cell center
#  18  : sine of the local angle between the grid orientation (e.g., U(1) velocity) and West-East direction at grid-cell center
#
# For all cca fields (1 to 5), the last row and last column (i.e., (33,:) & (:,33) are
# not used and are just zero.
# And for fields 11,13,16 : last column (:,33) is just zero.
# And for fields 12,14,15 : last row (33,:) is just zero.

using JLD2

const N_records = 18

function read_cubed_sphere_face(filepath)
    binary_data = read(filepath)

    # Calculate the number of Float64 elements in the binary file.
    N_numbers = Int(length(binary_data) / 8)

    # Calculate the size of the grid (number of faces)
    N = Nx = Ny = √(N_numbers / N_records) |> Int

    # Load binary data into a big array.
    cubed_sphere_face_data = Array{Float64}(undef, N_numbers)
    read!(filepath, cubed_sphere_face_data)

    # Reverse byte order since binary data is big-endian.
    cubed_sphere_face_data = bswap.(cubed_sphere_face_data)

    ## Move data into 2D arrays/fields.

    inds(n, N) = UnitRange((n-1) * N^2 + 1, n * N^2)
    contents(bin, n, N) = reshape(bin[inds(n, N)], N, N)

    cubed_sphere_variables = (
         λᶜᶜᵃ = contents(cubed_sphere_face_data,  1, N),
         φᶜᶜᵃ = contents(cubed_sphere_face_data,  2, N),
        Δxᶜᶜᵃ = contents(cubed_sphere_face_data,  3, N),
        Δyᶜᶜᵃ = contents(cubed_sphere_face_data,  4, N),
        Azᶜᶜᵃ = contents(cubed_sphere_face_data,  5, N),
         λᶠᶠᵃ = contents(cubed_sphere_face_data,  6, N),
         φᶠᶠᵃ = contents(cubed_sphere_face_data,  7, N),
        Δxᶠᶠᵃ = contents(cubed_sphere_face_data,  8, N),
        Δyᶠᶠᵃ = contents(cubed_sphere_face_data,  9, N),
        Azᶠᶠᵃ = contents(cubed_sphere_face_data, 10, N),
        Δxᶠᶜᵃ = contents(cubed_sphere_face_data, 11, N),
        Δyᶜᶠᵃ = contents(cubed_sphere_face_data, 12, N),
        Azᶠᶜᵃ = contents(cubed_sphere_face_data, 13, N),
        Azᶜᶠᵃ = contents(cubed_sphere_face_data, 14, N),
        Δxᶜᶠᵃ = contents(cubed_sphere_face_data, 15, N),
        Δyᶠᶜᵃ = contents(cubed_sphere_face_data, 16, N)
    )

    return cubed_sphere_variables
end

#####
##### Convert grid_cs32.face*.bin files to JLD2
#####

jldopen("cubed_sphere_32_grid.jld2", "w") do file
    for (face_number, face) in enumerate(["001", "002", "003", "004", "005", "006"])
        filepath = "grid_cs32.face$face.bin"
        @info "Converting $filepath..."

        cubed_sphere_vars = read_cubed_sphere_face(filepath)

        for (var_name, var) in pairs(cubed_sphere_vars)
            file["face$face_number/$var_name"] = var
        end
    end
end
