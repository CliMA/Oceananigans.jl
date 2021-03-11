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

# For all cca fields (1 to 5), the last row and last column (i.e., (33,:) & (:,33) are
# not used and are just zero.
# And for fields 11,13,16 : last column (:,33) is just zero.
# And for fields 12,14,15 : last row (33,:) is just zero.

using JLD2

# We'll only look at once face for now.
face = "001"

# Calculate the number of Float64 elements in the binary file.
cs32_bin = read("grid_cs32.face$face.bin")
N_numbers = Int(length(cs32_bin) / 8)

# Load binary data into a big array.
cs32_bin = Array{Float64}(undef, N_numbers)
read!("grid_cs32.face001.bin", cs32_bin)
cs32_bin = bswap.(cs32_bin)  # Reverse byte order since binary data is big-endian.

# Move data into 2D arrays/fields.
inds(n, N) = UnitRange((n-1) * N^2 + 1, n * N^2)
contents(bin, n, N) = reshape(bin[inds(n)], N, N)

N = 33  # CS32 arrays are 33×33

λᶜᶜᵃ = contents(cs32_bin, 1, N)
φᶜᶜᵃ = contents(cs32_bin, 2, N)

Δλᶜᶜᵃ = contents(cs32_bin, 3, N)
Δφᶜᶜᵃ = contents(cs32_bin, 4, N)

Azᶜᶜᵃ = contents(cs32_bin, 5, N)

λᶠᶠᵃ = contents(cs32_bin, 6, N)
φᶠᶠᵃ = contents(cs32_bin, 7, N)

Δλᶠᶠᵃ = contents(cs32_bin, 8, N)
Δφᶠᶠᵃ = contents(cs32_bin, 9, N)

Azᶠᶠᵃ = contents(cs32_bin, 10, N)

Δλᶠᶜᵃ = contents(cs32_bin, 11, N)
Δφᶜᶠᵃ = contents(cs32_bin, 12, N)

Azᶠᶜᵃ = contents(cs32_bin, 13, N)
Azᶜᶠᵃ = contents(cs32_bin, 14, N)

Δλᶜᶠᵃ = contents(cs32_bin, 15, N)
Δφᶠᶜᵃ = contents(cs32_bin, 16, N)

#####
##### Quick and dirty plots of all the variables
#####

using PyPlot

function quick_and_dirty_plot(var, var_name)
    @info "Plotting $var_name..."
    PyPlot.pcolormesh(var)
    PyPlot.title(var_name)
    PyPlot.colorbar()
    PyPlot.savefig("$var_name.png")
    PyPlot.close("all")
end

quick_and_dirty_plot(λᶜᶜᵃ, "λᶜᶜᵃ")
quick_and_dirty_plot(φᶜᶜᵃ, "φᶜᶜᵃ")
quick_and_dirty_plot(Δλᶜᶜᵃ, "Δλᶜᶜᵃ")
quick_and_dirty_plot(Δφᶜᶜᵃ, "Δφᶜᶜᵃ")
quick_and_dirty_plot(Azᶜᶜᵃ, "Azᶜᶜᵃ")
quick_and_dirty_plot(λᶠᶠᵃ, "λᶠᶠᵃ")
quick_and_dirty_plot(φᶠᶠᵃ, "φᶠᶠᵃ")
quick_and_dirty_plot(Δλᶠᶠᵃ, "Δλᶠᶠᵃ")
quick_and_dirty_plot(Δφᶠᶠᵃ, "Δφᶠᶠᵃ")
quick_and_dirty_plot(Azᶠᶠᵃ, "Azᶠᶠᵃ")
quick_and_dirty_plot(Δλᶠᶜᵃ, "Δλᶠᶜᵃ")
quick_and_dirty_plot(Δφᶜᶠᵃ, "Δφᶜᶠᵃ")
quick_and_dirty_plot(Azᶠᶜᵃ, "Azᶠᶜᵃ")
quick_and_dirty_plot(Azᶜᶠᵃ, "Azᶜᶠᵃ")
quick_and_dirty_plot(Δλᶜᶠᵃ, "Δλᶜᶠᵃ")
quick_and_dirty_plot(Δφᶠᶜᵃ, "Δφᶠᶜᵃ")
