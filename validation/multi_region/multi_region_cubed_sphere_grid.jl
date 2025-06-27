# Comparison of coordinates and metrics of a 32x32 cubed sphere grid with 4 halos
# to their counterparts from MITgcm

using Oceananigans, DataDeps, JLD2, Statistics

Nx, Ny, Nz = 32, 32, 1
cs_grid = ConformalCubedSphereGrid(; panel_size = (Nx, Ny, Nz), z = (-1, 0), radius=6370e3, horizontal_direction_halo = 4,
                                     z_halo = 1)
Hx, Hy, Hz = cs_grid.Hx, cs_grid.Hy, cs_grid.Hz

cs32_4 = DataDep("cubed_sphere_32_grid_with_4_halos",
                 "Conformal cubed sphere grid with 32×32 cells on each face and 4 halos on each side",
                 "https://github.com/glwagner/OceananigansArtifacts.jl/raw/main/cubed_sphere_grids/cs32_with_4_halos/cubed_sphere_32_grid_with_4_halos.jld2"
                 )

DataDeps.register(cs32_4)
grid_filepath = datadep"cubed_sphere_32_grid_with_4_halos/cubed_sphere_32_grid_with_4_halos.jld2"

cs_grid_MITgcm = ConformalCubedSphereGrid(grid_filepath;
                                          Nz = 1,
                                          z = (-1, 0),
                                          panel_halo = (4, 4, 1),
                                          radius = 6370e3)

"""
    same_longitude_at_poles!(grid1, grid2)

Change the longitude values in `grid1` that correspond to points situated _exactly_ at
the poles so that they match the corresponding longitude values of `grid2`.
"""
function same_longitude_at_poles!(grid1, grid2)
    for region in 1:6
        grid1[region].λᶠᶠᵃ[grid2[region].φᶠᶠᵃ .== +90]= grid2[region].λᶠᶠᵃ[grid2[region].φᶠᶠᵃ .== +90]
        grid1[region].λᶠᶠᵃ[grid2[region].φᶠᶠᵃ .== -90]= grid2[region].λᶠᶠᵃ[grid2[region].φᶠᶠᵃ .== -90]
    end
    return nothing
end

same_longitude_at_poles!(cs_grid_MITgcm, cs_grid)

coords = (:λᶜᶜᵃ, :λᶠᶠᵃ, :φᶜᶜᵃ, :φᶠᶠᵃ)

metrics = (:Δxᶜᶜᵃ, :Δxᶠᶜᵃ, :Δxᶜᶠᵃ, :Δxᶠᶠᵃ,
           :Δyᶜᶜᵃ, :Δyᶠᶜᵃ, :Δyᶜᶠᵃ, :Δyᶠᶠᵃ,
           :Azᶜᶜᵃ, :Azᶠᶜᵃ, :Azᶜᶠᵃ, :Azᶠᶠᵃ)

vars = (coords..., metrics...)

var_diffs = Tuple(Symbol(string(var) * "_difference_MITgcm") for var in vars)

for var_diff in var_diffs
    eval(:($var_diff = zeros(Nx+2Hx, Ny+2Hy, 6)))
end

jldopen("cs_grid_difference_with_MITgcm.jld2", "w") do file
    for panel in 1:6
        for (counter, var) in enumerate(vars)
            var_diff = var_diffs[counter]
            var_diff_name = string(var_diff)

            expr = quote
                $var_diff[:, :, $panel] = $cs_grid[$panel].$var - $cs_grid_MITgcm[$panel].$var
                $file[$var_diff_name * "/" * string($panel)] = $var_diff[:, :, $panel]
            end
            eval(expr)
        end
    end
end

"""
    zero_out_halos!(array, (Nx, Ny), (Hx, Hy))

Return the `array` after zeroing out all data that correspond to corner halo points.
"""
function zero_out_corner_halos!(array, (Nx, Ny), (Hx, Hy))
    array[1:Hx, 1:Hy] .= 0
    array[1:Hx, Ny+Hy+1:Ny+2Hy] .= 0
    array[Nx+Hx+1:Nx+2Hx, 1:Hy] .= 0
    array[Nx+Hx+1:Nx+2Hx, Ny+Hy+1:Ny+2Hy] .= 0

    return nothing
end

"""
    zero_out_halos!(array, (Nx, Ny), (Hx, Hy))

Return the `array` after zeroing out all data that correspond to halo points.
"""
function zero_out_halos!(array, (Nx, Ny), (Hx, Hy))
    array[1:Hx, :] .= 0
    array[Nx+Hx+1:Nx+2Hx, :] .= 0
    array[:, 1:Hy] .= 0
    array[:, Ny+Hy+1:Ny+2Hy] .= 0

    return nothing
end

"""
    mean_func(array, (Nx, Ny), (Hx, Hy), func=abs)

Return `mean(func, array)`.
"""
function mean_func(array, (Nx, Ny), (Hx, Hy), func=abs)
    zero_out_corner_halos!(array, (Nx, Ny), (Hx, Hy))
    return mean(func, array)
end

file = jldopen("cs_grid_difference_with_MITgcm.jld2")

for var in vars
    varname = string(var)
    for region in 1:6
        array = deepcopy(file[varname * "_difference_MITgcm/" * string(region)])
        error = mean_func(array, (32, 32), (4, 4), abs)
        @info varname * " panel " * string(region) * ": " * string(error)
    end
end

close(file)
