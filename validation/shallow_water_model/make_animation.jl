using Oceananigans.Grids
using Oceananigans.Utils: prettytime, hours, day, days, years

using Statistics
using JLD2
using Printf
using CairoMakie

output_prefix = "near_global_shallow_water_1440_600_surface"

filepath = output_prefix * ".jld2"

# file = jldopen(filepath)
file = jldopen("test.jld2")

Nx = file["grid/underlying_grid/Nx"]
Ny = file["grid/underlying_grid/Ny"]
Lλ = file["grid/underlying_grid/Lx"]
Lφ = file["grid/underlying_grid/Ly"]
Lz = file["grid/underlying_grid/Lz"]

grid = LatitudeLongitudeGrid(size = (Nx, Ny, 1),
                             longitude = (-180, 180),
                             latitude = (-Lφ/2, Lφ/2),
                             z = (-Lz, 0))

x, y, z = nodes((Center, Center, Center), grid)

smoothed_bathymetry = jldopen("smooth-bathymetry-2.jld2")
bat3 = smoothed_bathymetry["bathymetry"]
bat2 = deepcopy(bat3)
bat  = deepcopy(bat3)
bat2[ bat2 .> 0 ] .= NaN
bat[ bat .> 0 ] .= NaN
bat[ bat .< 0 ] .= 0.0

iter = Observable(0)
iters = parse.(Int, keys(file["timeseries/t"]))
ζ′ = @lift(file["timeseries/ζ/" * string($iter)][:, 1:end-1, 1] .+ bat)
h′ = @lift(file["timeseries/h/" * string($iter)][:, :,       1] .+ bat2)

clims_ζ = @lift 1.1 .* extrema(file["timeseries/ζ/" * string($iter)][:])

title = @lift(@sprintf("Vorticity in Shallow Water Model at time = %s", prettytime(file["timeseries/t/" * string($iter)])))
fig = Figure(resolution = (2000, 600))
ax = Axis(fig[1,1], xlabel = "longitude", ylabel = "latitude", title=title)
heatmap_plot = heatmap!(ax, x, y, ζ′, colormap=:blues, nan_color = :black, colorrange=(-5e-6, 5e-6))
Colorbar(fig[1,2], heatmap_plot, width=25)

ax = Axis(fig[1,3], xlabel = "longitude", ylabel = "latitude", title=title)
heatmap_plot = heatmap!(ax, x, y, h′, colormap=:hot, nan_color = :black, colorrange = (9.7, 10.3))
Colorbar(fig[1,4], heatmap_plot, width=25)

display(fig)

record(fig, output_prefix * ".mp4", iters[2:end-3], framerate=12) do i
    @info "Plotting iteration $i of $(iters[end])..."
    iter[] = i
end
