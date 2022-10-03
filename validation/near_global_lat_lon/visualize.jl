using Oceananigans.Grids
using Oceananigans.Utils: prettytime, hours, day, days, years

using Statistics
using JLD2
using Printf
using CairoMakie

output_prefix = "near_global_lat_lon_1440_600_1_fine_surface"
filepath = output_prefix * ".jld2"
file = jldopen(filepath)

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

bottom = Float32.(file["grid/immersed_boundary/bottom_height"][4:end-3,3:end-3,1])
bottom[ bottom .>0 ] .=  NaN
bottom[ bottom .<0 ] .= 0.0

iter = Observable(0)
iters = parse.(Int, keys(file["timeseries/t"]))
ζ′ = @lift file["timeseries/ζ/" * string($iter)][:, :, 1]
title = @lift(@sprintf("Surface Vorticity in Hydrostatic Model at time = %s", prettytime(file["timeseries/t/" * string($iter)])))
fig = Figure(resolution = (2000, 1000))
ax = Axis(fig[1,1], xlabel = "longitude", ylabel = "latitude", title=title)
heatmap_plot = heatmap!(ax, ζ′, colormap=:balance, colorrange=(-1e-6, 1e-6), nan_color=:black)
Colorbar(fig[1,2], heatmap_plot , width=25)
display(fig)

record(fig, output_prefix * ".mp4", iters[2:end], framerate=6) do i
    @info "Plotting iteration $i of $(iters[end])..."
    iter[] = i
end
