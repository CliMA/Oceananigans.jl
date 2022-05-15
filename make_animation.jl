using Oceananigans.Grids
using Oceananigans.Utils: prettytime, hours, day, days, years

using Statistics
using JLD2
using Printf
using GLMakie
#using CairoMakie

output_prefix = "near_global_lat_lon_1440_600__fine_surface"

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

iter = Observable(0)
iters = parse.(Int, keys(file["timeseries/t"]))
ζ′ = @lift(file["timeseries/ζ/" * string($iter)][:, :, 1])

clims_ζ = @lift 1.1 .* extrema(file["timeseries/ζ/" * string($iter)][:])

title = @lift(@sprintf("Vorticity in Shallow Water Model at time = %s", prettytime(file["timeseries/t/" * string($iter)])))
fig = Figure(resolution = (2000, 1000))
ax = Axis(fig[1,1], xlabel = "longitude", ylabel = "latitude", title=title)
heatmap!(ax, x, y, ζ′, colormap=:balance, colorrange=(-2e-5, 2e-5))

display(fig)

record(fig, output_prefix * ".mp4", iters[2:end], framerate=12) do i
    @info "Plotting iteration $i of $(iters[end])..."
    iter[] = i
end



