using Oceananigans.Grids

using Statistics
using JLD2
using Printf
using CairoMakie
using DataDeps

using Oceananigans.Utils: prettytime

path = "https://github.com/CliMA/OceananigansArtifacts.jl/raw/ss/new_hydrostatic_data_after_cleared_bugs/quarter_degree_near_global_input_data/"

datanames = ["tau_x-1440x600-latitude-75",
             "tau_y-1440x600-latitude-75",
             ""]

dh = DataDep("quarter_degree_near_global_lat_lon",
    "Forcing data for global latitude longitude simulation",
    path * "bathymetry-1440x600.jld2")

DataDeps.register(dh)

datadep"quarter_degree_near_global_lat_lon"
datadep_path = @datadep_str "quarter_degree_near_global_lat_lon/bathymetry-1440x600.jld2"
file_bathymetry = jldopen(datadep_path)

output_prefix = "near_global_shallow_water_1440_600_surface"

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

#x, y, z = nodes((Center, Center, Center), grid)
x = grid.λᶜᵃᵃ[1:Nx]
y = grid.φᵃᶜᵃ[1:Ny]

bat = file_bathymetry["bathymetry"]
# Do not allow regions shallower than 10 meters depth
bat[bat .> -10] .= 0

iter = Observable(0)
iters = parse.(Int, keys(file["timeseries/t"]))
η′ = @lift(file["timeseries/h/" * string($iter)][:, :,       1] .+ bat)

title = @lift(@sprintf("Free-surface in Shallow Water Model at time = %s", prettytime(file["timeseries/t/" * string($iter)])))
fig = Figure(size=(1000, 600))
ax = Axis(fig[1,1], xlabel = "longitude", ylabel = "latitude", title=title)
heatmap_plot = heatmap!(ax, x, y, η′, colormap=:blues, nan_color = :black, colorrange=(9.9, 10.1))
Colorbar(fig[1,2], heatmap_plot, width=25)

display(fig)

record(fig, output_prefix * ".mp4", iters[2:end-3], framerate=12) do i
    @info "Plotting iteration $i of $(iters[end])..."
    iter[] = i
end

#=
# x 900 1100
# y 250 350
fig = Figure()
ax = Axis(fig[1,1], title = "free-surface", xlabel="x", ylabel="y")
heatmap_bat = heatmap!(ax, η557, colorrange=(0, 10.))
#heatmap_bat = heatmap!(ax, η557[900:1100, 250:350], colorrange=(9.9999, 10.0001))
Colorbar(fig[1,2], heatmap_bat)
save("η557.png", fig)


fig = Figure()
ax = Axis(fig[1,1], title = "free-surface", xlabel="x", ylabel="y")
heatmap_bat = heatmap!(ax, η557[990:1030, 290:340], colorrange=(0, 10))
Colorbar(fig[1,2], heatmap_bat)
save("η557_closeup1.png", fig)

fig = Figure()
ax = Axis(fig[1,1], title = "bathymetry", xlabel="x", ylabel="y")
heatmap_bat = heatmap!(ax, bat[990:1030, 290:340])#, colorrange=(0, 10))
Colorbar(fig[1,2], heatmap_bat)
save("bat_closeup1.png", fig)

fig = Figure()
ax = Axis(fig[1,1], title = "bathymetry slice", xlabel="x", ylabel="z")
lines!(bat[990:1030, 320])
save("bat_slice_closeup1.png", fig)

=#