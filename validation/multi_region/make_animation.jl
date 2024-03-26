using Oceananigans.Grids

using Statistics
using JLD2
using Printf 
using CairoMakie
using DataDeps

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

output_prefix = "near_global_1440_600_48_fine_surface"

filepath = output_prefix * ".jld2"

filepath = "near_global_lat_lon_1440_600_48_fine_surface.jld2"
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

bat3 = file_bathymetry["bathymetry"]
bat2 = deepcopy(bat3)
bat  = deepcopy(bat3)
bat2[ bat2 .> 0 ] .= NaN
bat[ bat .> 0 ] .= NaN
bat[ bat .< 0 ] .= 0.0

iter = Observable(0)
iters = parse.(Int, keys(file["timeseries/t"]))
u′ = @lift(file["timeseries/u/" * string($iter)][:, :, 1] .+ bat)
T′ = @lift(file["timeseries/T/" * string($iter)][:, :, 1] .- bat2)

clims_u = @lift 1.1 .* extrema(file["timeseries/u/" * string($iter)][:])

title1 = @lift(@sprintf("Zonal Velocity HY Model at time = %s", prettytime(file["timeseries/t/" * string($iter)])))
fig = Figure(size=(2000, 600))
ax = Axis(fig[1,1], xlabel = "longitude", ylabel = "latitude", title=title1)
heatmap_plot = heatmap!(ax, x, y, u′, colormap=:blues, nan_color = :black)
Colorbar(fig[1,2], heatmap_plot, width=25)

title2 = @lift(@sprintf("Temperature of HY Model at time = %s", prettytime(file["timeseries/t/" * string($iter)])))
ax = Axis(fig[1,3], xlabel = "longitude", ylabel = "latitude", title=title2)
heatmap_plot = heatmap!(ax, x, y, T′, colormap=:hot, nan_color = :black)
Colorbar(fig[1,4], heatmap_plot, width=25)

display(fig)

record(fig, output_prefix * ".mp4", iters[2:end-3], framerate=12) do i
    @info "Plotting iteration $i of $(iters[end])..."
    iter[] = i
end
