using GLMakie
using JLD2

filename = "bathymetry-360x150-latitude-75.0.jld2"
file = jldopen(filename)
h = file["bathymetry"]
close(file)
h[h .> 0] .= NaN

fig = Figure(resolution=(1800, 800))
ax = Axis(fig[1, 1], xlabel="Longitude", ylabel="Latitude")
hm = heatmap!(ax, h)
Colorbar(fig[1, 2], hm, label="Elevation (m)")
display(fig)

