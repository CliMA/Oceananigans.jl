using JLD2
using GLMakie

include("one_degree_artifacts.jl")
include("one_degree_interface_heights.jl")
z_interfaces = one_degree_interface_heights()
z_centers = (z_interfaces[1:end-1] .+ z_interfaces[2:end]) ./ 2

filename = "bathymetry-360x150-latitude-75.0.jld2"
file = jldopen(filename)
h = file["bathymetry"]
close(file)
land = h .> 0

initial_condition_path = download_initial_condition()
file = jldopen(initial_condition_path)
Tᵢ = file["T"]
Nz = size(Tᵢ, 3)

for k in 1:Nz
    z = z_centers[k]
    earth = h .> z
    view(Tᵢ, :, :, k)[earth] .= NaN
end

fig = Figure(resolution=(1800, 800))
ax = Axis(fig[2, 1])
slider = Slider(fig[3, 1:2], range=1:Nz, startvalue=Nz)
k = slider.value

depth = @lift z_centers[$k]
title = @lift string("Temperature (ᵒC) at z = ", $depth, " m")
Label(fig[1, 1:2], title)

Tᵏ = @lift view(Tᵢ, :, :, $k)

hm = heatmap!(ax, Tᵏ, colorrange=(-2, 32), colormap=:thermal)
Colorbar(fig[2, 2], hm, label="Temperature (ᵒC)")

display(fig)

record(fig, "one_degree_initial_condition.mp4", 1:Nz, framerate=12) do kk
    k[] = kk
end
