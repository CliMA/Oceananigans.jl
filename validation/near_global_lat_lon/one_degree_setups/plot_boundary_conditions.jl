using JLD2
using GLMakie

include("one_degree_artifacts.jl")
bathymetry_path = download_bathymetry()
boundary_conditions_path = download_boundary_conditions()

file = jldopen(bathymetry_path)
h = file["bathymetry"]
land = h .> 0
close(file)

ρ₀ = 1025
φ = -74.5:74.5
file = jldopen(boundary_conditions_path)

τˣ = file["τˣ"] ./ ρ₀
τʸ = file["τʸ"] ./ ρ₀
Tˢ = file["Tˢ"]
Sˢ = file["Sˢ"]

close(file)

τˣ[isnan.(τˣ)] .= 0
τʸ[isnan.(τʸ)] .= 0

τlim = 0.8 * max(maximum(abs, τˣ), maximum(abs, τʸ))

for n = 1:12
    view(τˣ, :, :, n)[land] .= NaN
    view(τʸ, :, :, n)[land] .= NaN
    view(Tˢ, :, :, n)[land] .= NaN
    view(Sˢ, :, :, n)[land] .= NaN
end

fig = Figure(resolution=(2000, 900))
axx = Axis(fig[2, 2], xlabel="Longitude", ylabel="Latitude", title="Zonal surface stress")
axy = Axis(fig[2, 3], xlabel="Longitude", ylabel="Latitude", title="Meridonal surface stress")
axT = Axis(fig[3, 2], xlabel="Longitude", ylabel="Latitude")
axS = Axis(fig[3, 3], xlabel="Longitude", ylabel="Latitude")
slider = Slider(fig[4, 2:3], range=1:12, startvalue=1)
n = slider.value

title = @lift string("Boundary conditions in month ", $n)
Label(fig[1, 2:3], title)

τˣn = @lift view(τˣ, :, :, $n)
τʸn = @lift view(τʸ, :, :, $n)
Tn = @lift view(Tˢ, :, :, $n)
Sn = @lift view(Sˢ, :, :, $n)

hmx = heatmap!(axx, τˣn, colorrange=(-τlim, τlim), colormap=:redblue)
hmy = heatmap!(axy, τʸn, colorrange=(-τlim, τlim), colormap=:redblue)
hmT = heatmap!(axT, Tn, colormap=:thermal)
hmS = heatmap!(axS, Sn, colormap=:haline)

Colorbar(fig[2, 4], hmx, label="Stress (m² s⁻²)")
Colorbar(fig[3, 1], hmT, label="Surface temperature (ᵒC)", flipaxis=false)
Colorbar(fig[3, 4], hmS, label="Surface salinity (psu)")

display(fig)

record(fig, "one_degree_boundary_conditions.mp4", 1:12, framerate=1) do nn
    n[] = nn
end
