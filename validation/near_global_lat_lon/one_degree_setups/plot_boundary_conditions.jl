using JLD2
using GLMakie

include("one_degree_artifacts.jl")
# bathymetry_path = download_bathymetry() # not needed because we uploaded to repo
boundary_conditions_path = download_boundary_conditions()

file = jldopen(bathymetry_path)
h = file["bathymetry"]
land = h .> 0
wet_Nx = sum(h -> h < 0, h, dims=1)
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
hmT = heatmap!(axT, Tn, colorrange=(-1, 31), colormap=:thermal)
hmS = heatmap!(axS, Sn, colorrange=(25, 38), colormap=:haline)

Colorbar(fig[2, 4], hmx, label="Stress (m² s⁻²)")
Colorbar(fig[3, 1], hmT, label="Surface temperature (ᵒC)", flipaxis=false)
Colorbar(fig[3, 4], hmS, label="Surface salinity (psu)")

display(fig)

record(fig, "one_degree_boundary_conditions.mp4", 1:12, framerate=1) do nn
    n[] = nn
end

#####
##### Zonal means
#####

filternan(t) = ifelse(isnan(t), 0.0, t)
zonal_mean_τˣ = sum(filternan, τˣ, dims=1) ./ wet_Nx
zonal_mean_τʸ = sum(filternan, τʸ, dims=1) ./ wet_Nx
zonal_mean_Tˢ = sum(filternan, Tˢ, dims=1) ./ wet_Nx
zonal_mean_Sˢ = sum(filternan, Sˢ, dims=1) ./ wet_Nx

zonal_mean_τˣ = dropdims(zonal_mean_τˣ, dims=1)'
zonal_mean_τʸ = dropdims(zonal_mean_τʸ, dims=1)'
zonal_mean_Tˢ = dropdims(zonal_mean_Tˢ, dims=1)'
zonal_mean_Sˢ = dropdims(zonal_mean_Sˢ, dims=1)'

fig = Figure(resolution=(2000, 900))

axx = Axis(fig[1, 2], xlabel="Month", ylabel="Latitude", title="Zonally-averaged zonal surface stress")
axy = Axis(fig[1, 3], xlabel="Month", ylabel="Latitude", title="Zonally-averaged meridonal surface stress")
axT = Axis(fig[2, 2], xlabel="Month", ylabel="Latitude")
axS = Axis(fig[2, 3], xlabel="Month", ylabel="Latitude")

τlim = 1.5e-4
hmx = heatmap!(axx, zonal_mean_τˣ, colorrange=(-τlim, τlim), colormap=:redblue)
hmy = heatmap!(axy, zonal_mean_τʸ, colorrange=(-τlim, τlim), colormap=:redblue)
hmT = heatmap!(axT, zonal_mean_Tˢ, colorrange=(-1, 31), colormap=:thermal)
hmS = heatmap!(axS, zonal_mean_Sˢ, colorrange=(25, 38), colormap=:haline)

Colorbar(fig[1, 4], hmx, label="Stress (m² s⁻²)")
Colorbar(fig[2, 1], hmT, label="Surface temperature (ᵒC)", flipaxis=false)
Colorbar(fig[2, 4], hmS, label="Surface salinity (psu)")

display(fig)

save("zonally_averaged_one_degree_boundary_conditions.png", fig)

