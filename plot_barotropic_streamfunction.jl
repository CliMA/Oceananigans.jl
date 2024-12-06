using CairoMakie
using Oceananigans

filename = "doublegyre_RiBasedVerticalDiffusivity_streamfunction"
FILE_DIR = "./Output/$(filename)/"

Ψ_data = FieldTimeSeries("$(FILE_DIR)/doublegyre_Ri_based_vertical_diffusivity_2Pr_streamfunction.jld2", "Ψ")

Nx = Ψ_data.grid.Nx
Ny = Ψ_data.grid.Ny

xF = Ψ_data.grid.xᶠᵃᵃ[1:Nx+1]
yC = Ψ_data.grid.yᵃᶜᵃ[1:Ny]

Nt = length(Ψ_data)
times = Ψ_data.times / 24 / 60^2 / 365
#%%
timeframe = 31
Ψ_frame = interior(Ψ_data[timeframe], :, :, 1) ./ 1e6
clim = maximum(abs, Ψ_frame)
N_levels = 16
levels = range(-clim, stop=clim, length=N_levels)
fig = Figure(size=(800, 800))
ax = Axis(fig[1, 1], xlabel="x (m)", ylabel="y (m)", title="Ri-based Vertical Diffusivity, Yearly-Averaged Barotropic streamfunction Ψ, Year $(times[timeframe])")
cf = contourf!(ax, xF, yC, Ψ_frame, levels=levels, colormap=Reverse(:RdBu_11))
Colorbar(fig[1, 2], cf, label="Ψ (Sv)")
tightlimits!(ax)
save("$(FILE_DIR)/barotropic_streamfunction_$(timeframe).png", fig, px_per_unit=4)
display(fig)
#%%