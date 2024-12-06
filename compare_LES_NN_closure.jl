using Oceananigans
using Statistics
using JLD2
using CairoMakie

# LES_FILE_DIR = "./NN_2D_channel_horizontal_convection_0.0003_LES.jld2"

# u_data_LES = FieldTimeSeries(LES_FILE_DIR, "u", backend=OnDisk())
# v_data_LES = FieldTimeSeries(LES_FILE_DIR, "v", backend=OnDisk())
# T_data_LES = FieldTimeSeries(LES_FILE_DIR, "T", backend=OnDisk())
# S_data_LES = FieldTimeSeries(LES_FILE_DIR, "S", backend=OnDisk())

# yC_LES = ynodes(T_data_LES.grid, Center())
# yF_LES = ynodes(T_data_LES.grid, Face())

# zC_LES = znodes(T_data_LES.grid, Center())
# zF_LES = znodes(T_data_LES.grid, Face())

# Nt_LES = findfirst(x -> x ≈ end_time, T_data_LES.times)

# Δy_LES = T_data_LES.grid.Ly / T_data_LES.grid.Ny
# Δz_LES = T_data_LES.grid.Lz / T_data_LES.grid.Nz

MODEL_FILE_DIR = "./NN_closure_2D_channel.jld2"

u_data_model = FieldTimeSeries(MODEL_FILE_DIR, "u")
v_data_model = FieldTimeSeries(MODEL_FILE_DIR, "v")
T_data_model = FieldTimeSeries(MODEL_FILE_DIR, "T")
S_data_model = FieldTimeSeries(MODEL_FILE_DIR, "S")

end_time = 23 * 60^2 * 24

Ny_model = T_data_model.grid.Ny
Nz_model = T_data_model.grid.Nz

Δy_model = T_data_model.grid.Ly / Ny_model
Δz_model = T_data_model.grid.Lz / Nz_model

yC_model = ynodes(T_data_model.grid, Center())
zC_model = znodes(T_data_model.grid, Center())

# coarse_ratio_y = Int(Δy_model / Δy_LES)
# coarse_ratio_z = Int(Δz_model / Δz_LES)

# function coarsen_dataᵃᶜᵃ(data_LES, coarse_ratio_y, coarse_ratio_z, Ny_model, Nz_model, Nt_LES)
#     Ny_LES = data_LES.grid.Ny
#     Nz_LES = data_LES.grid.Nz
#     data_LES_coarse = zeros(1, Ny_model, Nz_model, Nt_LES)
#     LES_temp = zeros(Ny_LES, Nz_LES)
    
#     for nt in 1:Nt_LES
#         LES_temp .= interior(data_LES[nt], 1, :, :)
#         Threads.@threads for j in axes(data_LES_coarse, 2)
#             @info "nt = $nt, Processing j = $j"
#             for k in axes(data_LES_coarse, 3)
#                 data_LES_coarse[1, j, k, nt] = mean(LES_temp[(j-1)*coarse_ratio_y+1:j*coarse_ratio_y, (k-1)*coarse_ratio_z+1:k*coarse_ratio_z])
#             end
#         end
#     end

#     return data_LES_coarse
# end
    
# T_data_LES_coarse = coarsen_dataᵃᶜᵃ(T_data_LES, coarse_ratio_y, coarse_ratio_z, Ny_model, Nz_model, Nt_LES)
# S_data_LES_coarse = coarsen_dataᵃᶜᵃ(S_data_LES, coarse_ratio_y, coarse_ratio_z, Ny_model, Nz_model, Nt_LES)
# u_data_LES_coarse = coarsen_dataᵃᶜᵃ(u_data_LES, coarse_ratio_y, coarse_ratio_z, Ny_model, Nz_model, Nt_LES)

# function coarsen_dataᵃᶠᵃ(data_LES, coarse_ratio_y, coarse_ratio_z, Ny_model, Nz_model, Nt)
#     Ny_LES = data_LES.grid.Ny
#     Nz_LES = data_LES.grid.Nz

#     dataᵃᶠᵃ_temp = zeros(Ny_LES+1, Nz_LES)
#     dataᵃᶜᵃ_temp = zeros(Ny_LES, Nz_LES)
    
#     data_coarse = zeros(1, Ny_model, Nz_model, Nt)
    
#     for nt in 1:Nt
#         @info "nt = $nt, Interpolating for LES data"
#         dataᵃᶠᵃ_temp .= interior(data_LES[nt], 1, :, :)
#         Threads.@threads for j in 1:Ny_LES
#         # for j in 1:Ny_LES
#             for k in 1:Nz_LES
#                 dataᵃᶜᵃ_temp[j, k] = mean(dataᵃᶠᵃ_temp[j:j+1, k])
#             end
#         end

#         @info "nt = $nt, Coarsening data"
#         Threads.@threads for j in axes(data_coarse, 2)
#         # for j in axes(data_coarse, 2)
#             for k in axes(data_coarse, 3)
#                 data_coarse[1, j, k, nt] = mean(dataᵃᶜᵃ_temp[(j-1)*coarse_ratio_y+1:j*coarse_ratio_y, (k-1)*coarse_ratio_z+1:k*coarse_ratio_z])
#             end
#         end
#     end

#     return data_coarse
# end

# v_data_LES_coarse = coarsen_dataᵃᶠᵃ(v_data_LES, coarse_ratio_y, coarse_ratio_z, Ny_model, Nz_model, Nt_LES)
# v_data_model_coarse = coarsen_dataᵃᶠᵃ(v_data_model, 1, 1, Ny_model, Nz_model, Nt_LES)

# u_data_model_coarse = interior(u_data_model)
# T_data_model_coarse = interior(T_data_model)
# S_data_model_coarse = interior(S_data_model)

# jldopen("./LES_NDE_FC_Qb_absf_24simnew_2layer_128_relu_2Pr_coarsened.jld2", "w") do file
#     file["u_LES"] = u_data_LES_coarse
#     file["v_LES"] = v_data_LES_coarse
#     file["T_LES"] = T_data_LES_coarse
#     file["S_LES"] = S_data_LES_coarse
#     file["u_NN_model"] = u_data_model_coarse
#     file["v_NN_model"] = v_data_model_coarse
#     file["T_NN_model"] = T_data_model_coarse
#     file["S_NN_model"] = S_data_model_coarse
# end

FILE_DIR = "./LES_NDE_FC_Qb_absf_24simnew_2layer_128_relu_2Pr_coarsened.jld2"

u_data_LES_coarse, v_data_LES_coarse, T_data_LES_coarse, S_data_LES_coarse, u_data_model_coarse, v_data_model_coarse, T_data_model_coarse, S_data_model_coarse = jldopen(FILE_DIR, "r") do file
    u_data_LES_coarse = file["u_LES"]
    v_data_LES_coarse = file["v_LES"]
    T_data_LES_coarse = file["T_LES"]
    S_data_LES_coarse = file["S_LES"]
    u_data_model_coarse = file["u_NN_model"]
    v_data_model_coarse = file["v_NN_model"]
    T_data_model_coarse = file["T_NN_model"]
    S_data_model_coarse = file["S_NN_model"]
    return u_data_LES_coarse, v_data_LES_coarse, T_data_LES_coarse, S_data_LES_coarse, u_data_model_coarse, v_data_model_coarse, T_data_model_coarse, S_data_model_coarse
end

Nt = size(u_data_LES_coarse, 4)
#%%
fig = CairoMakie.Figure(size = (2000, 900))
axu_LES = CairoMakie.Axis(fig[1, 1], xlabel = "y (m)", ylabel = "z (m)", title = "u (LES) m/s")
axv_LES = CairoMakie.Axis(fig[1, 3], xlabel = "y (m)", ylabel = "z (m)", title = "v (LES) m/s")
axT_LES = CairoMakie.Axis(fig[1, 5], xlabel = "y (m)", ylabel = "z (m)", title = "Temperature (LES) °C")
axS_LES = CairoMakie.Axis(fig[1, 7], xlabel = "y (m)", ylabel = "z (m)", title = "Salinity (LES) psu")
axu_model = CairoMakie.Axis(fig[2, 1], xlabel = "y (m)", ylabel = "z (m)", title = "u (NN closure) (m/s)")
axv_model = CairoMakie.Axis(fig[2, 3], xlabel = "y (m)", ylabel = "z (m)", title = "v (NN closure) (m/s)")
axT_model = CairoMakie.Axis(fig[2, 5], xlabel = "y (m)", ylabel = "z (m)", title = "Temperature (NN closure) °C")
axS_model = CairoMakie.Axis(fig[2, 7], xlabel = "y (m)", ylabel = "z (m)", title = "Salinity (NN closure) psu")
axΔu = CairoMakie.Axis(fig[3, 1], xlabel = "y (m)", ylabel = "z (m)", title = "u (LES) - u(NN closure) (m/s)")
axΔv = CairoMakie.Axis(fig[3, 3], xlabel = "y (m)", ylabel = "z (m)", title = "v (LES) - v(NN closure) (m/s)")
axΔT = CairoMakie.Axis(fig[3, 5], xlabel = "y (m)", ylabel = "z (m)", title = "Temperature (LES) - Temperature (NN closure) (°C)")
axΔS = CairoMakie.Axis(fig[3, 7], xlabel = "y (m)", ylabel = "z (m)", title = "Salinity (LES) - Salinity (NN closure) (psu)")

n = Observable(1)
u_LESₙ = @lift u_data_LES_coarse[1, :, :, $n]
v_LESₙ = @lift v_data_LES_coarse[1, :, :, $n]
T_LESₙ = @lift T_data_LES_coarse[1, :, :, $n]
S_LESₙ = @lift S_data_LES_coarse[1, :, :, $n]

u_modelₙ = @lift u_data_model_coarse[1, :, :, $n]
v_modelₙ = @lift v_data_model_coarse[1, :, :, $n]
T_modelₙ = @lift T_data_model_coarse[1, :, :, $n]
S_modelₙ = @lift S_data_model_coarse[1, :, :, $n]

Δuₙ = @lift $u_LESₙ .- $u_modelₙ
Δvₙ = @lift $v_LESₙ .- $v_modelₙ
ΔTₙ = @lift $T_LESₙ .- $T_modelₙ
ΔSₙ = @lift $S_LESₙ .- $S_modelₙ

ulim = @lift (-maximum([maximum(abs, $u_LESₙ), 1e-16, maximum(abs, $u_modelₙ)]), 
               maximum([maximum(abs, $u_LESₙ), 1e-16, maximum(abs, $u_modelₙ)]))
vlim = @lift (-maximum([maximum(abs, $v_LESₙ), 1e-16, maximum(abs, $v_modelₙ)]),
               maximum([maximum(abs, $v_LESₙ), 1e-16, maximum(abs, $v_modelₙ)]))
Tlim = (minimum(T_data_LES_coarse[1, :, :, 1]), maximum(T_data_LES_coarse[1, :, :, 1]))
Slim = (minimum(S_data_LES_coarse[1, :, :, 1]), maximum(S_data_LES_coarse[1, :, :, 1]))

Δulim = @lift (-maximum([maximum(abs, $Δuₙ), 1e-16]), maximum([maximum(abs, $Δuₙ), 1e-16]))
Δvlim = @lift (-maximum([maximum(abs, $Δvₙ), 1e-16]), maximum([maximum(abs, $Δvₙ), 1e-16]))
ΔTlim = @lift (-maximum([maximum(abs, $ΔTₙ), 1e-16]), maximum([maximum(abs, $ΔTₙ), 1e-16]))
ΔSlim = @lift (-maximum([maximum(abs, $ΔSₙ), 1e-16]), maximum([maximum(abs, $ΔSₙ), 1e-16]))

hu = heatmap!(axu_LES, yC_model, zC_model, u_LESₙ, colormap = :RdBu_9, colorrange = ulim)
hv = heatmap!(axv_LES, yC_model, zC_model, v_LESₙ, colormap = :RdBu_9, colorrange = vlim)
hT = heatmap!(axT_LES, yC_model, zC_model, T_LESₙ, colorrange = Tlim)
hS = heatmap!(axS_LES, yC_model, zC_model, S_LESₙ, colorrange = Slim)

hu_model = heatmap!(axu_model, yC_model, zC_model, u_modelₙ, colormap = :RdBu_9, colorrange = ulim)
hv_model = heatmap!(axv_model, yC_model, zC_model, v_modelₙ, colormap = :RdBu_9, colorrange = vlim)
hT_model = heatmap!(axT_model, yC_model, zC_model, T_modelₙ, colorrange = Tlim)
hS_model = heatmap!(axS_model, yC_model, zC_model, S_modelₙ, colorrange = Slim)

hΔu = heatmap!(axΔu, yC_model, zC_model, Δuₙ, colormap = :RdBu_9, colorrange = Δulim)
hΔv = heatmap!(axΔv, yC_model, zC_model, Δvₙ, colormap = :RdBu_9, colorrange = Δvlim)
hΔT = heatmap!(axΔT, yC_model, zC_model, ΔTₙ, colormap = :RdBu_9, colorrange = ΔTlim)
hΔS = heatmap!(axΔS, yC_model, zC_model, ΔSₙ, colormap = :RdBu_9, colorrange = ΔSlim)

Colorbar(fig[1:2, 2], hu, label = "u (m/s)")
Colorbar(fig[1:2, 4], hv, label = "v (m/s)")
Colorbar(fig[1:2, 6], hT, label = "T (°C)")
Colorbar(fig[1:2, 8], hS, label = "S (psu)")

Colorbar(fig[3, 2], hΔu, label = "u (m/s)")
Colorbar(fig[3, 4], hΔv, label = "v (m/s)")
Colorbar(fig[3, 6], hΔT, label = "T (°C)")
Colorbar(fig[3, 8], hΔS, label = "S (psu)")

# display(fig)

CairoMakie.record(fig, "./LES_NN_2D_sin_cooling_heating_3e-4_23_days_comparison.mp4", 1:Nt, framerate=30) do nn
    @info nn
    n[] = nn
end


#%%