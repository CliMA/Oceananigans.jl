using Oceananigans
using Statistics
using JLD2

LES_FILE_DIR = "./NN_2D_channel_horizontal_convection_0.0003_LES.jld2"
MODEL_FILE_DIR = "./NN_closure_2D_channel.jld2"

u_data_LES = FieldTimeSeries(LES_FILE_DIR, "u", backend=OnDisk())
v_data_LES = FieldTimeSeries(LES_FILE_DIR, "v", backend=OnDisk())
T_data_LES = FieldTimeSeries(LES_FILE_DIR, "T", backend=OnDisk())
S_data_LES = FieldTimeSeries(LES_FILE_DIR, "S", backend=OnDisk())

u_data_model = FieldTimeSeries(MODEL_FILE_DIR, "u")
v_data_model = FieldTimeSeries(MODEL_FILE_DIR, "v")
T_data_model = FieldTimeSeries(MODEL_FILE_DIR, "T")
S_data_model = FieldTimeSeries(MODEL_FILE_DIR, "S")

end_time = 23 * 60^2 * 24

yC_LES = ynodes(T_data_LES.grid, Center())
yF_LES = ynodes(T_data_LES.grid, Face())

zC_LES = znodes(T_data_LES.grid, Center())
zF_LES = znodes(T_data_LES.grid, Face())

Nt_LES = findfirst(x -> x ≈ end_time, T_data_LES.times)

Δy_LES = T_data_LES.grid.Ly / T_data_LES.grid.Ny
Δz_LES = T_data_LES.grid.Lz / T_data_LES.grid.Nz

Ny_model = T_data_model.grid.Ny
Nz_model = T_data_model.grid.Nz

Δy_model = T_data_model.grid.Ly / Ny_model
Δz_model = T_data_model.grid.Lz / Nz_model

coarse_ratio_y = Int(Δy_model / Δy_LES)
coarse_ratio_z = Int(Δz_model / Δz_LES)

function coarsen_dataᵃᶜᵃ(data_LES, coarse_ratio_y, coarse_ratio_z, Ny_model, Nz_model, Nt_LES)
    Ny_LES = data_LES.grid.Ny
    Nz_LES = data_LES.grid.Nz
    data_LES_coarse = zeros(1, Ny_model, Nz_model, Nt_LES)
    LES_temp = zeros(Ny_LES, Nz_LES)
    
    for nt in 1:Nt_LES
        LES_temp .= interior(data_LES[nt], 1, :, :)
        Threads.@threads for j in axes(data_LES_coarse, 2)
            @info "nt = $nt, Processing j = $j"
            for k in axes(data_LES_coarse, 3)
                data_LES_coarse[1, j, k, nt] = mean(LES_temp[(j-1)*coarse_ratio_y+1:j*coarse_ratio_y, (k-1)*coarse_ratio_z+1:k*coarse_ratio_z])
            end
        end
    end

    return data_LES_coarse
end
    
T_data_LES_coarse = coarsen_dataᵃᶜᵃ(T_data_LES, coarse_ratio_y, coarse_ratio_z, Ny_model, Nz_model, Nt_LES)
S_data_LES_coarse = coarsen_dataᵃᶜᵃ(S_data_LES, coarse_ratio_y, coarse_ratio_z, Ny_model, Nz_model, Nt_LES)
u_data_LES_coarse = coarsen_dataᵃᶜᵃ(u_data_LES, coarse_ratio_y, coarse_ratio_z, Ny_model, Nz_model, Nt_LES)

function coarsen_dataᵃᶠᵃ(data_LES, coarse_ratio_y, coarse_ratio_z, Ny_model, Nz_model, Nt)
    Ny_LES = data_LES.grid.Ny
    Nz_LES = data_LES.grid.Nz

    dataᵃᶠᵃ_temp = zeros(Ny_LES+1, Nz_LES)
    dataᵃᶜᵃ_temp = zeros(Ny_LES, Nz_LES)
    
    data_coarse = zeros(1, Ny_model, Nz_model, Nt)
    
    for nt in 1:Nt
        @info "nt = $nt, Interpolating for LES data"
        dataᵃᶠᵃ_temp .= interior(data_LES[nt], 1, :, :)
        Threads.@threads for j in 1:Ny_LES
        # for j in 1:Ny_LES
            for k in 1:Nz_LES
                dataᵃᶜᵃ_temp[j, k] = mean(dataᵃᶠᵃ_temp[j:j+1, k])
            end
        end

        @info "nt = $nt, Coarsening data"
        Threads.@threads for j in axes(data_coarse, 2)
        # for j in axes(data_coarse, 2)
            for k in axes(data_coarse, 3)
                data_coarse[1, j, k, nt] = mean(dataᵃᶜᵃ_temp[(j-1)*coarse_ratio_y+1:j*coarse_ratio_y, (k-1)*coarse_ratio_z+1:k*coarse_ratio_z])
            end
        end
    end

    return data_coarse
end

v_data_LES_coarse = coarsen_dataᵃᶠᵃ(v_data_LES, coarse_ratio_y, coarse_ratio_z, Ny_model, Nz_model, Nt_LES)
v_data_model_coarse = coarsen_dataᵃᶠᵃ(v_data_model, 1, 1, Ny_model, Nz_model, Nt_LES)

u_data_model_coarse = interior(u_data_model)
T_data_model_coarse = interior(T_data_model)
S_data_model_coarse = interior(S_data_model)

jldopen("./LES_NDE_FC_Qb_absf_24simnew_2layer_128_relu_2Pr_coarsened.jld2", "w") do file
    file["u_LES"] = u_data_LES_coarse
    file["v_LES"] = v_data_LES_coarse
    file["T_LES"] = T_data_LES_coarse
    file["S_LES"] = S_data_LES_coarse
    file["u_NN_model"] = u_data_model_coarse
    file["v_NN_model"] = v_data_model_coarse
    file["T_NN_model"] = T_data_model_coarse
    file["S_NN_model"] = S_data_model_coarse
end

