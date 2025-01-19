using GLMakie
using Oceananigans
using ColorSchemes
using SeawaterPolynomials
using SeawaterPolynomials.TEOS10

NN_FILE_DIR  = "./Output/doublegyre_30Cwarmflushbottom10_relaxation_8days_NN_closure_BBLkappazonelast41_temp"
CATKE_FILE_DIR = "./Output/doublegyre_30Cwarmflushbottom10_relaxation_8days_baseclosure_trainFC24new_scalingtrain54new_2Pr_2step"

fieldname = "S"
ρ_NN_data_00 = FieldTimeSeries("$(NN_FILE_DIR)/instantaneous_fields_yz.jld2", fieldname, backend=OnDisk())
ρ_NN_data_10 = FieldTimeSeries("$(NN_FILE_DIR)/instantaneous_fields_yz_10.jld2", fieldname, backend=OnDisk())
ρ_NN_data_20 = FieldTimeSeries("$(NN_FILE_DIR)/instantaneous_fields_yz_20.jld2", fieldname, backend=OnDisk())
ρ_NN_data_30 = FieldTimeSeries("$(NN_FILE_DIR)/instantaneous_fields_yz_30.jld2", fieldname, backend=OnDisk())
ρ_NN_data_40 = FieldTimeSeries("$(NN_FILE_DIR)/instantaneous_fields_yz_40.jld2", fieldname, backend=OnDisk())
ρ_NN_data_50 = FieldTimeSeries("$(NN_FILE_DIR)/instantaneous_fields_yz_50.jld2", fieldname, backend=OnDisk())
ρ_NN_data_60 = FieldTimeSeries("$(NN_FILE_DIR)/instantaneous_fields_yz_60.jld2", fieldname, backend=OnDisk())
ρ_NN_data_70 = FieldTimeSeries("$(NN_FILE_DIR)/instantaneous_fields_yz_70.jld2", fieldname, backend=OnDisk())
ρ_NN_data_80 = FieldTimeSeries("$(NN_FILE_DIR)/instantaneous_fields_yz_80.jld2", fieldname, backend=OnDisk())
ρ_NN_data_90 = FieldTimeSeries("$(NN_FILE_DIR)/instantaneous_fields_yz_90.jld2", fieldname, backend=OnDisk())

ρ_CATKE_data_00 = FieldTimeSeries("$(CATKE_FILE_DIR)/instantaneous_fields_yz.jld2", fieldname, backend=OnDisk())
ρ_CATKE_data_10 = FieldTimeSeries("$(CATKE_FILE_DIR)/instantaneous_fields_yz_10.jld2", fieldname, backend=OnDisk())
ρ_CATKE_data_20 = FieldTimeSeries("$(CATKE_FILE_DIR)/instantaneous_fields_yz_20.jld2", fieldname, backend=OnDisk())
ρ_CATKE_data_30 = FieldTimeSeries("$(CATKE_FILE_DIR)/instantaneous_fields_yz_30.jld2", fieldname, backend=OnDisk())
ρ_CATKE_data_40 = FieldTimeSeries("$(CATKE_FILE_DIR)/instantaneous_fields_yz_40.jld2", fieldname, backend=OnDisk())
ρ_CATKE_data_50 = FieldTimeSeries("$(CATKE_FILE_DIR)/instantaneous_fields_yz_50.jld2", fieldname, backend=OnDisk())
ρ_CATKE_data_60 = FieldTimeSeries("$(CATKE_FILE_DIR)/instantaneous_fields_yz_60.jld2", fieldname, backend=OnDisk())
ρ_CATKE_data_70 = FieldTimeSeries("$(CATKE_FILE_DIR)/instantaneous_fields_yz_70.jld2", fieldname, backend=OnDisk())
ρ_CATKE_data_80 = FieldTimeSeries("$(CATKE_FILE_DIR)/instantaneous_fields_yz_80.jld2", fieldname, backend=OnDisk())
ρ_CATKE_data_90 = FieldTimeSeries("$(CATKE_FILE_DIR)/instantaneous_fields_yz_90.jld2", fieldname, backend=OnDisk())

# ρ_NN_data_00 = FieldTimeSeries("$(NN_FILE_DIR)/instantaneous_fields_yz.jld2", fieldname)
# ρ_NN_data_10 = FieldTimeSeries("$(NN_FILE_DIR)/instantaneous_fields_yz_10.jld2", fieldname)
# ρ_NN_data_20 = FieldTimeSeries("$(NN_FILE_DIR)/instantaneous_fields_yz_20.jld2", fieldname)
# ρ_NN_data_30 = FieldTimeSeries("$(NN_FILE_DIR)/instantaneous_fields_yz_30.jld2", fieldname)
# ρ_NN_data_40 = FieldTimeSeries("$(NN_FILE_DIR)/instantaneous_fields_yz_40.jld2", fieldname)
# ρ_NN_data_50 = FieldTimeSeries("$(NN_FILE_DIR)/instantaneous_fields_yz_50.jld2", fieldname)
# ρ_NN_data_60 = FieldTimeSeries("$(NN_FILE_DIR)/instantaneous_fields_yz_60.jld2", fieldname)
# ρ_NN_data_70 = FieldTimeSeries("$(NN_FILE_DIR)/instantaneous_fields_yz_70.jld2", fieldname)
# ρ_NN_data_80 = FieldTimeSeries("$(NN_FILE_DIR)/instantaneous_fields_yz_80.jld2", fieldname)
# ρ_NN_data_90 = FieldTimeSeries("$(NN_FILE_DIR)/instantaneous_fields_yz_90.jld2", fieldname)

# ρ_CATKE_data_00 = FieldTimeSeries("$(CATKE_FILE_DIR)/instantaneous_fields_yz.jld2", fieldname)
# ρ_CATKE_data_10 = FieldTimeSeries("$(CATKE_FILE_DIR)/instantaneous_fields_yz_10.jld2", fieldname)
# ρ_CATKE_data_20 = FieldTimeSeries("$(CATKE_FILE_DIR)/instantaneous_fields_yz_20.jld2", fieldname)
# ρ_CATKE_data_30 = FieldTimeSeries("$(CATKE_FILE_DIR)/instantaneous_fields_yz_30.jld2", fieldname)
# ρ_CATKE_data_40 = FieldTimeSeries("$(CATKE_FILE_DIR)/instantaneous_fields_yz_40.jld2", fieldname)
# ρ_CATKE_data_50 = FieldTimeSeries("$(CATKE_FILE_DIR)/instantaneous_fields_yz_50.jld2", fieldname)
# ρ_CATKE_data_60 = FieldTimeSeries("$(CATKE_FILE_DIR)/instantaneous_fields_yz_60.jld2", fieldname)
# ρ_CATKE_data_70 = FieldTimeSeries("$(CATKE_FILE_DIR)/instantaneous_fields_yz_70.jld2", fieldname)
# ρ_CATKE_data_80 = FieldTimeSeries("$(CATKE_FILE_DIR)/instantaneous_fields_yz_80.jld2", fieldname)
# ρ_CATKE_data_90 = FieldTimeSeries("$(CATKE_FILE_DIR)/instantaneous_fields_yz_90.jld2", fieldname)

first_index_data = FieldTimeSeries("$(NN_FILE_DIR)/instantaneous_fields_NN_active_diagnostics.jld2", "first_index")
last_index_data = FieldTimeSeries("$(NN_FILE_DIR)/instantaneous_fields_NN_active_diagnostics.jld2", "last_index")
Qb_data = FieldTimeSeries("$(NN_FILE_DIR)/instantaneous_fields_NN_active_diagnostics.jld2", "Qb")

Nx, Ny, Nz = ρ_NN_data_00.grid.Nx, ρ_NN_data_00.grid.Ny, ρ_NN_data_00.grid.Nz

xC = ρ_NN_data_00.grid.xᶜᵃᵃ[1:ρ_NN_data_00.grid.Nx]
yC = ρ_NN_data_00.grid.yᵃᶜᵃ[1:ρ_NN_data_00.grid.Ny]
zC = ρ_NN_data_00.grid.zᵃᵃᶜ[1:ρ_NN_data_00.grid.Nz]
zF = ρ_NN_data_00.grid.zᵃᵃᶠ[1:ρ_NN_data_00.grid.Nz+1]

Nt = length(ρ_NN_data_90)
times = ρ_NN_data_00.times / 24 / 60^2 / 365
timeframes = 1:Nt

function find_min(a...)
  return minimum(minimum.([a...]))
end

function find_max(a...)
  return maximum(maximum.([a...]))
end

#%%
fig = Figure(size=(2400, 2400))

axNN_00 = GLMakie.Axis(fig[1, 1], xlabel="y (m)", ylabel="z (m)", title="NN, x = $(ρ_NN_data_00.grid.xᶜᵃᵃ[ρ_NN_data_00.indices[1][1]] / 1000) km")
axNN_10 = GLMakie.Axis(fig[1, 3], xlabel="y (m)", ylabel="z (m)", title="NN, x = $(ρ_NN_data_10.grid.xᶜᵃᵃ[ρ_NN_data_10.indices[1][1]] / 1000) km")
axNN_20 = GLMakie.Axis(fig[2, 1], xlabel="y (m)", ylabel="z (m)", title="NN, x = $(ρ_NN_data_20.grid.xᶜᵃᵃ[ρ_NN_data_20.indices[1][1]] / 1000) km")
axNN_30 = GLMakie.Axis(fig[2, 3], xlabel="y (m)", ylabel="z (m)", title="NN, x = $(ρ_NN_data_30.grid.xᶜᵃᵃ[ρ_NN_data_30.indices[1][1]] / 1000) km")
axNN_40 = GLMakie.Axis(fig[3, 1], xlabel="y (m)", ylabel="z (m)", title="NN, x = $(ρ_NN_data_40.grid.xᶜᵃᵃ[ρ_NN_data_40.indices[1][1]] / 1000) km")
axNN_50 = GLMakie.Axis(fig[3, 3], xlabel="y (m)", ylabel="z (m)", title="NN, x = $(ρ_NN_data_50.grid.xᶜᵃᵃ[ρ_NN_data_50.indices[1][1]] / 1000) km")
axNN_60 = GLMakie.Axis(fig[4, 1], xlabel="y (m)", ylabel="z (m)", title="NN, x = $(ρ_NN_data_60.grid.xᶜᵃᵃ[ρ_NN_data_60.indices[1][1]] / 1000) km")
axNN_70 = GLMakie.Axis(fig[4, 3], xlabel="y (m)", ylabel="z (m)", title="NN, x = $(ρ_NN_data_70.grid.xᶜᵃᵃ[ρ_NN_data_70.indices[1][1]] / 1000) km")
axNN_80 = GLMakie.Axis(fig[5, 1], xlabel="y (m)", ylabel="z (m)", title="NN, x = $(ρ_NN_data_80.grid.xᶜᵃᵃ[ρ_NN_data_80.indices[1][1]] / 1000) km")
axNN_90 = GLMakie.Axis(fig[5, 3], xlabel="y (m)", ylabel="z (m)", title="NN, x = $(ρ_NN_data_90.grid.xᶜᵃᵃ[ρ_NN_data_90.indices[1][1]] / 1000) km")

axCATKE_00 = GLMakie.Axis(fig[1, 2], xlabel="y (m)", ylabel="z (m)", title="Base Closure, x = $(ρ_CATKE_data_00.grid.xᶜᵃᵃ[ρ_CATKE_data_00.indices[1][1]] / 1000) km")
axCATKE_10 = GLMakie.Axis(fig[1, 4], xlabel="y (m)", ylabel="z (m)", title="Base Closure, x = $(ρ_CATKE_data_10.grid.xᶜᵃᵃ[ρ_CATKE_data_10.indices[1][1]] / 1000) km")
axCATKE_20 = GLMakie.Axis(fig[2, 2], xlabel="y (m)", ylabel="z (m)", title="Base Closure, x = $(ρ_CATKE_data_20.grid.xᶜᵃᵃ[ρ_CATKE_data_20.indices[1][1]] / 1000) km")
axCATKE_30 = GLMakie.Axis(fig[2, 4], xlabel="y (m)", ylabel="z (m)", title="Base Closure, x = $(ρ_CATKE_data_30.grid.xᶜᵃᵃ[ρ_CATKE_data_30.indices[1][1]] / 1000) km")
axCATKE_40 = GLMakie.Axis(fig[3, 2], xlabel="y (m)", ylabel="z (m)", title="Base Closure, x = $(ρ_CATKE_data_40.grid.xᶜᵃᵃ[ρ_CATKE_data_40.indices[1][1]] / 1000) km")
axCATKE_50 = GLMakie.Axis(fig[3, 4], xlabel="y (m)", ylabel="z (m)", title="Base Closure, x = $(ρ_CATKE_data_50.grid.xᶜᵃᵃ[ρ_CATKE_data_50.indices[1][1]] / 1000) km")
axCATKE_60 = GLMakie.Axis(fig[4, 2], xlabel="y (m)", ylabel="z (m)", title="Base Closure, x = $(ρ_CATKE_data_60.grid.xᶜᵃᵃ[ρ_CATKE_data_60.indices[1][1]] / 1000) km")
axCATKE_70 = GLMakie.Axis(fig[4, 4], xlabel="y (m)", ylabel="z (m)", title="Base Closure, x = $(ρ_CATKE_data_70.grid.xᶜᵃᵃ[ρ_CATKE_data_70.indices[1][1]] / 1000) km")
axCATKE_80 = GLMakie.Axis(fig[5, 2], xlabel="y (m)", ylabel="z (m)", title="Base Closure, x = $(ρ_CATKE_data_80.grid.xᶜᵃᵃ[ρ_CATKE_data_80.indices[1][1]] / 1000) km")
axCATKE_90 = GLMakie.Axis(fig[5, 4], xlabel="y (m)", ylabel="z (m)", title="Base Closure, x = $(ρ_CATKE_data_90.grid.xᶜᵃᵃ[ρ_CATKE_data_90.indices[1][1]] / 1000) km")

n = Observable(1096)

z_indices = 1:200

ρlim = (find_min(interior(ρ_NN_data_00[timeframes[1]], :, :, z_indices), interior(ρ_NN_data_00[timeframes[end]], :, :, z_indices), interior(ρ_CATKE_data_00[timeframes[1]], :, :, z_indices), interior(ρ_CATKE_data_00[timeframes[end]], :, :, z_indices)),
         find_max(interior(ρ_NN_data_00[timeframes[1]], :, :, z_indices), interior(ρ_NN_data_00[timeframes[end]], :, :, z_indices), interior(ρ_CATKE_data_00[timeframes[1]], :, :, z_indices), interior(ρ_CATKE_data_00[timeframes[end]], :, :, z_indices)))

NN_00ₙ = @lift interior(ρ_NN_data_00[$n], 1, :, z_indices)
NN_10ₙ = @lift interior(ρ_NN_data_10[$n], 1, :, z_indices)
NN_20ₙ = @lift interior(ρ_NN_data_20[$n], 1, :, z_indices)
NN_30ₙ = @lift interior(ρ_NN_data_30[$n], 1, :, z_indices)
NN_40ₙ = @lift interior(ρ_NN_data_40[$n], 1, :, z_indices)
NN_50ₙ = @lift interior(ρ_NN_data_50[$n], 1, :, z_indices)
NN_60ₙ = @lift interior(ρ_NN_data_60[$n], 1, :, z_indices)
NN_70ₙ = @lift interior(ρ_NN_data_70[$n], 1, :, z_indices)
NN_80ₙ = @lift interior(ρ_NN_data_80[$n], 1, :, z_indices)
NN_90ₙ = @lift interior(ρ_NN_data_90[$n], 1, :, z_indices)

CATKE_00ₙ = @lift interior(ρ_CATKE_data_00[$n], 1, :, z_indices)
CATKE_10ₙ = @lift interior(ρ_CATKE_data_10[$n], 1, :, z_indices)
CATKE_20ₙ = @lift interior(ρ_CATKE_data_20[$n], 1, :, z_indices)
CATKE_30ₙ = @lift interior(ρ_CATKE_data_30[$n], 1, :, z_indices)
CATKE_40ₙ = @lift interior(ρ_CATKE_data_40[$n], 1, :, z_indices)
CATKE_50ₙ = @lift interior(ρ_CATKE_data_50[$n], 1, :, z_indices)
CATKE_60ₙ = @lift interior(ρ_CATKE_data_60[$n], 1, :, z_indices)
CATKE_70ₙ = @lift interior(ρ_CATKE_data_70[$n], 1, :, z_indices)
CATKE_80ₙ = @lift interior(ρ_CATKE_data_80[$n], 1, :, z_indices)
CATKE_90ₙ = @lift interior(ρ_CATKE_data_90[$n], 1, :, z_indices)

zs_first_index_00ₙ = @lift zF[Int.(vec(interior(first_index_data[$n], ρ_NN_data_00.indices[1][1], :, :)))]
zs_first_index_10ₙ = @lift zF[Int.(vec(interior(first_index_data[$n], ρ_NN_data_10.indices[1][1], :, :)))]
zs_first_index_20ₙ = @lift zF[Int.(vec(interior(first_index_data[$n], ρ_NN_data_20.indices[1][1], :, :)))]
zs_first_index_30ₙ = @lift zF[Int.(vec(interior(first_index_data[$n], ρ_NN_data_30.indices[1][1], :, :)))]
zs_first_index_40ₙ = @lift zF[Int.(vec(interior(first_index_data[$n], ρ_NN_data_40.indices[1][1], :, :)))]
zs_first_index_50ₙ = @lift zF[Int.(vec(interior(first_index_data[$n], ρ_NN_data_50.indices[1][1], :, :)))]
zs_first_index_60ₙ = @lift zF[Int.(vec(interior(first_index_data[$n], ρ_NN_data_60.indices[1][1], :, :)))]
zs_first_index_70ₙ = @lift zF[Int.(vec(interior(first_index_data[$n], ρ_NN_data_70.indices[1][1], :, :)))]
zs_first_index_80ₙ = @lift zF[Int.(vec(interior(first_index_data[$n], ρ_NN_data_80.indices[1][1], :, :)))]
zs_first_index_90ₙ = @lift zF[Int.(vec(interior(first_index_data[$n], ρ_NN_data_90.indices[1][1], :, :)))]

zs_last_index_00ₙ = @lift zF[Int.(vec(interior(last_index_data[$n], ρ_NN_data_00.indices[1][1], :, :)))]
zs_last_index_10ₙ = @lift zF[Int.(vec(interior(last_index_data[$n], ρ_NN_data_10.indices[1][1], :, :)))]
zs_last_index_20ₙ = @lift zF[Int.(vec(interior(last_index_data[$n], ρ_NN_data_20.indices[1][1], :, :)))]
zs_last_index_30ₙ = @lift zF[Int.(vec(interior(last_index_data[$n], ρ_NN_data_30.indices[1][1], :, :)))]
zs_last_index_40ₙ = @lift zF[Int.(vec(interior(last_index_data[$n], ρ_NN_data_40.indices[1][1], :, :)))]
zs_last_index_50ₙ = @lift zF[Int.(vec(interior(last_index_data[$n], ρ_NN_data_50.indices[1][1], :, :)))]
zs_last_index_60ₙ = @lift zF[Int.(vec(interior(last_index_data[$n], ρ_NN_data_60.indices[1][1], :, :)))]
zs_last_index_70ₙ = @lift zF[Int.(vec(interior(last_index_data[$n], ρ_NN_data_70.indices[1][1], :, :)))]
zs_last_index_80ₙ = @lift zF[Int.(vec(interior(last_index_data[$n], ρ_NN_data_80.indices[1][1], :, :)))]
zs_last_index_90ₙ = @lift zF[Int.(vec(interior(last_index_data[$n], ρ_NN_data_90.indices[1][1], :, :)))]

ys_convection_00ₙ = @lift yC[interior(Qb_data[$n], ρ_NN_data_00.indices[1][1], :, 1) .> 0]
ys_convection_10ₙ = @lift yC[interior(Qb_data[$n], ρ_NN_data_10.indices[1][1], :, 1) .> 0]
ys_convection_20ₙ = @lift yC[interior(Qb_data[$n], ρ_NN_data_20.indices[1][1], :, 1) .> 0]
ys_convection_30ₙ = @lift yC[interior(Qb_data[$n], ρ_NN_data_30.indices[1][1], :, 1) .> 0]
ys_convection_40ₙ = @lift yC[interior(Qb_data[$n], ρ_NN_data_40.indices[1][1], :, 1) .> 0]
ys_convection_50ₙ = @lift yC[interior(Qb_data[$n], ρ_NN_data_50.indices[1][1], :, 1) .> 0]
ys_convection_60ₙ = @lift yC[interior(Qb_data[$n], ρ_NN_data_60.indices[1][1], :, 1) .> 0]
ys_convection_70ₙ = @lift yC[interior(Qb_data[$n], ρ_NN_data_70.indices[1][1], :, 1) .> 0]
ys_convection_80ₙ = @lift yC[interior(Qb_data[$n], ρ_NN_data_80.indices[1][1], :, 1) .> 0]
ys_convection_90ₙ = @lift yC[interior(Qb_data[$n], ρ_NN_data_90.indices[1][1], :, 1) .> 0]

zs_convection_00ₙ = @lift fill(zC[1], length($ys_convection_00ₙ))
zs_convection_10ₙ = @lift fill(zC[1], length($ys_convection_10ₙ))
zs_convection_20ₙ = @lift fill(zC[1], length($ys_convection_20ₙ))
zs_convection_30ₙ = @lift fill(zC[1], length($ys_convection_30ₙ))
zs_convection_40ₙ = @lift fill(zC[1], length($ys_convection_40ₙ))
zs_convection_50ₙ = @lift fill(zC[1], length($ys_convection_50ₙ))
zs_convection_60ₙ = @lift fill(zC[1], length($ys_convection_60ₙ))
zs_convection_70ₙ = @lift fill(zC[1], length($ys_convection_70ₙ))
zs_convection_80ₙ = @lift fill(zC[1], length($ys_convection_80ₙ))
zs_convection_90ₙ = @lift fill(zC[1], length($ys_convection_90ₙ))

# colorscheme = Reverse(colorschemes[:jet])
colorscheme = colorschemes[:jet]

NN_00_surface = heatmap!(axNN_00, yC, zC[z_indices], NN_00ₙ, colormap=colorscheme, colorrange=ρlim)
NN_10_surface = heatmap!(axNN_10, yC, zC[z_indices], NN_10ₙ, colormap=colorscheme, colorrange=ρlim)
NN_20_surface = heatmap!(axNN_20, yC, zC[z_indices], NN_20ₙ, colormap=colorscheme, colorrange=ρlim)
NN_30_surface = heatmap!(axNN_30, yC, zC[z_indices], NN_30ₙ, colormap=colorscheme, colorrange=ρlim)
NN_40_surface = heatmap!(axNN_40, yC, zC[z_indices], NN_40ₙ, colormap=colorscheme, colorrange=ρlim)
NN_50_surface = heatmap!(axNN_50, yC, zC[z_indices], NN_50ₙ, colormap=colorscheme, colorrange=ρlim)
NN_60_surface = heatmap!(axNN_60, yC, zC[z_indices], NN_60ₙ, colormap=colorscheme, colorrange=ρlim)
NN_70_surface = heatmap!(axNN_70, yC, zC[z_indices], NN_70ₙ, colormap=colorscheme, colorrange=ρlim)
NN_80_surface = heatmap!(axNN_80, yC, zC[z_indices], NN_80ₙ, colormap=colorscheme, colorrange=ρlim)
NN_90_surface = heatmap!(axNN_90, yC, zC[z_indices], NN_90ₙ, colormap=colorscheme, colorrange=ρlim)

CATKE_00_surface = heatmap!(axCATKE_00, yC, zC[z_indices], CATKE_00ₙ, colormap=colorscheme, colorrange=ρlim)
CATKE_10_surface = heatmap!(axCATKE_10, yC, zC[z_indices], CATKE_10ₙ, colormap=colorscheme, colorrange=ρlim)
CATKE_20_surface = heatmap!(axCATKE_20, yC, zC[z_indices], CATKE_20ₙ, colormap=colorscheme, colorrange=ρlim)
CATKE_30_surface = heatmap!(axCATKE_30, yC, zC[z_indices], CATKE_30ₙ, colormap=colorscheme, colorrange=ρlim)
CATKE_40_surface = heatmap!(axCATKE_40, yC, zC[z_indices], CATKE_40ₙ, colormap=colorscheme, colorrange=ρlim)
CATKE_50_surface = heatmap!(axCATKE_50, yC, zC[z_indices], CATKE_50ₙ, colormap=colorscheme, colorrange=ρlim)
CATKE_60_surface = heatmap!(axCATKE_60, yC, zC[z_indices], CATKE_60ₙ, colormap=colorscheme, colorrange=ρlim)
CATKE_70_surface = heatmap!(axCATKE_70, yC, zC[z_indices], CATKE_70ₙ, colormap=colorscheme, colorrange=ρlim)
CATKE_80_surface = heatmap!(axCATKE_80, yC, zC[z_indices], CATKE_80ₙ, colormap=colorscheme, colorrange=ρlim)
CATKE_90_surface = heatmap!(axCATKE_90, yC, zC[z_indices], CATKE_90ₙ, colormap=colorscheme, colorrange=ρlim)

# contourlevels = range(ρlim[1], ρlim[2], length=10)

# NN_00_surface = contourf!(axNN_00, yC, zC, NN_00ₙ, colormap=colorscheme, levels=contourlevels)
# NN_10_surface = contourf!(axNN_10, yC, zC, NN_10ₙ, colormap=colorscheme, levels=contourlevels)
# NN_20_surface = contourf!(axNN_20, yC, zC, NN_20ₙ, colormap=colorscheme, levels=contourlevels)
# NN_30_surface = contourf!(axNN_30, yC, zC, NN_30ₙ, colormap=colorscheme, levels=contourlevels)
# NN_40_surface = contourf!(axNN_40, yC, zC, NN_40ₙ, colormap=colorscheme, levels=contourlevels)
# NN_50_surface = contourf!(axNN_50, yC, zC, NN_50ₙ, colormap=colorscheme, levels=contourlevels)
# NN_60_surface = contourf!(axNN_60, yC, zC, NN_60ₙ, colormap=colorscheme, levels=contourlevels)
# NN_70_surface = contourf!(axNN_70, yC, zC, NN_70ₙ, colormap=colorscheme, levels=contourlevels)
# NN_80_surface = contourf!(axNN_80, yC, zC, NN_80ₙ, colormap=colorscheme, levels=contourlevels)
# NN_90_surface = contourf!(axNN_90, yC, zC, NN_90ₙ, colormap=colorscheme, levels=contourlevels)

# CATKE_00_surface = contourf!(axCATKE_00, yC, zC, CATKE_00ₙ, colormap=colorscheme, levels=contourlevels)
# CATKE_10_surface = contourf!(axCATKE_10, yC, zC, CATKE_10ₙ, colormap=colorscheme, levels=contourlevels)
# CATKE_20_surface = contourf!(axCATKE_20, yC, zC, CATKE_20ₙ, colormap=colorscheme, levels=contourlevels)
# CATKE_30_surface = contourf!(axCATKE_30, yC, zC, CATKE_30ₙ, colormap=colorscheme, levels=contourlevels)
# CATKE_40_surface = contourf!(axCATKE_40, yC, zC, CATKE_40ₙ, colormap=colorscheme, levels=contourlevels)
# CATKE_50_surface = contourf!(axCATKE_50, yC, zC, CATKE_50ₙ, colormap=colorscheme, levels=contourlevels)
# CATKE_60_surface = contourf!(axCATKE_60, yC, zC, CATKE_60ₙ, colormap=colorscheme, levels=contourlevels)
# CATKE_70_surface = contourf!(axCATKE_70, yC, zC, CATKE_70ₙ, colormap=colorscheme, levels=contourlevels)
# CATKE_80_surface = contourf!(axCATKE_80, yC, zC, CATKE_80ₙ, colormap=colorscheme, levels=contourlevels)
# CATKE_90_surface = contourf!(axCATKE_90, yC, zC, CATKE_90ₙ, colormap=colorscheme, levels=contourlevels)

Colorbar(fig[1:5, 5], NN_00_surface)

# lines!(axNN_00, yC, zs_first_index_00ₙ, color=:black)
# lines!(axNN_10, yC, zs_first_index_10ₙ, color=:black)
# lines!(axNN_20, yC, zs_first_index_20ₙ, color=:black)
# lines!(axNN_30, yC, zs_first_index_30ₙ, color=:black)
# lines!(axNN_40, yC, zs_first_index_40ₙ, color=:black)
# lines!(axNN_50, yC, zs_first_index_50ₙ, color=:black)
# lines!(axNN_60, yC, zs_first_index_60ₙ, color=:black)
# lines!(axNN_70, yC, zs_first_index_70ₙ, color=:black)
# lines!(axNN_80, yC, zs_first_index_80ₙ, color=:black)
# lines!(axNN_90, yC, zs_first_index_90ₙ, color=:black)

# lines!(axNN_00, yC, zs_last_index_00ₙ, color=:black)
# lines!(axNN_10, yC, zs_last_index_10ₙ, color=:black)
# lines!(axNN_20, yC, zs_last_index_20ₙ, color=:black)
# lines!(axNN_30, yC, zs_last_index_30ₙ, color=:black)
# lines!(axNN_40, yC, zs_last_index_40ₙ, color=:black)
# lines!(axNN_50, yC, zs_last_index_50ₙ, color=:black)
# lines!(axNN_60, yC, zs_last_index_60ₙ, color=:black)
# lines!(axNN_70, yC, zs_last_index_70ₙ, color=:black)
# lines!(axNN_80, yC, zs_last_index_80ₙ, color=:black)
# lines!(axNN_90, yC, zs_last_index_90ₙ, color=:black)

scatter!(axNN_00, ys_convection_00ₙ, zs_convection_00ₙ, color=:red, markersize=10)
scatter!(axNN_10, ys_convection_10ₙ, zs_convection_10ₙ, color=:red, markersize=10)
scatter!(axNN_20, ys_convection_20ₙ, zs_convection_20ₙ, color=:red, markersize=10)
scatter!(axNN_30, ys_convection_30ₙ, zs_convection_30ₙ, color=:red, markersize=10)
scatter!(axNN_40, ys_convection_40ₙ, zs_convection_40ₙ, color=:red, markersize=10)
scatter!(axNN_50, ys_convection_50ₙ, zs_convection_50ₙ, color=:red, markersize=10)
scatter!(axNN_60, ys_convection_60ₙ, zs_convection_60ₙ, color=:red, markersize=10)
scatter!(axNN_70, ys_convection_70ₙ, zs_convection_70ₙ, color=:red, markersize=10)
scatter!(axNN_80, ys_convection_80ₙ, zs_convection_80ₙ, color=:red, markersize=10)
scatter!(axNN_90, ys_convection_90ₙ, zs_convection_90ₙ, color=:red, markersize=10)

xlims!(axNN_00, minimum(yC), maximum(yC))
xlims!(axNN_10, minimum(yC), maximum(yC))
xlims!(axNN_20, minimum(yC), maximum(yC))
xlims!(axNN_30, minimum(yC), maximum(yC))
xlims!(axNN_40, minimum(yC), maximum(yC))
xlims!(axNN_50, minimum(yC), maximum(yC))
xlims!(axNN_60, minimum(yC), maximum(yC))
xlims!(axNN_70, minimum(yC), maximum(yC))
xlims!(axNN_80, minimum(yC), maximum(yC))
xlims!(axNN_90, minimum(yC), maximum(yC))

ylims!(axNN_00, minimum(zC[z_indices]), maximum(zC[z_indices]))
ylims!(axNN_10, minimum(zC[z_indices]), maximum(zC[z_indices]))
ylims!(axNN_20, minimum(zC[z_indices]), maximum(zC[z_indices]))
ylims!(axNN_30, minimum(zC[z_indices]), maximum(zC[z_indices]))
ylims!(axNN_40, minimum(zC[z_indices]), maximum(zC[z_indices]))
ylims!(axNN_50, minimum(zC[z_indices]), maximum(zC[z_indices]))
ylims!(axNN_60, minimum(zC[z_indices]), maximum(zC[z_indices]))
ylims!(axNN_70, minimum(zC[z_indices]), maximum(zC[z_indices]))
ylims!(axNN_80, minimum(zC[z_indices]), maximum(zC[z_indices]))
ylims!(axNN_90, minimum(zC[z_indices]), maximum(zC[z_indices]))

xlims!(axCATKE_00, minimum(yC), maximum(yC))
xlims!(axCATKE_10, minimum(yC), maximum(yC))
xlims!(axCATKE_20, minimum(yC), maximum(yC))
xlims!(axCATKE_30, minimum(yC), maximum(yC))
xlims!(axCATKE_40, minimum(yC), maximum(yC))
xlims!(axCATKE_50, minimum(yC), maximum(yC))
xlims!(axCATKE_60, minimum(yC), maximum(yC))
xlims!(axCATKE_70, minimum(yC), maximum(yC))
xlims!(axCATKE_80, minimum(yC), maximum(yC))
xlims!(axCATKE_90, minimum(yC), maximum(yC))

ylims!(axCATKE_00, minimum(zC[z_indices]), maximum(zC[z_indices]))
ylims!(axCATKE_10, minimum(zC[z_indices]), maximum(zC[z_indices]))
ylims!(axCATKE_20, minimum(zC[z_indices]), maximum(zC[z_indices]))
ylims!(axCATKE_30, minimum(zC[z_indices]), maximum(zC[z_indices]))
ylims!(axCATKE_40, minimum(zC[z_indices]), maximum(zC[z_indices]))
ylims!(axCATKE_50, minimum(zC[z_indices]), maximum(zC[z_indices]))
ylims!(axCATKE_60, minimum(zC[z_indices]), maximum(zC[z_indices]))
ylims!(axCATKE_70, minimum(zC[z_indices]), maximum(zC[z_indices]))
ylims!(axCATKE_80, minimum(zC[z_indices]), maximum(zC[z_indices]))
ylims!(axCATKE_90, minimum(zC[z_indices]), maximum(zC[z_indices]))

# title_str = @lift "Temperature (°C), Time = $(round(times[$n], digits=2)) years"
title_str = @lift "Salinity (psu), Time = $(round(times[$n], digits=2)) years"
# title_str = @lift "Potential Density (kg m⁻³), Time = $(round(times[$n], digits=2)) years"
Label(fig[0, :], text=title_str, tellwidth=false, font=:bold)

trim!(fig.layout)

display(fig)
GLMakie.record(fig, "./Output/doublegyre_relaxation_8days_30Cwarmflush10bottom_NNclosure_BBLkappazonelast41_baseclosure_S_BBLlines_yzslices.mp4", 1:Nt, framerate=15, px_per_unit=2) do nn
  @info "Recording frame $nn"
  n[] = nn
end
#%%