using CairoMakie, JLD2, Statistics, HDF5, Oceananigans, ProgressBars
using KernelAbstractions
using Oceananigans.Utils
using Oceananigans.Architectures: device
using KernelAbstractions: @kernel, @index
using Oceananigans.Operators
using Oceananigans.Architectures: architecture

include("coarse_graining_utils.jl")

plot_data = true 
save_data = true
plot_stream_function = true
data_directory = "/orcd/data/raffaele/001/sandre/OceananigansData/"
analysis_directory = "/orcd/data/raffaele/001/sandre/DoubleGyreAnalysisData/"
figure_directory = "oceananigans_figure/"
figure_directory = "quick_check_figure/"

casevar = 5

si = 1000 #starting index
levels = 1:1
kmax = 2
NN = 3

@info "loading data"
jlfile = jldopen(data_directory * "baroclinic_double_gyre_free_surface_$casevar.jld2", "r")
ηkeys =  keys(jlfile["timeseries"]["η"])[2:end]

η = zeros(size(jlfile["timeseries"]["η"]["0"])[1:2]..., length(ηkeys))

for (i, ηkey) in ProgressBar(enumerate(ηkeys))
    η[:, :, i] .= jlfile["timeseries"]["η"][ηkey]
end
close(jlfile)

@info "loading data 3D"
@info "loading v"
v = FieldTimeSeries(data_directory  * "baroclinic_double_gyre_$casevar.jld2", "v"; backend = InMemory(225))
@info "loading w"
w = FieldTimeSeries(data_directory  * "baroclinic_double_gyre_$casevar.jld2", "w"; backend = InMemory(225))

zs = v.grid.zᵃᵃᶜ[1:15]
levels = 1:15
timekeys = 1126:length(ηkeys)

@info "saving v"
averaged_field_v = zeros(256, 15, length(timekeys));
for i in ProgressBar(eachindex(timekeys))
    vfield = interior(v[timekeys[i]])
    vfield = (vfield[1:end, 1:end-1, :] .+ vfield[1:end, 2:end, :])/2
    averaged_field_v[:, :, i] .= mean(vfield, dims = 1)[1, :, :]
end

hfile = h5open(analysis_directory * "moc_$casevar.hdf5", "w")
hfile["v"] = averaged_field_v
close(hfile)

@info "saving w"
averaged_field_w = zeros(256, 15, length(timekeys));
for i in ProgressBar(eachindex(timekeys))
    wfield = interior(w[timekeys[i]])
    wfield = (wfield[1:end, 1:end, 1:end-1] .+ wfield[1:end, 1:end, 2:end])/2
    averaged_field_w[:, :, i] .= mean(wfield, dims = 1)[1, :, :]
end

hfile = h5open(analysis_directory * "moc_$casevar.hdf5", "r+")
hfile["w"] = averaged_field_w
close(hfile)

hfile = h5open(analysis_directory * "moc_$casevar.hdf5", "r+")
hfile["z"] = zs[levels]
hfile["time"] = collect(timekeys) 
close(hfile)

dz = reshape(v.grid.Δzᵃᵃᶜ[1:15], (1, 15, 1))
dy = reshape(v.grid.Δxᶜᶜᵃ[1:256], (256, 1, 1))
hfile = h5open(analysis_directory * "moc_$casevar.hdf5", "r+")
hfile["dz"] = dz
hfile["dy"] = dy

streamfunction_from_w = cumsum(averaged_field_w .* dy, dims = 1)
stream_function_from_w = streamfunction_from_w .- mean(streamfunction_from_w)
streamfunction_from_v = -cumsum(averaged_field_v .* dz, dims = 2)
stream_function_from_v = streamfunction_from_v .- mean(streamfunction_from_v)


hfile["streamfunction_from_w"] = streamfunction_from_w
hfile["streamfunction_from_v"] = streamfunction_from_v

close(hfile)

tmp1 = interior(v[end])
tmp2 = interior(w[end])
lats = range(15, 75, length = 256)
fig = Figure() 
ax = Axis(fig[1, 1])
cr = extrema( mean(streamfunction_from_w[:, :, 1:2000], dims = 3)[:, :, 1])
msf =  mean(streamfunction_from_w[:, :, 1:2000], dims = 3)[:, :, 1]
heatmap!(ax, lats, zs, msf, colormap = :viridis, colorrange = cr)
ax = Axis(fig[1, 2])
heatmap!(ax, lats, zs, streamfunction_from_w[:, :, 1], colormap = :viridis, colorrange = cr)
ax = Axis(fig[2, 1])
heatmap!(ax, lats, zs, streamfunction_from_w[:, :, 1000], colormap = :viridis, colorrange = cr)
ax = Axis(fig[2, 2])
heatmap!(ax, lats, zs, streamfunction_from_w[:, :, 2000], colormap = :viridis, colorrange = cr)

offset = 2
ax = Axis(fig[1+offset, 1])
msf =  mean(streamfunction_from_v[:, :, 1:2000], dims = 3)[:, :, 1]
heatmap!(ax, lats, zs, msf, colormap = :viridis, colorrange = cr)
ax = Axis(fig[1+offset, 2])
heatmap!(ax, lats, zs, streamfunction_from_v[:, :, 1], colormap = :viridis, colorrange = cr)
ax = Axis(fig[2+offset, 1])
heatmap!(ax, lats, zs, streamfunction_from_v[:, :, 1000], colormap = :viridis, colorrange = cr)
ax = Axis(fig[2+offset, 2])
heatmap!(ax, lats, zs, streamfunction_from_v[:, :, 2000], colormap = :viridis, colorrange = cr)
save("moc_$(casevar).png", fig)


fig = Figure() 
ax = Axis(fig[1, 1])
cr = extrema( mean(streamfunction_from_w[:, :, 1:2000], dims = 3)[:, :, 1])
msf =  mean(streamfunction_from_w[:, :, 1:2000], dims = 3)[:, :, 1]
contour!(ax, lats, zs, msf, colormap = :viridis, colorrange = cr)
ax = Axis(fig[1, 2])
contour!(ax, lats, zs, streamfunction_from_w[:, :, 1], colormap = :viridis, colorrange = cr)
ax = Axis(fig[2, 1])
contour!(ax, lats, zs, streamfunction_from_w[:, :, 1000], colormap = :viridis, colorrange = cr)
ax = Axis(fig[2, 2])
contour!(ax, lats, zs, streamfunction_from_w[:, :, 2000], colormap = :viridis, colorrange = cr)

offset = 2
ax = Axis(fig[1+offset, 1])
msf =  mean(streamfunction_from_v[:, :, 1:2000], dims = 3)[:, :, 1]
contour!(ax, lats, zs, msf, colormap = :viridis, colorrange = cr)
ax = Axis(fig[1+offset, 2])
contour!(ax, lats, zs, streamfunction_from_v[:, :, 1], colormap = :viridis, colorrange = cr)
ax = Axis(fig[2+offset, 1])
contour!(ax, lats, zs, streamfunction_from_v[:, :, 1000], colormap = :viridis, colorrange = cr)
ax = Axis(fig[2+offset, 2])
contour!(ax, lats, zs, streamfunction_from_v[:, :, 2000], colormap = :viridis, colorrange = cr)
save("moc_contour_$(casevar).png", fig)


##
fig = Figure() 
L1 = mean(abs.(η), dims = (1, 2))
Linf =  [maximum(η[:, :, i]) for i in eachindex(ηkeys)]

ax = Axis(fig[1, 1])
lines!(ax, 1:length(L1), L1[:])
ax = Axis(fig[1, 2])
lines!(ax, 1:length(Linf), Linf[:])
save("eta_$(casevar).png", fig)