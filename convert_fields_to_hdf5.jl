using CairoMakie, JLD2, Statistics, HDF5
data_directory = "/nobackup1/sandre/OceananigansData/"
figure_directory = "oceananigans_figure/"

jlfile = jldopen(data_directory * "baroclinic_double_gyre_free_surface.jld2", "r")
jlfile2 = jldopen(data_directory * "baroclinic_double_gyre.jld2", "r")
ηkeys =  keys(jlfile["timeseries"]["η"])[2:end]

η = zeros(size(jlfile["timeseries"]["η"]["0"])[1:2]..., length(ηkeys))
M, N, L = size(jlfile2["timeseries"]["b"]["0"])
u = zeros(M, N, L, length(ηkeys))
v = zeros(M, N, L, length(ηkeys))
w = zeros(M, N, L, length(ηkeys))
b = zeros(M, N, L, length(ηkeys))
for (i, ηkey) in enumerate(ηkeys)
    η[:, :, i] .= jlfile["timeseries"]["η"][ηkey]
    u[:, :, :, i] .= (jlfile2["timeseries"]["u"][ηkey][2:end, :, :] + jlfile2["timeseries"]["u"][ηkey][1:end-1, :, :])/2
    v[:, :, :, i] .= (jlfile2["timeseries"]["v"][ηkey][:, 2:end, :] + jlfile2["timeseries"]["v"][ηkey][:, 1:end-1, :])/2
    w[:, :, :, i] .= (jlfile2["timeseries"]["w"][ηkey][:, :, 2:end] + jlfile2["timeseries"]["w"][ηkey][:, :, 1:end-1])/2
    b[:, :, :, i] .= jlfile2["timeseries"]["b"][ηkey]
end
close(jlfile)
close(jlfile2)

NN = 3
fig = Figure() 
for i in 1:NN
    for j in 1:NN
        ii = (i - 1) * NN + j
        ax = Axis(fig[i, j]; xlabel = "x", ylabel = "y")
        heatmap!(ax, η[:, :, end - ii], colormap = :viridis)
    end
end
save(figure_directory * "etafield.png", fig)

fig = Figure() 
for i in 1:NN
    for j in 1:NN
        ii = (i - 1) * NN + j
        ax = Axis(fig[i, j]; xlabel = "x", ylabel = "y")
        heatmap!(ax, η[:, :, ii], colormap = :viridis)
    end
end
save(figure_directory * "etafield_start.png", fig)

for k in 1:2
    fig = Figure() 
    for i in 1:NN
        for j in 1:NN
            ii = (i - 1) * NN + j
            ax = Axis(fig[i, j]; xlabel = "x", ylabel = "y")
            heatmap!(ax, u[:, :, k, end - ii], colormap = :viridis)
        end
    end
    save(figure_directory * "ufield$k.png", fig)
end

for k in 1:2
    fig = Figure() 
    for i in 1:NN
        for j in 1:NN
            ii = (i - 1) * NN + j
            ax = Axis(fig[i, j]; xlabel = "x", ylabel = "y")
            heatmap!(ax, v[:, :, k, end - ii], colormap = :viridis)
        end
    end
    save(figure_directory * "vfield$k.png", fig)
end

fig = Figure() 
for i in 1:NN
    for j in 1:NN
        ii = (i - 1) * NN + j
        ax = Axis(fig[i, j]; xlabel = "x", ylabel = "y")
        heatmap!(ax, v[:, :, 2, end - ii], colormap = :viridis)
    end
end
save(figure_directory * "vfield2.png", fig)


fig = Figure() 
for i in 1:NN
    for j in 1:NN
        ii = (i - 1) * NN + j
        ax = Axis(fig[i, j]; xlabel = "x", ylabel = "y")
        heatmap!(ax, b[:, :, 1, end - ii], colormap = :viridis)
    end
end

save(figure_directory * "bfield1.png", fig)

fig = Figure() 
for i in 1:NN
    for j in 1:NN
        ii = (i - 1) * NN + j
        ax = Axis(fig[i, j]; xlabel = "x", ylabel = "y")
        heatmap!(ax, b[:, :, 2, end - ii], colormap = :viridis)
    end
end

save(figure_directory * "bfield2.png", fig)

fig = Figure() 
for i in 1:NN
    for j in 1:NN
        ii = (i - 1) * NN + j
        ax = Axis(fig[i, j]; xlabel = "x", ylabel = "y")
        heatmap!(ax, w[:, :, 2, end - ii], colormap = :viridis)
    end
end

save(figure_directory * "wfield2.png", fig)


##
squareheight = [mean(η[:, :, i] .^2) for i in eachindex(ηkeys)]


fig = Figure()
ax = Axis(fig[1,1]; xlabel = "time", ylabel = "mean(η^2)")
lines!(ax, squareheight[120:end], color = :blue)
save(figure_directory * "squareheight.png", fig)

##
si = 120 #starting index
η̄ = mean(η[:,:,si:end], dims = 3)
ση = std(η[:,:,si:end], dims = 3)
rη = (η[:, :, si:end] .- η̄ ) ./ ση

ū = mean(u[:,:,:,si:end], dims = 4)
σu = std(u[:,:,:,si:end], dims = 4)
ru = (u[:, :, :, si:end] .- ū ) ./ σu

v̄ = mean(v[:,:,:,si:end], dims = 4)
σv = std(v[:,:,:,si:end], dims = 4)
rv = (v[:, :, :, si:end] .- v̄ ) ./ σv

b̄ = mean(b[:,:,:,si:end], dims = 4)
σb = std(b[:,:,:,si:end], dims = 4)
rb = (b[:, :, :, si:end] .- b̄ ) ./ σb

hfile = h5open(data_directory * "baroclinic_double_gyre.hdf5", "w")
hfile["eta"] = rη
hfile["u"] = ru
hfile["v"] = rv
hfile["b"] = rb
hfile["mean eta"] = η̄
hfile["mean u"] = ū
hfile["mean v"] = v̄
hfile["mean b"] = b̄
hfile["std eta"] = ση
hfile["std u"] = σu
hfile["std v"] = σv
hfile["std b"] = σb
close(hfile)

state = zeros(M, N, 4, length(ηkeys) - si+1)
state[:, :, 1, :] .= ru[:, :, 1, :]
state[:, :, 2, :] .= rv[:, :, 1, :]
state[:, :, 3, :] .= rb[:, :, 1, :]
state[:, :, 4, :] .= rη[:, :, :]
hfile = h5open(data_directory * "baroclinic_training_data.hdf5", "w")
hfile["timeseries"] = state
close(hfile)