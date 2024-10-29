using CairoMakie, JLD2, Statistics, HDF5, Oceananigans
data_directory = "/nobackup1/sandre/OceananigansData/"
figure_directory = "oceananigans_figure/"

casevar = 1
jlfile = jldopen(data_directory * "baroclinic_double_gyre_free_surface_$casevar.jld2", "r")
jlfile2 = jldopen(data_directory * "baroclinic_double_gyre_$casevar.jld2", "r")
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
kmax = 2
kmax = 2

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

for k in 1:kmax
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

for k in 1:kmax
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
for k in 1:kmax
    for i in 1:NN
        for j in 1:NN
            ii = (i - 1) * NN + j
            ax = Axis(fig[i, j]; xlabel = "x", ylabel = "y")
            heatmap!(ax, b[:, :, k, end - ii], colormap = :viridis)
        end
    end

    save(figure_directory * "bfield$k.png", fig)
end



fig = Figure() 
for k in 2:kmax
    for i in 1:NN
        for j in 1:NN
            ii = (i - 1) * NN + j
            ax = Axis(fig[i, j]; xlabel = "x", ylabel = "y")
            heatmap!(ax, w[:, :, k, end - ii], colormap = :viridis)
        end
    end
    save(figure_directory * "wfield$k.png", fig)
end


##
squareheight = [mean(η[:, :, i] .^2) for i in eachindex(ηkeys)]


fig = Figure()
ax = Axis(fig[1,1]; xlabel = "time", ylabel = "mean(η^2)")
lines!(ax, squareheight, color = :blue)
save(figure_directory * "squareheight.png", fig)

##
si = 200 #starting index
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

#=
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
=#

##

using Oceananigans.Utils
using Oceananigans.Architectures: device
using KernelAbstractions: @kernel, @index
using Oceananigans.Operators
using Oceananigans.Architectures: architecture

u = FieldTimeSeries(data_directory * "baroclinic_double_gyre_$casevar.jld2", "u")

function barotropic_streamfunction(u)
    U = Field(Integral(u, dims=3))
    compute!(U)
    ψ = Field{Face,Face,Nothing}(u.grid)
    D = device(architecture(u.grid))

    _compute_ψ!(D, 16, u.grid.Nx + 1)(ψ, u.grid, U)

    return ψ
end

@kernel function _compute_ψ!(ψ, grid, U)
    i = @index(Global, Linear)
    ψ[i, 1, 1] = 0

    for j in 2:grid.Ny
        ψ[i, j, 1] = ψ[i, j-1, 1] - U[i, j, 1] * Δyᶠᶜᶜ(i, j, 1, grid)
    end
end
ψ = barotropic_streamfunction(u[end])

Nt = size(u, 4)
avgψ = mean([interior(barotropic_streamfunction(u[i]))[:,:,1] for i in Nt-120:Nt])

fig = Figure()
psimax = quantile(avgψ[:], 0.99)
ax = Axis(fig[1,1]; xlabel = "x", ylabel = "y", title = "instantaneous streamfunction")
heatmap!(ax, ψ, colormap = :balance, colorrange = (-psimax, psimax))
ax = Axis(fig[1,2]; xlabel = "x", ylabel = "y", title = "average streamfunction")
heatmap!(ax, avgψ, colormap = :balance, colorrange = (-psimax, psimax))
psimax = quantile(avgψ[:], 0.99)
ax = Axis(fig[1,1]; xlabel = "x", ylabel = "y", title = "instantaneous streamfunction")
heatmap!(ax, ψ, colormap = :balance, colorrange = (-psimax, psimax))
ax = Axis(fig[1,2]; xlabel = "x", ylabel = "y", title = "average streamfunction")
heatmap!(ax, avgψ, colormap = :balance, colorrange = (-psimax, psimax))
# contour!(ax, ψ, color = :black)
save(figure_directory * "streamfunction.png", fig)