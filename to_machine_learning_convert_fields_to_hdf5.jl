using CairoMakie, JLD2, Statistics, HDF5, Oceananigans, ProgressBars
using KernelAbstractions
using Oceananigans.Utils
using Oceananigans.Architectures: device
using KernelAbstractions: @kernel, @index
using Oceananigans.Operators
using Oceananigans.Architectures: architecture

plot_data = true 
save_data = true
plot_stream_function = true
data_directory = "/nobackup1/sandre/OceananigansData/"
figure_directory = "oceananigans_figure/"

casevar = 7

si = 1000 #starting index
levels = 1:1
kmax = 2
NN = 3

jlfile = jldopen(data_directory * "baroclinic_double_gyre_free_surface_$casevar.jld2", "r")
ηkeys =  keys(jlfile["timeseries"]["η"])[2:end]

η = zeros(size(jlfile["timeseries"]["η"]["0"])[1:2]..., length(ηkeys))

for (i, ηkey) in ProgressBar(enumerate(ηkeys))
    η[:, :, i] .= jlfile["timeseries"]["η"][ηkey]
end
close(jlfile)
@info "closing jld2 file"

if plot_data 
    @info "plotting data"
    etamax = maximum(abs.(η[:, :, end]))
    fig = Figure() 
    for i in 1:NN
        for j in 1:NN
            ii = (i - 1) * NN + j
            ax = Axis(fig[i, j]; xlabel = "x", ylabel = "y")
            heatmap!(ax, η[:, :, end - ii], colormap = :balance, colorrange = (-etamax, etamax))
        end
    end
    save(figure_directory * "etafield.png", fig)

    squareheight = [mean(η[:, :, i] .^2) for i in eachindex(ηkeys)]


    fig = Figure()
    ax = Axis(fig[1,1]; xlabel = "time", ylabel = "mean(η^2)")
    lines!(ax, squareheight, color = :blue)
    save(figure_directory * "squareheight.png", fig)
end

if save_data
    η̄ = mean(η[:,:,si:end], dims = 3)
    ση = std(η[:,:,si:end], dims = 3)
    rη = (η[:, :, si:end] .- η̄ ) ./ ση
    hfile = h5open(data_directory * "baroclinic_double_gyre_$casevar.hdf5", "w")
    hfile["eta"] = rη
    hfile["mean eta"] = η̄
    hfile["std eta"] = ση
end

jlfile2 = jldopen(data_directory * "baroclinic_double_gyre_$casevar.jld2", "r")
M, N, L = size(jlfile2["timeseries"]["b"]["0"])
ϕ = zeros(M, N, L, length(ηkeys));
@info "u field"
for (i, ηkey) in ProgressBar(enumerate(ηkeys))
    tmp = jlfile2["timeseries"]["u"][ηkey]
    ϕ[:, :, :, i] .= (tmp[2:end, :, :] .+ tmp[1:end-1, :, :])/2
end

if save_data
    ū = mean(ϕ[:,:,levels,si:end], dims = 4)
    σu = std(ϕ[:,:,levels,si:end], dims = 4)
    ru = (ϕ[:, :, levels, si:end] .- ū ) ./ σu

    hfile = h5open(data_directory * "baroclinic_double_gyre_$casevar.hdf5", "r+")
    hfile["u"] = ru
    hfile["mean u"] = ū
    hfile["std u"] = σu
    close(hfile)
end

if plot_data
    @info "plotting data"
    for k in 1:kmax
        fig = Figure() 
        for i in 1:NN
            for j in 1:NN
                ii = (i - 1) * NN + j
                ax = Axis(fig[i, j]; xlabel = "x", ylabel = "y")
                heatmap!(ax, ϕ[:, :, k, end - ii], colormap = :balance, colorrange = (-0.5, 0.5))
            end
        end
        save(figure_directory * "ufield$k.png", fig)
    end
end

@info "v field"
for (i, ηkey) in ProgressBar(enumerate(ηkeys))
    tmp = jlfile2["timeseries"]["v"][ηkey]
    ϕ[:, :, :, i] .= (tmp[:, 2:end, :] .+ tmp[:, 1:end-1, :])/2
end

if save_data 
    v̄ = mean(ϕ[:,:,levels,si:end], dims = 4)
    σv = std(ϕ[:,:,levels,si:end], dims = 4)
    rv = (ϕ[:, :, levels, si:end] .- v̄ ) ./ σv

    hfile = h5open(data_directory * "baroclinic_double_gyre_$casevar.hdf5", "r+")
    hfile["v"] = rv
    hfile["mean v"] = v̄
    hfile["std v"] = σv
    close(hfile)
end

if plot_data
    @info "plotting data"
    for k in 1:kmax
        fig = Figure() 
        for i in 1:NN
            for j in 1:NN
                ii = (i - 1) * NN + j
                ax = Axis(fig[i, j]; xlabel = "x", ylabel = "y")
                heatmap!(ax, ϕ[:, :, k, end - ii], colormap = :balance, colorrange = (-0.5, 0.5))
            end
        end
        save(figure_directory * "vfield$k.png", fig)
    end
end

@info "b field"
for (i, ηkey) in ProgressBar(enumerate(ηkeys))
    tmp = jlfile2["timeseries"]["b"][ηkey]
    ϕ[:, :, :, i] .= tmp
end
close(jlfile2)

if save_data
    b̄ = mean(ϕ[:,:,levels,si:end], dims = 4)
    σb = std(ϕ[:,:,levels,si:end], dims = 4)
    rb = (ϕ[:, :, levels, si:end] .- b̄ ) ./ σb

    hfile = h5open(data_directory * "baroclinic_double_gyre_$casevar.hdf5", "r+")
    hfile["b"] = rb
    hfile["mean b"] = b̄
    hfile["std b"] = σb
    close(hfile)
end

if plot_data
    @info "plotting data"
    fig = Figure() 
    for k in 1:kmax
        for i in 1:NN
            for j in 1:NN
                ii = (i - 1) * NN + j
                ax = Axis(fig[i, j]; xlabel = "x", ylabel = "y")
                heatmap!(ax, ϕ[:, :, k, end - ii], colormap = :viridis)
            end
        end

        save(figure_directory * "bfield$k.png", fig)
    end
end

if save_data
    @info "saving training data"
    state = zeros(M, N, 4, length(ηkeys) - si+1)
    state[:, :, 1, :] .= ru[:, :, 1, :]
    state[:, :, 2, :] .= rv[:, :, 1, :]
    state[:, :, 3, :] .= rb[:, :, 1, :]
    state[:, :, 4, :] .= rη[:, :, :]
    hfile = h5open(data_directory * "baroclinic_training_data_$casevar.hdf5", "w")
    hfile["timeseries"] = state
    close(hfile)
end

if plot_stream_function 


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
    ψ = interior(barotropic_streamfunction(u[end]))[:, :, 1]

    Nt = size(u, 4)
    Ntmin = max(1, Nt-120)
    avgψ = mean([interior(barotropic_streamfunction(u[i]))[:,:,1] for i in Ntmin:Nt])

    fig = Figure()
    psimax = quantile(avgψ[:], 0.99)
    ax = Axis(fig[1,1]; xlabel = "x", ylabel = "y", title = "instantaneous streamfunction")
    heatmap!(ax, ψ, colormap = :balance, colorrange = (-psimax, psimax))
    ax = Axis(fig[1,2]; xlabel = "x", ylabel = "y", title = "average streamfunction")
    heatmap!(ax, avgψ, colormap = :balance, colorrange = (-psimax, psimax))
    save(figure_directory * "streamfunction.png", fig)

end