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
figure_directory = "oceananigans_figure/"
figure_directory = "quick_check_figure/"

casevar = 5

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


month_indices = zeros(Int, NN^2)
if length(ηkeys) ≥ 5000
    month_indices[1] = 1
    month_indices[2] = argmax( squareheight )
    month_indices[3] = 100 
    month_indices[4] = 1000
    month_indices[5] = 2000
    month_indices[6] = 3000
    month_indices[7] = 4000
    month_indices[8] = 5000
    month_indices[9] = 5100
else
    month_indices .= rand(1:length(ηkeys), NN^2)
end

if plot_data 
    @info "plotting data"
    etamax = maximum(abs.(η[:, :, end]))
    fig = Figure() 
    for i in 1:NN
        for j in 1:NN
            ii = (i - 1) * NN + j
            ax = Axis(fig[i, j]; xlabel = "x", ylabel = "y", title = "$(month_indices[ii])")
            heatmap!(ax, η[:, :, month_indices[ii]], colormap = :balance, colorrange = (-etamax, etamax))
        end
    end
    save(figure_directory * "etafield.png", fig)

    @info "plotting coarse-grained data"
    for factor in [2, 4, 8, 16, 32, 64, 128]
        etamax = maximum(abs.(η[:, :, end]))
        fig = Figure() 
        for i in 1:NN
            for j in 1:NN
                ii = (i - 1) * NN + j
                ax = Axis(fig[i, j]; xlabel = "x", ylabel = "y", title = "$(month_indices[ii])")
                heatmap!(ax, coarse_grain(η[:, :, month_indices[ii]], factor), colormap = :balance, colorrange = (-etamax, etamax))
            end
        end
        save(figure_directory * "coarsegrained_factor_$(factor)_etafield.png", fig)
    end

    squareheight = [mean(η[:, :, i] .^2) for i in eachindex(ηkeys)]


    fig = Figure()
    ax = Axis(fig[1,1]; xlabel = "time", ylabel = "mean(η^2)")
    lines!(ax, squareheight, color = :blue)
    save(figure_directory * "squareheight.png", fig)
end


u = FieldTimeSeries(data_directory  * "baroclinic_double_gyre_$casevar.jld2", "u"; backend = InMemory(225))
v = FieldTimeSeries(data_directory  * "baroclinic_double_gyre_$casevar.jld2", "v"; backend = InMemory(225))
w = FieldTimeSeries(data_directory  * "baroclinic_double_gyre_$casevar.jld2", "w"; backend = InMemory(225))
b = FieldTimeSeries(data_directory  * "baroclinic_double_gyre_$casevar.jld2", "b"; backend = InMemory(225))


meanabsus = zeros(length(ηkeys))
for i in ProgressBar(1:length(ηkeys))
    meanabsus[i] = mean(abs.(interior(u[i])[:, :, 1]))
end
meanabsvs = zeros(length(ηkeys))
for i in ProgressBar(1:length(ηkeys))
    meanabsvs[i] = mean(abs.(interior(v[i])[:, :, 1]))
end
meanabsws = zeros(length(ηkeys))
for i in ProgressBar(1:length(ηkeys))
    meanabsws[i] = mean(abs.(interior(w[i])[:, :, 1]))
end
meanbs = zeros(length(ηkeys))
for i in ProgressBar(1:length(ηkeys))
    meanbs[i] = mean(interior(b[i])[:, :, 1])
end

fig = Figure()
ax = Axis(fig[1,1]; xlabel = "time", ylabel = "mean(bottom_b)")
lines!(ax, meanbs, color = :blue)
ax = Axis(fig[1,2]; xlabel = "time", ylabel = "mean(|bottom_u|)")
lines!(ax, meanabsus, color = :blue)
ax = Axis(fig[2, 1]; xlabel = "time", ylabel = "mean(|bottom_v|)")
lines!(ax, meanabsvs, color = :blue)
ax = Axis(fig[2, 2]; xlabel = "time", ylabel = "mean(|bottom_w|)")
lines!(ax, meanabsws, color = :blue)
save(figure_directory * "bottom_layer_means.png", fig)
3+3