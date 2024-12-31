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
@info "loading u"
u = FieldTimeSeries(data_directory  * "baroclinic_double_gyre_$casevar.jld2", "u"; backend = InMemory(225))
@info "loading v"
v = FieldTimeSeries(data_directory  * "baroclinic_double_gyre_$casevar.jld2", "v"; backend = InMemory(225))
@info "loading w"
w = FieldTimeSeries(data_directory  * "baroclinic_double_gyre_$casevar.jld2", "w"; backend = InMemory(225))
@info "loading b"
b = FieldTimeSeries(data_directory  * "baroclinic_double_gyre_$casevar.jld2", "b"; backend = InMemory(225))

zs = u.grid.zᵃᵃᶜ[1:15]

levels = setdiff(1:15, [3, collect(9:14)...])
timekeys = 1126:length(ηkeys)

Ns = reverse([2^i for i in 1:1])
for N in Ns
    M = 256 ÷ N
    @info "saving eta"
    mean_etas_avgd_C = zeros(M, M, 1, length(timekeys));
    for i in ProgressBar(eachindex(timekeys))
        mean_etas_avgd_C[:, :, 1, i] .= coarse_grain(η[:, :, timekeys[i]], N)
    end
    hfile = h5open(analysis_directory * "training_baroclinic_double_gyre_$(M)_$(casevar)_complement.hdf5", "w")
    hfile["eta"] = mean_etas_avgd_C
    close(hfile)

    @info "saving b"
    averaged_field = zeros(M, M, length(levels), length(timekeys));
    for i in ProgressBar(eachindex(timekeys))
        bfield = interior(b[timekeys[i]])
        for k in eachindex(levels)
            averaged_field[:, :, k, i] .= coarse_grain(bfield[:, :, levels[k]], N)
        end
    end
    hfile = h5open(analysis_directory * "training_baroclinic_double_gyre_$(M)_$(casevar)_complement.hdf5", "r+")
    hfile["b"] = averaged_field
    close(hfile)

    @info "saving u"
    for i in ProgressBar(eachindex(timekeys))
        ufield = interior(u[timekeys[i]])
        ufield = (ufield[1:end-1, 1:end, :] .+ ufield[2:end, 1:end, :])/2
        for k in eachindex(levels)
            averaged_field[:, :, k, i] .= coarse_grain(ufield[:, :, levels[k]], N)
        end
    end

    hfile = h5open(analysis_directory * "training_baroclinic_double_gyre_$(M)_$(casevar)_complement.hdf5", "r+")
    hfile["u"] = averaged_field
    close(hfile)

    @info "saving v"
    for i in ProgressBar(eachindex(timekeys))
        vfield = interior(v[timekeys[i]])
        vfield = (vfield[1:end, 1:end-1, :] .+ vfield[1:end, 2:end, :])/2
        for k in eachindex(levels)
            averaged_field[:, :, k, i] .= coarse_grain(vfield[:, :, levels[k]], N)
        end
    end

    hfile = h5open(analysis_directory * "training_baroclinic_double_gyre_$(M)_$(casevar)_complement.hdf5", "r+")
    hfile["v"] = averaged_field
    close(hfile)

    @info "saving w"
    for i in ProgressBar(eachindex(timekeys))
        wfield = interior(w[timekeys[i]])
        wfield = (wfield[1:end, 1:end, 1:end-1] .+ wfield[1:end, 1:end, 2:end])/2
        for k in eachindex(levels)
            averaged_field[:, :, k, i] .= coarse_grain(wfield[:, :, levels[k]], N)
        end
    end

    hfile = h5open(analysis_directory * "training_baroclinic_double_gyre_$(M)_$(casevar)_complement.hdf5", "r+")
    hfile["w"] = averaged_field
    close(hfile)

    hfile = h5open(analysis_directory * "training_baroclinic_double_gyre_$(M)_$(casevar)_complement.hdf5", "r+")
    hfile["z"] = zs[levels]
    hfile["time"] = collect(timekeys) 
    close(hfile)
end
