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

@info "convergence with depth"
averages = zeros(length(ηkeys), 15, 5);
for i in ProgressBar(1:length(ηkeys))
    averages[i, :, 1] .= mean(abs.(interior(u[i])), dims = (1, 2))[:]
    averages[i, :, 2] .= mean(abs.(interior(v[i])), dims = (1, 2))[:]
    tmp = mean(abs.(interior(w[i])), dims = (1, 2))
    averages[i, :, 3] .= (tmp[1:end-1] + tmp[2:end]) / 2
    averages[i, :, 4] .= mean(abs.(interior(b[i])), dims = (1, 2))[:]
    averages[i, :, 5] .= mean(abs.(η[:, :, i]))
end


hfile = h5open(analysis_directory * "convergence_with_depth_$casevar.hdf5", "w") 
hfile["averages"] = averages
close(hfile)