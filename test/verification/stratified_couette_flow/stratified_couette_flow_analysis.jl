using Printf
using JLD2, PyPlot

Nxy, Nz = 128, 64
Ris = (0, 0.01, 0.04)

scalar_files = Dict()
scalar_iters = Dict()

for Ri in Ris
    base_dir = @sprintf("stratified_couette_flow_data_Nxy%d_Nz%d_Ri%.2f", Nxy, Nz, Ri)
    prefix = @sprintf("stratified_couette_flow_Nxy%d_Nz%d_Ri%.2f", Nxy, Nz, Ri)
    
    scalar_filepath = joinpath(base_dir, prefix * "_scalars.jld2")
    
    scalar_files[Ri] = jldopen(scalar_filepath, "r")
    scalar_iters[Ri] = keys(scalar_files[Ri]["timeseries/t"])
end

####
#### Plot Re and Nu timeseries
####

fig, (ax1, ax2) = subplots(nrows=2, ncols=1, figsize=(16, 9))

c = Dict(Ris[1]=> "tab:blue", Ris[2]=> "tab:orange", Ris[3]=> "tab:green")

for Ri in Ris
    i = scalar_iters[Ri][end]

    t   = [scalar_files[Ri]["timeseries/t/"      * i]    for i in scalar_iters[Ri]]
    ReT = [scalar_files[Ri]["timeseries/Re_tau/" * i][1] for i in scalar_iters[Ri]]
    ReB = [scalar_files[Ri]["timeseries/Re_tau/" * i][2] for i in scalar_iters[Ri]]
    NuT = [scalar_files[Ri]["timeseries/Nu_tau/" * i][1] for i in scalar_iters[Ri]]
    NuB = [scalar_files[Ri]["timeseries/Nu_tau/" * i][2] for i in scalar_iters[Ri]]

    ax1.plot(t, ReT, color=c[Ri], linestyle="-",  label="Ri = $Ri (top wall)")
    ax1.plot(t, ReB, color=c[Ri], linestyle="--", label="Ri = $Ri (bottom wall)")
   
    if Ri != 0
        ax2.plot(t, NuT, color=c[Ri], linestyle="-",  label="Ri = $Ri (top wall)")
        ax2.plot(t, NuB, color=c[Ri], linestyle="--", label="Ri = $Ri (bottom wall)")
    end
end

ax1.set_xlabel(L"tU_w/h")
ax1.set_ylabel(L"Re$_\tau$")
ax1.set_xlim([0, 1000])
ax1.legend(frameon=false)

ax2.set_xlabel(L"tU_w/h")
ax2.set_ylabel("Nu")
ax2.set_xlim([0, 1000])
ax2.legend(frameon=false)

png_filepath = "stratified_couette_flow_Re_Nu_timeseries.png"
@info "Saving $png_filepath..."
savefig(png_filepath, dpi=200)
close(fig)
