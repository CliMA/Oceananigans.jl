using Printf, Statistics
using JLD2, PyPlot

Nxy, Nz = 128, 64
Ris = (0, 0.01, 0.04)

####
#### Load data from JLD2
####

scalar_files = Dict()
scalar_iters = Dict()

profile_files = Dict()
profile_iters = Dict()

for Ri in Ris
    base_dir = @sprintf("stratified_couette_flow_data_Nxy%d_Nz%d_Ri%.2f", Nxy, Nz, Ri)
    prefix = @sprintf("stratified_couette_flow_Nxy%d_Nz%d_Ri%.2f", Nxy, Nz, Ri)

    scalar_filepath = joinpath(base_dir, prefix * "_scalars.jld2")
    profile_filepath = joinpath(base_dir, prefix * "_profiles.jld2")

    scalar_files[Ri] = jldopen(scalar_filepath, "r")
    scalar_iters[Ri] = keys(scalar_files[Ri]["timeseries/t"])

    profile_files[Ri] = jldopen(profile_filepath, "r")
    profile_iters[Ri] = keys(profile_files[Ri]["timeseries/t"])
end

####
#### Plot Re and Nu timeseries
####

fig, (ax1, ax2) = subplots(nrows=2, ncols=1, figsize=(12, 6.75))

c = Dict(Ris[1]=> "tab:blue", Ris[2]=> "tab:orange", Ris[3]=> "tab:green")

Reτ = Dict()
Nu  = Dict()

for Ri in Ris
    i = scalar_iters[Ri][end]

    t   = [scalar_files[Ri]["timeseries/t/"      * i]    for i in scalar_iters[Ri]]
    ReT = [scalar_files[Ri]["timeseries/Re_tau/" * i][1] for i in scalar_iters[Ri]]
    ReB = [scalar_files[Ri]["timeseries/Re_tau/" * i][2] for i in scalar_iters[Ri]]
    NuT = [scalar_files[Ri]["timeseries/Nu_tau/" * i][1] for i in scalar_iters[Ri]]
    NuB = [scalar_files[Ri]["timeseries/Nu_tau/" * i][2] for i in scalar_iters[Ri]]

    ax1.plot(t, ReT, color=c[Ri], linestyle="-",  label="Ri = $Ri (top wall)")
    ax1.plot(t, ReB, color=c[Ri], linestyle="--", label="Ri = $Ri (bottom wall)")

    N½ = length(t) ÷ 2
    Reτ_t = mean(ReT[N½:end])
    Reτ_b = mean(ReB[N½:end])
    Reτ[Ri] = (Reτ_t + Reτ_b) / 2

    ax1.axhline(y=Reτ[Ri], xmin=0.5, xmax=1, color=c[Ri], linestyle=":")
    ax1.text(x=1005, y=Reτ[Ri], s=@sprintf("%.0f", Reτ[Ri]), color=c[Ri])

    if Ri != 0
        ax2.plot(t, NuT, color=c[Ri], linestyle="-",  label="Ri = $Ri (top wall)")
        ax2.plot(t, NuB, color=c[Ri], linestyle="--", label="Ri = $Ri (bottom wall)")

        Nu_t = mean(NuT[N½:end])
        Nu_b = mean(NuB[N½:end])
        Nu[Ri] = (Nu_t + Nu_b) / 2

        ax2.axhline(y=Nu[Ri], xmin=0.5, xmax=1, color=c[Ri], linestyle=":")
        ax2.text(x=1005, y=Nu[Ri], s=@sprintf("%.1f", Nu[Ri]), color=c[Ri])
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

####
#### Re and Nu scatter plot
####

# Re and Nu values reported in runs 3-5 of Table 1, Vreugdenhil & Taylor (2018).
Reτ_VT18 = Dict(0 => 223,  0.01 => 212, 0.04 => 183)
Nu_VT18  = Dict(0 => 10.6, 0.01 => 9.6, 0.04 => 7.1)

# Re and Nu values reported in runs 1-3 of Table 1, Zhou, Taylor, and Caulfield (2017).
Reτ_ZTC17 = Dict(0 => 233, 0.01 => 215, 0.04 => 181)
Nu_ZTC17  = Dict(0 => 10.6, 0.01 => 9.26, 0.04 => 6.40)

fig, (ax1, ax2) = subplots(nrows=1, ncols=2, figsize=(12, 6.75))

ax1.scatter(Ris, [Reτ_ZTC17[Ri] for Ri in Ris], color="tab:orange", marker="s", label="ZTC 2017")
ax1.scatter(Ris, [Reτ_VT18[Ri]  for Ri in Ris], color="tab:green",  marker="x", label="V&T 2018")
ax1.scatter(Ris, [Reτ[Ri]       for Ri in Ris], color="tab:blue",   marker="o", label="Oceananigans")

ax2.scatter(Ris, [Nu_ZTC17[Ri] for Ri in Ris], color="tab:orange", marker="s", label="ZTC 2017")
ax2.scatter(Ris, [Nu_VT18[Ri]  for Ri in Ris], color="tab:green",  marker="x", label="V&T 2018")
ax2.scatter((0.01, 0.04), [Nu[Ri] for Ri in (0.01, 0.04)], color="tab:blue",   marker="o", label="Oceananigans")

ax1.set_xlabel("Ri")
ax1.set_ylabel(L"Re$_\tau$")
ax1.legend(frameon=false)

ax2.set_xlabel("Ri")
ax2.set_ylabel("Nu")
ax2.legend(frameon=false)

png_filepath = "stratified_couette_flow_Re_Nu_scatter.png"
@info "Saving $png_filepath..."
savefig(png_filepath, dpi=200)
close(fig)

####
#### Plot velocity and temperature profiles
####

fig, (ax1, ax2) = subplots(nrows=1, ncols=2, figsize=(12, 3.375))

c = Dict(Ris[1]=> "tab:blue", Ris[2]=> "tab:orange", Ris[3]=> "tab:green")

for Ri in Ris
    i = profile_iters[Ri][end]

    Uw = profile_files[Ri]["parameters/wall_velocity"]
    Θw = profile_files[Ri]["parameters/wall_temperature"]

    Nz = profile_files[Ri]["grid/Nz"]
    Hz = profile_files[Ri]["grid/Hz"]
    zC = profile_files[Ri]["grid/zC"]

    U = profile_files[Ri]["timeseries/u/" * i][1+Hz:Nz+Hz]
    Θ = profile_files[Ri]["timeseries/T/" * i][1+Hz:Nz+Hz]

               ax1.plot(U/Uw, zC .+ 1, color=c[Ri], label="Ri = $Ri")
    Ri != 0 && ax2.plot(Θ/Θw, zC .+ 1, color=c[Ri], label="Ri = $Ri")
end

ax1.set_xlabel(L"$U/U_w$")
ax1.set_ylabel(L"z/h")
ax1.set_xlim([-1, 1])
ax1.set_ylim([-1, 1])
ax1.legend(frameon=false)

ax2.set_xlabel(L"$Θ/Θ_w$")
ax2.set_ylabel(L"z/h")
ax2.set_xlim([-1, 1])
ax2.set_ylim([-1, 1])
ax2.legend(frameon=false)

png_filepath = "stratified_couette_flow_velocity_temperature_profile.png"
@info "Saving $png_filepath..."
savefig(png_filepath, dpi=200)
close(fig)

####
#### Plot LES model viscosity and diffusivity profiles
####

####
#### Plot horizontal slices of velocity and temperature.
####

