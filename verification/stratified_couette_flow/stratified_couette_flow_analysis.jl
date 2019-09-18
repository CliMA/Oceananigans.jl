using Printf, Statistics
using JLD2, PyPlot

Nxy, Nz = 128, 128
Ris = (0, 0.01, 0.04)

width, height = 12, 6.75

c = Dict(0 => "tab:blue", 0.01 => "tab:orange", 0.04 => "tab:green")

####
#### Load data from JLD2
####

statistic_files = Dict()
statistic_iters = Dict()

profile_files = Dict()
profile_iters = Dict()

field_files = Dict()
field_iters = Dict()

for Ri in Ris
    base_dir = @sprintf("stratified_couette_flow_data_Nxy%d_Nz%d_Ri%.2f", Nxy, Nz, Ri)
    prefix = @sprintf("stratified_couette_flow_Nxy%d_Nz%d_Ri%.2f", Nxy, Nz, Ri)

    statistic_filepath = joinpath(base_dir, prefix * "_statistics.jld2")
    profile_filepath = joinpath(base_dir, prefix * "_profiles.jld2")
    field_filepath = joinpath(base_dir, prefix * "_fields.jld2")

    statistic_files[Ri] = jldopen(statistic_filepath, "r")
    statistic_iters[Ri] = keys(statistic_files[Ri]["timeseries/t"])

    profile_files[Ri] = jldopen(profile_filepath, "r")
    profile_iters[Ri] = keys(profile_files[Ri]["timeseries/t"])

    field_files[Ri] = jldopen(field_filepath, "r")
    field_iters[Ri] = keys(field_files[Ri]["timeseries/t"])
end

####
#### Plot Re and Nu timeseries
####

fig, (ax1, ax2) = subplots(nrows=2, ncols=1, figsize=(width, height))

Reτ = Dict()
Nu  = Dict()

for Ri in Ris
    i = statistic_iters[Ri][end]

    t   = [statistic_files[Ri]["timeseries/t/"      * i]    for i in statistic_iters[Ri]]
    ReT = [statistic_files[Ri]["timeseries/Re_tau/" * i][1] for i in statistic_iters[Ri]]
    ReB = [statistic_files[Ri]["timeseries/Re_tau/" * i][2] for i in statistic_iters[Ri]]
    NuT = [statistic_files[Ri]["timeseries/Nu_tau/" * i][1] for i in statistic_iters[Ri]]
    NuB = [statistic_files[Ri]["timeseries/Nu_tau/" * i][2] for i in statistic_iters[Ri]]

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
ax1.legend(ncol=3, frameon=false)

ax2.set_xlabel(L"tU_w/h")
ax2.set_ylabel("Nu")
ax2.set_xlim([0, 1000])
ax2.legend(ncol=2, frameon=false)

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

fig, (ax1, ax2) = subplots(nrows=1, ncols=2, figsize=(width, height))

ax1.scatter(Ris, [Reτ_ZTC17[Ri] for Ri in Ris], color="tab:orange", marker="s", label="DNS 256x129x256 (ZTC 2017)")
ax1.scatter(Ris, [Reτ_VT18[Ri]  for Ri in Ris], color="tab:green",  marker="x", label="LES 64x49x64 (V&T 2018)")
ax1.scatter(Ris, [Reτ[Ri]       for Ri in Ris], color="tab:blue",   marker="o", label=L"$128^3$ Oceananigans")

ax2.scatter(Ris, [Nu_ZTC17[Ri] for Ri in Ris], color="tab:orange", marker="s", label="DNS 256x129x256 (ZTC 2017)")
ax2.scatter(Ris, [Nu_VT18[Ri]  for Ri in Ris], color="tab:green",  marker="x", label="LES 64x49x64 (V&T 2018)")
ax2.scatter((0.01, 0.04), [Nu[Ri] for Ri in (0.01, 0.04)], color="tab:blue",   marker="o", label=L"$128^3$ Oceananigans")

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

fig, (ax1, ax2) = subplots(nrows=1, ncols=2, figsize=(0.8width, 0.7height))

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
ax2.set_xlim([-1, 1])
ax2.set_ylim([-1, 1])
ax2.legend(frameon=false)

png_filepath = "stratified_couette_flow_velocity_temperature_profiles.png"
@info "Saving $png_filepath..."
savefig(png_filepath, dpi=200)
close(fig)

####
#### Plot LES model viscosity and diffusivity profiles
####

fig, (ax1, ax2, ax3) = subplots(nrows=1, ncols=3, figsize=(width, height))

for Ri in Ris
    i = field_iters[Ri][end]

    ν = field_files[Ri]["parameters/viscosity"]
    κ = field_files[Ri]["parameters/diffusivity"]

    Nx, Ny, Nz = field_files[Ri]["grid/Nx"], field_files[Ri]["grid/Ny"], field_files[Ri]["grid/Nz"]
    Hx, Hy, Hz = field_files[Ri]["grid/Hx"], field_files[Ri]["grid/Hy"], field_files[Ri]["grid/Hz"]
    zC = field_files[Ri]["grid/zC"]
    
    νSGS = field_files[Ri]["timeseries/nu/"     * i][1+Hx:Nx+Hx, 1+Hy:Ny+Hy, 1+Hz:Nz+Hz]
    κSGS = field_files[Ri]["timeseries/kappaT/" * i][1+Hx:Nx+Hx, 1+Hy:Ny+Hy, 1+Hz:Nz+Hz]

    PrSGS = νSGS ./ κSGS

    νSGSp  = mean(νSGS,  dims=[1, 2])[:]
    κSGSp  = mean(κSGS,  dims=[1, 2])[:]
    PrSGSp = mean(PrSGS, dims=[1, 2])[:]

    ax1.plot((νSGSp .- ν) ./ ν, zC .+ 1, color=c[Ri], label="Ri = $Ri")
    ax2.plot((κSGSp .- κ) ./ κ, zC .+ 1, color=c[Ri], label="Ri = $Ri")
    ax3.plot(PrSGSp,            zC .+ 1, color=c[Ri], label="Ri = $Ri")
end

ax1.set_xlabel(L"$\nu_{SGS} / \nu_0$")
ax1.set_ylabel(L"$z/h$")
ax1.set_ylim([-1, 1])
ax1.legend(frameon=false)

ax2.set_xlabel(L"$\kappa_{SGS} / \kappa_0$")
ax2.set_ylim([-1, 1])
ax2.legend(frameon=false)

ax3.set_xlabel(L"Pr$_{SGS}$")
ax3.set_ylim([-1, 1])
ax3.legend(frameon=false)

png_filepath = "stratified_couette_flow_LES_profiles.png"
@info "Saving $png_filepath..."
savefig(png_filepath, dpi=200)
close(fig)

####
#### Plot horizontal slices of velocity and temperature.
####

fig, axes = subplots(nrows=3, ncols=2, figsize=(width, 1.5*height))

for (idx, Ri) in enumerate(Ris)
    i = field_iters[Ri][end]

    Uw = field_files[Ri]["parameters/wall_velocity"]
    Θw = field_files[Ri]["parameters/wall_temperature"]

    Nx, Ny, Nz = field_files[Ri]["grid/Nx"], field_files[Ri]["grid/Ny"], field_files[Ri]["grid/Nz"]
    Hx, Hy, Hz = field_files[Ri]["grid/Hx"], field_files[Ri]["grid/Hy"], field_files[Ri]["grid/Hz"]

    xC, xF, yC, zC = field_files[Ri]["grid/xC"], field_files[Ri]["grid/xF"], field_files[Ri]["grid/yC"], field_files[Ri]["grid/zC"]
    
    u = field_files[Ri]["timeseries/u/" * i][1+Hx:Nx+Hx, 1+Hy:Ny+Hy, 1+Hz:Nz+Hz]
    θ = field_files[Ri]["timeseries/T/" * i][1+Hx:Nx+Hx, 1+Hy:Ny+Hy, 1+Hz:Nz+Hz]

    k = 5
    z_k = zC[k]

    im1 = axes[idx, 1].pcolormesh(xF ./ π, yC ./ π, u[:, :, k]' ./ Uw, cmap="viridis")
    im2 = axes[idx, 2].pcolormesh(xC ./ π, yC ./ π, θ[:, :, k]' ./ Θw, cmap="inferno")

    fig.colorbar(im1, ax=axes[idx, 1])
    fig.colorbar(im2, ax=axes[idx, 2])  
 
    axes[idx, 1].set_xlim([0, 2]) 
    axes[idx, 2].set_xlim([0, 2]) 
    axes[idx, 1].set_xlim([0, 4])
    axes[idx, 2].set_xlim([0, 4])

    axes[1, 1].set_title(L"$u(x,y)/U_w$ @ z = " * @sprintf("%.2f", z_k + 1))
    axes[1, 2].set_title(L"$\theta(x,y)/\Theta_w$ @ z = " * @sprintf("%.2f", z_k + 1))
end

axes[1, 1].set_ylabel(L"$y/\pi$")
axes[2, 1].set_ylabel(L"$y/\pi$")
axes[3, 1].set_ylabel(L"$y/\pi$")
axes[3, 1].set_xlabel(L"$x/\pi$")
axes[3, 2].set_xlabel(L"$x/\pi$")

png_filepath = "stratified_couette_flow_velocity_temperature_slices.png"
@info "Saving $png_filepath..."
savefig(png_filepath, dpi=200)
close(fig)

