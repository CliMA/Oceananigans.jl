using Printf
using NCDatasets
using PyCall
using PyPlot

using Oceananigans
using Oceananigans.Utils
using JULES

cmocean = pyimport("cmocean")

const plt = PyPlot
const km = kilometers

Tvar = Energy

ds = NCDataset("dry_rising_thermal_bubble_$(Tvar)_bad.nc")

function plot_dry_rising_thermal_bubble_frame(n)
    fig, axes = plt.subplots(nrows=2, ncols=2, sharex=true, sharey=true, figsize=(16, 9), dpi=200)

    xC, xF = ds["xC"] / km, ds["xF"] / km
    zC, zF = ds["zC"] / km, ds["zF"] / km

    ρ₀ = ds["ρ reference"][:, 1, :]'
    ρe₀ = ds["ρe reference"][:, 1, :]'

    ρ  = ds["ρ"][:, 1, :, n]'
    ρw = ds["ρw"][:, 1, 2:end, n]'
    ρu = ds["ρu"][:, 1, :, n]'
    ρe = ds["ρe"][:, 1, :, n]'

    fig.suptitle("Dry rising thermal bubble: time = $(prettytime(ds["time"][n]))")

    ax_u = axes[1, 1]
    im = ax_u.pcolormesh(xF, zC, ρu ./ ρ, cmap=cmocean.cm.balance)
    ax_u.set_title("u(x, z)")
    fig.colorbar(im, ax=ax_u)

    ax_w = axes[1, 2]
    im = ax_w.pcolormesh(xC, zC, ρw ./ ρ, cmap=cmocean.cm.balance)
    ax_w.set_title("w(x, z)")
    fig.colorbar(im, ax=ax_w)

    ax_ρ = axes[2, 1]
    im = ax_ρ.pcolormesh(xC, zC, ρ - ρ₀, cmap=cmocean.cm.dense)
    ax_ρ.set_title("ρ′(x, z)")
    fig.colorbar(im, ax=ax_ρ)

    ax_e = axes[2, 2]
    im = ax_e.pcolormesh(xC, zC, (ρe .- ρe₀) ./ ρ, cmap=cmocean.cm.thermal)
    ax_e.set_title("e′(x, z)")
    fig.colorbar(im, ax=ax_e)

    plt.xlim(-10, 10)
    plt.ylim(0, 10)

    filename = @sprintf("dry_rising_thermal_bubble_%03d.png", n)
    plt.savefig(filename)
end

# close(ds)