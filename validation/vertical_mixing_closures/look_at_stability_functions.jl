using Oceananigans

using Oceananigans.TurbulenceClosures.TKEBasedVerticalDiffusivities:
    TKEDissipationVerticalDiffusivity,
    momentum_stability_function,
    maximum_stratification_number,
    minimum_stratification_number,
    maximum_shear_number,
    minimum_shear_number,
    tracer_stability_function

using GLMakie
using Printf

closure = TKEDissipationVerticalDiffusivity()

αᴺmin = minimum_stratification_number(closure)
αᴺmax = 30.0
αᴺ = αᴺmin:0.01:αᴺmax

αᴹmin = minimum_shear_number(closure)
αᴹmax = maximum_shear_number.(Ref(closure), αᴺ)
αᴹ = αᴹmin:0.01:maximum(αᴹmax)

fig = Figure(size=(1600, 600))
ax1 = Axis(fig[1, 1], title="Stability functions", xlabel="αᴺ", ylabel="𝕊c")
ax2 = Axis(fig[2, 1], title="Prandtl number", xlabel="αᴺ", ylabel="𝕊c")
ax3 = Axis(fig[1:2, 2], title="Tracer stability functions", xlabel="αᴺ", ylabel="αᴹ")
ax4 = Axis(fig[1:2, 3], title="Momentum stability functions", xlabel="αᴺ", ylabel="αᴹ")
ax5 = Axis(fig[1:2, 4], title="Prandtl number", xlabel="αᴺ", ylabel="αᴹ")

𝕊c_max_αᴹ = tracer_stability_function.(Ref(closure), αᴺ, αᴹmax)
𝕊u_max_αᴹ = momentum_stability_function.(Ref(closure), αᴺ, αᴹmax)

𝕊c_min_αᴹ = tracer_stability_function.(Ref(closure), αᴺ, αᴹmin)
𝕊u_min_αᴹ = momentum_stability_function.(Ref(closure), αᴺ, αᴹmin)

NN = length(αᴺ)
NM = length(αᴹ)
𝕊c = tracer_stability_function.(Ref(closure), reshape(αᴺ, NN, 1), reshape(αᴹ, 1, NM))
𝕊u = momentum_stability_function.(Ref(closure), reshape(αᴺ, NN, 1), reshape(αᴹ, 1, NM))
Pr = 𝕊u ./ 𝕊c

Pr_max = maximum(Pr, dims=2)[:]
Pr_min = minimum(Pr, dims=2)[:]

lines!(ax1, αᴺ, 𝕊c_max_αᴹ, label="max(αᴹ)", color=:blue)
lines!(ax1, αᴺ, 𝕊c_min_αᴹ, label="min(αᴹ)", linestyle=:dash, color=:blue)

lines!(ax1, αᴺ, 𝕊u_max_αᴹ, color=:red)
lines!(ax1, αᴺ, 𝕊u_min_αᴹ, linestyle=:dash, color=:red)

band!(ax1, αᴺ, 𝕊c_min_αᴹ, 𝕊c_max_αᴹ, label="𝕊c", color=(:blue, 0.5))
band!(ax1, αᴺ, 𝕊u_min_αᴹ, 𝕊u_max_αᴹ, label="𝕊u", color=(:red, 0.5))
axislegend(ax1)

band!(ax2, αᴺ, Pr_min, Pr_max, label="Pr", color=(:blue, 0.5))

cf = contourf!(ax3, αᴺ, αᴹ, 𝕊c, levels=0.01:0.03:0.2, colorrrange=(0.01, 0.2))
cf = contourf!(ax4, αᴺ, αᴹ, 𝕊u, levels=0.01:0.03:0.2, colorrrange=(0.01, 0.2))
Colorbar(fig[3, 2:3], cf, vertical=false, tellwidth=false, label="Stability functions", flipaxis=false)

cf = contourf!(ax6, αᴺ, αᴹ, Pr, levels=0.3:0.1:3.0, colorrrange=(0.35, 2.8), colormap=:solar)
Colorbar(fig[3, 4], cf, vertical=false, tellwidth=false, label="Prandtl number", flipaxis=false)

display(fig)

