using GLMakie
using Oceananigans
using Oceananigans.Units
using Printf
using Statistics

b_ts = []
u_ts = []
v_ts = []
e_ts = []
κᶜ_ts = []
κᵘ_ts = []

filepaths = [
    "windy_convection_CATKEVerticalDiffusivity.jld2",
    "new_windy_convection_CATKEVerticalDiffusivity.jld2",
    "fully_implicit_windy_convection_CATKEVerticalDiffusivity.jld2",
]

labels = ["Fully 'explicit'", "Partially 'implicit'", "Fully 'implicit'"]

for filepath in filepaths
    push!(b_ts, FieldTimeSeries(filepath, "b"))
    push!(u_ts, FieldTimeSeries(filepath, "u"))
    push!(v_ts, FieldTimeSeries(filepath, "v"))
    push!(e_ts, FieldTimeSeries(filepath, "e"))
    push!(κᶜ_ts, FieldTimeSeries(filepath, "κᶜ"))
    push!(κᵘ_ts, FieldTimeSeries(filepath, "κᵘ"))
end

b1 = first(b_ts)
e1 = first(e_ts)
κ1 = first(κᶜ_ts)
@show maximum(e_ts[1])
@show maximum(e_ts[2])

zc = znodes(b1)
zf = znodes(κ1)
times = b1.times
Nt = length(times)

fig = Figure(size=(1800, 600))

slider = Slider(fig[2, 1:4], range=1:Nt, startvalue=1)
n = slider.value

buoyancy_label = @lift "Buoyancy at t = " * prettytime(b1.times[$n])
velocities_label = @lift "Velocities at t = " * prettytime(b1.times[$n])
TKE_label = @lift "Turbulent kinetic energy t = " * prettytime(b1.times[$n])
diffusivities_label = @lift "Eddy diffusivities at t = " * prettytime(b1.times[$n])

b1n  = @lift interior(b_ts[1][$n], 1, 1, :)
u1n  = @lift interior(u_ts[1][$n], 1, 1, :)
v1n  = @lift interior(v_ts[1][$n], 1, 1, :)
e1n  = @lift interior(e_ts[1][$n], 1, 1, :)
κᶜ1n = @lift interior(κᶜ_ts[1][$n], 1, 1, :)
κᵘ1n = @lift interior(κᵘ_ts[1][$n], 1, 1, :)

b2n  = @lift interior(b_ts[2][$n], 1, 1, :)
u2n  = @lift interior(u_ts[2][$n], 1, 1, :)
v2n  = @lift interior(v_ts[2][$n], 1, 1, :)
e2n  = @lift interior(e_ts[2][$n], 1, 1, :)
κᶜ2n = @lift interior(κᶜ_ts[2][$n], 1, 1, :)
κᵘ2n = @lift interior(κᵘ_ts[2][$n], 1, 1, :)

b3n  = @lift interior(b_ts[3][$n], 1, 1, :)
u3n  = @lift interior(u_ts[3][$n], 1, 1, :)
v3n  = @lift interior(v_ts[3][$n], 1, 1, :)
e3n  = @lift interior(e_ts[3][$n], 1, 1, :)
κᶜ3n = @lift interior(κᶜ_ts[3][$n], 1, 1, :)
κᵘ3n = @lift interior(κᵘ_ts[3][$n], 1, 1, :)
 
btitle = @lift begin
    tstr = prettytime(times[$n])
    # mse = mean(($b1n .- $b2n).^2)
    #@sprintf("Buoyancy at t = %s, , mse = %.2e", mse)
    string("Buoyancy at t = ", tstr)
end

etitle = @lift begin
    #mse = mean(($e1n .- $e2n).^2)
    #@sprintf("TKE, mse = %.2e", mse)
    tstr = prettytime(times[$n])
    string("Turbulent kinetic energy at t = ", tstr)
end

axb = Axis(fig[1, 1], xlabel=buoyancy_label, ylabel="z (m)", title=btitle)
axu = Axis(fig[1, 2], xlabel=velocities_label, ylabel="z (m)")
axe = Axis(fig[1, 3], xlabel=TKE_label, ylabel="z (m)", title=etitle)
axκ = Axis(fig[1, 4], xlabel=diffusivities_label, ylabel="z (m)")

grid = b1.grid
N² = 1e-6
xlims!(axb, -grid.Lz * N², 0)
xlims!(axu, -0.2, 0.2)
xlims!(axe, -1e-4, 2e-4)
xlims!(axκ, -1e-1, 2e0)

colors = [
    (:black, 0.8),
    :royalblue,
    :indianred1,
    :orange]

i = 1
label = labels[i]
linewidth=5
lines!(axb, b1n,  zc; linewidth, label=label, color=colors[i])
lines!(axu, u1n,  zc; linewidth, label="u, " * label, color=colors[i])
lines!(axu, v1n,  zc; linewidth, label="v, " * label, linestyle=:dash, color=colors[i])
lines!(axe, e1n,  zc; linewidth, label="e, " * label, color=colors[i])
lines!(axκ, κᶜ1n, zf; linewidth, label="κᶜ, " * label, color=colors[i])
lines!(axκ, κᵘ1n, zf; linewidth, label="κᵘ, " * label, linestyle=:dash, color=colors[i])

i = 2
label = labels[i]
linewidth=5
lines!(axb, b2n,  zc; linewidth, label=label, color=colors[i])
lines!(axu, u2n,  zc; linewidth, label="u, " * label, color=colors[i])
lines!(axu, v2n,  zc; linewidth, label="v, " * label, linestyle=:dash, color=colors[i])
lines!(axe, e2n,  zc; linewidth, label="e, " * label, color=colors[i])
lines!(axκ, κᶜ2n, zf; linewidth, label="κᶜ, " * label, color=colors[i])
lines!(axκ, κᵘ2n, zf; linewidth, label="κᵘ, " * label, linestyle=:dash, color=colors[i])

i = 3
label = labels[i]
linewidth=5
lines!(axb, b3n,  zc; linewidth, label=label, color=colors[i])
lines!(axu, u3n,  zc; linewidth, label="u, " * label, color=colors[i])
lines!(axu, v3n,  zc; linewidth, label="v, " * label, linestyle=:dash, color=colors[i])
lines!(axe, e3n,  zc; linewidth, label="e, " * label, color=colors[i])
lines!(axκ, κᶜ3n, zf; linewidth, label="κᶜ, " * label, color=colors[i])
lines!(axκ, κᵘ3n, zf; linewidth, label="κᵘ, " * label, linestyle=:dash, color=colors[i])

axislegend(axb, position=:lb)
axislegend(axu, position=:rb)
axislegend(axe, position=:rb)
axislegend(axκ, position=:rb)

display(fig)

record(fig, "compare_catkes_windy_convection.mp4", 1:Nt, framerate=24) do nn
    n[] = nn
end

