# Figures 4–6 and 8 of Jansen et al. (2015) from the output of jansen_2015_channel.jl.
# Run from the directory holding the output: julia --project plot_jansen_2015.jl

using Oceananigans
using Oceananigans.Units
using CairoMakie
using FFTW
using Statistics

runs = [replace(file, "_averages.jld2" => "") for file in readdir() if endswith(file, "_averages.jld2")]

# The second averaging window holds the statistically steady state
last_average(run, name) = FieldTimeSeries(run * "_averages.jld2", name)[end]

# One color per resolution and backscatter setting; WENO runs are dashed
without_weno(run) = replace(run, "_weno" => "")
colors = Dict(run => color for (run, color) in zip(unique(without_weno.(runs)), Makie.wong_colors()))
line_style(run) = (color = colors[without_weno(run)], linestyle = occursin("weno", run) ? :dash : :solid,
                   label = replace(run, "jansen_2015_" => ""))

#####
##### Fig. 5: overturning transport Ψ = [v₁h₁]
#####

fig = Figure(size=(700, 450))
ax = Axis(fig[1, 1], xlabel="y (km)", ylabel="Ψ (m² s⁻¹)")

for run in runs
    V₁ = last_average(run, "V₁")
    Ψ = Field(Average(V₁, dims=1))
    y = ynodes(Ψ) ./ kilometers
    lines!(ax, y, interior(Ψ, 1, :, 1); line_style(run)...)
    println(run, ": peak |Ψ| = ", round(maximum(abs, interior(Ψ)), digits=2), " m² s⁻¹")
end

axislegend(ax)
save("jansen_2015_overturning.png", fig)

#####
##### Figs. 4 and 8: time-mean EKE near the surface and near the bottom
#####

ekes = map(runs) do run
    u, v = last_average(run, "u"), last_average(run, "v")
    uu, vv = last_average(run, "uu"), last_average(run, "vv")
    eke = Field((uu - u * u + vv - v * v) / 2)
    println(run, ": mean upper EKE = ", round(mean(interior(eke, :, :, size(eke, 3))), sigdigits=2), " m² s⁻²")
    eke
end

fig = Figure(size=(550, 300 * length(runs)))

Nz = size(first(ekes), 3)
for (m, k, label) in ((1, Nz, "upper"), (2, 2, "lower"))
    colorrange = (0, maximum(eke -> maximum(interior(eke, :, :, k)), ekes))
    for (n, (run, eke)) in enumerate(zip(runs, ekes))
        eke_axis = Axis(fig[n, m], title=replace(run, "jansen_2015_" => "") * " $label", aspect=1)
        heatmap!(eke_axis, view(eke, :, :, k); colormap=:viridis, colorrange)
    end
    Colorbar(fig[length(runs) + 1, m]; colormap=:viridis, colorrange, vertical=false, label="$label EKE (m² s⁻²)")
end

save("jansen_2015_eke.png", fig)

#####
##### Fig. 6: spectrum of barotropic EKE
#####

fig = Figure(size=(600, 450))
ax = Axis(fig[1, 1], xscale=log10, yscale=log10, xlabel="K (2π / L)", ylabel="E(K)")

for run in runs
    Ut = FieldTimeSeries(run * "_snapshots.jld2", "U")
    Vt = FieldTimeSeries(run * "_snapshots.jld2", "V")
    grid = Ut.grid
    one_field = CenterField(grid)
    set!(one_field, 1)
    H = interior(Field(Integral(one_field, dims=3)), :, :, 1)

    Nx, Ny = size(grid, 1), size(grid, 2)
    averaging = findall(t -> t > Ut.times[end] / 2, Ut.times)
    U = [interior(Ut[n], :, :, 1) ./ max.(H, 1) for n in averaging]
    V = [(view(interior(Vt[n]), :, 1:Ny, 1) .+ view(interior(Vt[n]), :, 2:Ny+1, 1)) ./ 2 ./ max.(H, 1) for n in averaging]
    Ū, V̄ = mean(U), mean(V)

    E = sum(abs2.(fft(Uₙ .- Ū)) .+ abs2.(fft(Vₙ .- V̄)) for (Uₙ, Vₙ) in zip(U, V)) ./ (2 * length(U) * (Nx * Ny)^2)

    k = fftfreq(Nx, Nx)
    l = fftfreq(Ny, Ny)
    K = [round(Int, sqrt(kᵢ^2 + lⱼ^2)) for kᵢ in k, lⱼ in l]
    spectrum = [sum(E[K .== n]) for n in 1:maximum(K)]

    lines!(ax, 1:maximum(K), spectrum; line_style(run)...)
end

axislegend(ax, position=:lb)
save("jansen_2015_spectra.png", fig)
