# Inertia–gravity dispersion of C-grid Coriolis discretizations for linear shallow water on a uniform f-plane with
# R_d = Δ/4, in index units (Δ = 1, f = 1). Frequencies come from the Fourier symbol of each scheme:
# the four-point average (energy and enstrophy conserving), the C-D scheme of Adcroft et al. (1999) without
# relaxation, and the oriented operator form with ε = 1/8, ϵc = 1/32.

using LinearAlgebra
using CairoMakie
using Printf

const gH = 1 / 16

σ(k) = 2sin(k / 2)
D(k) = 2im * sin(k / 2)
oriented_m(k, l; ε=1/8, ϵc=1/32) = cos(k/2) * cos(l/2) + 4ϵc * (σ(k)^2 + σ(l)^2) * sin(k/2) * sin(l/2) +
                                2im * ε * (σ(k)^2 + σ(l)^2) * sin((k - l) / 2)

exact_frequency(k, l) = sqrt(1 + gH * (k^2 + l^2))

# Largest inertia–gravity frequency of a symbol whose first three unknowns are (u, v, η)
function inertia_gravity_frequency(symbol)
    F = eigen(symbol)
    weights = [norm(F.vectors[1:3, n]) / norm(F.vectors[:, n]) for n in axes(F.vectors, 2)]
    physical = sortperm(weights, rev=true)[1:3]
    return maximum(abs ∘ real, im .* F.values[physical])
end

frequency(::Val{:four_point}, k, l) = sqrt(cos(k/2)^2 * cos(l/2)^2 + gH * (σ(k)^2 + σ(l)^2))
frequency(::Val{:oriented}, k, l) = sqrt(abs2(oriented_m(k, l)) + gH * (σ(k)^2 + σ(l)^2))

# Unknowns (u, v, η, uᴰ, vᴰ): uᴰ at v points and vᴰ at u points, advanced with the averaged C-grid tendency
function frequency(::Val{:cd}, k, l)
    A = cos(k/2) * cos(l/2)
    symbol = [0         0        -D(k)      0    1;
              0         0        -D(l)     -1    0;
              -gH*D(k) -gH*D(l)   0         0    0;
              0         1        -A*D(k)    0    0;
              -1        0        -A*D(l)    0    0]
    return inertia_gravity_frequency(symbol)
end

schemes = (:four_point => "Energy / enstrophy conserving", :cd => "C-D (no relaxation)", :oriented => "Oriented (this work)")

N = 128
wavenumbers = range(-π, π, length=N)
errors = Dict(scheme => [frequency(Val(scheme), k, l) / exact_frequency(k, l) - 1 for k in wavenumbers, l in wavenumbers]
              for (scheme, _) in schemes)

for (scheme, label) in schemes
    e = errors[scheme]
    @info @sprintf("%-32s worst %+.1f%%   (π,0) %+.1f%%   (π,π) %+.1f%%", label, 100 * minimum(e),
                   100 * (frequency(Val(scheme), π, 0) / exact_frequency(π, 0) - 1),
                   100 * (frequency(Val(scheme), π, π) / exact_frequency(π, π) - 1))
end

fig = Figure(size=(1100, 700), fontsize=16)
heatmaps = map(enumerate(schemes)) do (n, (scheme, label))
    ax = Axis(fig[1, n]; title=label, xlabel="kΔ", ylabel=n == 1 ? "lΔ" : "", aspect=DataAspect(),
              xticks=([-π, 0, π], ["-π", "0", "π"]), yticks=([-π, 0, π], ["-π", "0", "π"]))
    n > 1 && hideydecorations!(ax, ticks=false)
    heatmap!(ax, wavenumbers, wavenumbers, 100 .* errors[scheme]; colormap=:balance, colorrange=(-60, 60))
end
Colorbar(fig[1, length(schemes) + 1], last(heatmaps); label="ω / ω_exact − 1 [%]")

k = range(0, π, length=200)
paths = (("l = 0", k, zero.(k)), ("l = k", k, k), ("l = −k", k, .-k))
for (n, (title, ks, ls)) in enumerate(paths)
    ax = Axis(fig[2, n]; title, xlabel="kΔ", ylabel=n == 1 ? "ω / ω_exact − 1 [%]" : "",
              xticks=([0, π/2, π], ["0", "π/2", "π"]), limits=(0, π, -65, 25))
    for (scheme, label) in schemes
        lines!(ax, ks, [100 * (frequency(Val(scheme), a, b) / exact_frequency(a, b) - 1) for (a, b) in zip(ks, ls)]; label, linewidth=2)
    end
    n == 1 && Legend(fig[3, 1:4], ax; orientation=:horizontal, framevisible=false)
end

output = get(ENV, "FIGURE_DIRECTORY", ".")
save(joinpath(output, "dispersion.pdf"), fig)
