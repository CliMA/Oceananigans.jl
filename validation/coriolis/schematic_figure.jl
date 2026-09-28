# Schematic of the C-grid null mode and of the oriented coupling, with the symbol |m| of the potential-vorticity
# interpolation for the four-point average and for the oriented scheme (ε = 1/8, ϵc = 1/32).

using CairoMakie

const ε = 1/8
const ϵc = 1/32

corner_weights = (SW = ((0, 0), ϵc), SE = ((1, 0), ε - ϵc), NW = ((0, 1), -ε - ϵc), NE = ((1, 1), ϵc))

four_point_m(K, L) = cos(K/2) * cos(L/2)

function oriented_m(K, L)
    σ² = 4sin(K/2)^2 + 4sin(L/2)^2
    return cos(K/2) * cos(L/2) + 4ϵc * σ² * sin(K/2) * sin(L/2) + 2im * ε * σ² * sin((K - L)/2)
end

fig = Figure(size=(1000, 900), fontsize=16)

ax = Axis(fig[1, 1]; title="(a) Null mode v = (−1)ⁱ of the four-point average", aspect=DataAspect(),
          limits=(-0.3, 4.3, -0.8, 3.4))
hidedecorations!(ax); hidespines!(ax)
for x in 0:4
    lines!(ax, [x, x], [0, 3]; color=:gray70)
end
for y in 0:3
    lines!(ax, [0, 4], [y, y]; color=:gray70)
end
scatter!(ax, [i + 0.5 for i in 0:3 for j in 0:2], [j + 0.5 for i in 0:3 for j in 0:2]; color=:gray50, markersize=7)
scatter!(ax, [x for x in 0:4 for y in 0:3], [y for x in 0:4 for y in 0:3]; marker=:xcross, color=:gray60, markersize=8)
for i in 0:3, j in 0:3
    s = iseven(i) ? 1 : -1
    arrows!(ax, [i + 0.5], [j - 0.22s], [0], [0.44s]; color=iseven(i) ? :crimson : :royalblue, linewidth=2.5, arrowsize=12)
end
for i in 0:4, j in 0:2
    scatter!(ax, [i], [j + 0.5]; marker=:rect, color=:white, strokecolor=:black, strokewidth=1.2, markersize=9)
end
poly!(ax, Circle(Point2f(2, 1.5), 0.62); color=(:gold, 0.25), strokecolor=:goldenrod, strokewidth=2)
text!(ax, 2, -0.4; text="the four v around each u point sum to zero", align=(:center, :top), fontsize=14)

ax = Axis(fig[1, 2]; title="(b) Oriented coupling of a cell to its corners", aspect=DataAspect(), limits=(-0.6, 1.6, -0.6, 1.6))
hidedecorations!(ax); hidespines!(ax)
lines!(ax, [0, 1, 1, 0, 0], [0, 0, 1, 1, 0]; color=:black, linewidth=1.5)
lines!(ax, [1.25, -0.25], [-0.25, 1.25]; color=:gray40, linestyle=:dash)
scatter!(ax, [0.5], [0.5]; color=:black, markersize=14)
text!(ax, 0.5, 0.42; text="c", align=(:center, :top), fontsize=16)
labels = Dict(:SW => "SW  1/32", :SE => "SE  3/32", :NW => "NW  −5/32", :NE => "NE  1/32")
for (name, ((x, y), w)) in pairs(corner_weights)
    scatter!(ax, [x], [y]; color=w, colormap=:balance, colorrange=(-5/32, 5/32), markersize=18 + 120abs(w),
             strokecolor=:black, strokewidth=1)
    text!(ax, x, y + (y == 0 ? -0.2 : 0.2); text=labels[name], align=(:center, y == 0 ? :top : :bottom), fontsize=14)
    arrows!(ax, [x + 0.18 * (0.5 - x) * 2], [y + 0.18 * (0.5 - y) * 2], [0.22 * (0.5 - x) * 2], [0.22 * (0.5 - y) * 2];
            color=:gray30, arrowsize=10)
end
text!(ax, 0.5, -0.55; text="A = Σ α f Γ at the centre,   B = −Σ α f D at the corners", align=(:center, :bottom), fontsize=14)

K = range(-π, π, length=241)
for (column, (title, m)) in enumerate((("(c) |m|, four-point average", four_point_m), ("(d) |m|, oriented scheme", oriented_m)))
    local ax = Axis(fig[2, column]; title, xlabel="K", ylabel="L", aspect=DataAspect(),
              xticks=([-π, 0, π], ["−π", "0", "π"]), yticks=([-π, 0, π], ["−π", "0", "π"]))
    hm = heatmap!(ax, K, K, [abs(m(k, l)) for k in K, l in K]; colormap=:viridis, colorrange=(0, 1.4))
    column == 2 && Colorbar(fig[2, 3], hm; label="|m|")
end

output = get(ENV, "FIGURE_DIRECTORY", ".")
save(joinpath(output, "schematic.pdf"), fig)
save(joinpath(get(ENV, "PREVIEW_DIRECTORY", output), "schematic.png"), fig)
