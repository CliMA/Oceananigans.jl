#
# Render the snapshots dumped by `immersed_hat_advection.jl` as a movie.
#
# The colour range is pinned to the initial range of the tracer, [0, 1], with a tolerance, so
# every coloured pixel is a new extremum that the reconstruction invented: magenta below zero,
# red above one, grey for the staircase. Cells that leave the range are also ringed, and counted
# in the panel title, so a handful of stray cells is as visible as a whole overshooting front.
#
# Under the panels, the variance lost -- the dissipation the reconstruction adds, since the flow
# is divergence free to roundoff and tangent to the wall -- and the running extrema.
#
# Usage
# =====
#
#   # once per build, from the validation/advection directory
#   julia --project=<env> immersed_hat_advection.jl boundary
#
#   # any number of panels, each `file[:scheme-pattern]`
#   julia --project=<env> hat_advection_movie.jl boundary.jld2:Centered boundary.jld2:Upwind boundary.jld2:CWENOZ
#   julia --project=<env> hat_advection_movie.jl main.jld2 cwenoz.jld2
#
# The scheme pattern is matched against the saved scheme names and defaults to the only scheme in
# the file. With exactly two panels a difference panel is added.
#

using JLD2
using CairoMakie
using Printf

# Only excursions bigger than this light up, so that roundoff-level negatives do not swamp the
# picture. Set to 0 to see every cell that leaves [0, 1].
const TOL = 0.02

isempty(ARGS) && error("give at least one .jld2 written by immersed_hat_advection.jl")

struct Run
    label :: String
    snapshots :: Vector{Matrix{Float64}}
    times :: Vector{Float64}
    variance_lost :: Vector{Float64}
    lo :: Vector{Float64}
    hi :: Vector{Float64}
end

parse_spec(spec) = (parts = split(spec, ':');
                    length(parts) == 1 ? (parts[1], nothing) : (parts[1], parts[2]))

function pick_scheme(names, pattern)
    isnothing(pattern) && length(names) == 1 && return only(names)
    isnothing(pattern) && error("$(length(names)) schemes in the file; name one of: " * join(names, ", "))
    pattern in names && return pattern
    hits = filter(n -> occursin(pattern, n), names)
    length(hits) == 1 && return only(hits)
    isempty(hits) && error("no scheme matching \"$pattern\"; available: " * join(names, ", "))
    error("\"$pattern\" is ambiguous: " * join(hits, ", "))
end

function load_run(spec)
    path, pattern = parse_spec(spec)
    jldopen(path, "r") do f
        names  = collect(keys(f["schemes"]))
        scheme = pick_scheme(names, pattern)
        g = "schemes/" * scheme
        return Run(scheme, f[g * "/snapshots"], f[g * "/times"],
                   f[g * "/snapshot_variance_lost"], f[g * "/snapshot_min"], f[g * "/snapshot_max"]),
               f["wet"], f["Lx"], f["Lz"]
    end
end

runs = Run[]
local wet, Lx, Lz
for (n, spec) in enumerate(ARGS)
    r, w, lx, lz = load_run(spec)
    push!(runs, r)
    n == 1 && (global wet = w; global Lx = lx; global Lz = lz)
end

Nx, Nz = size(wet)
x = range(0, Lx, length = Nx)
z = range(-Lz, 0, length = Nz)

mask(field) = [wet[i, k] ? field[i, k] : NaN for i in 1:Nx, k in 1:Nz]

# the cells that left [0, 1], split by side, as points to ring on the heatmap
function excursions(field)
    under, over = Point2f[], Point2f[]
    for i in 1:Nx, k in 1:Nz
        wet[i, k] || continue
        field[i, k] < -TOL    && push!(under, Point2f(x[i], z[k]))
        field[i, k] > 1 + TOL && push!(over,  Point2f(x[i], z[k]))
    end
    return under, over
end

nframes = minimum(length(r.snapshots) for r in runs)
show_difference = length(runs) == 2
npanels = length(runs) + (show_difference ? 1 : 0)
ncolumns = npanels + 1                      # panels plus the colourbar

palette = [:black, :firebrick, :royalblue, :seagreen, :darkorange]

fig = Figure(size = (430 * npanels + 190, 900), backgroundcolor = :white)

obs = [Observable(mask(r.snapshots[1])) for r in runs]
under_obs = [Observable(first(excursions(r.snapshots[1]))) for r in runs]
over_obs  = [Observable(last(excursions(r.snapshots[1])))  for r in runs]
titles = [Observable("") for _ in runs]

for (n, r) in enumerate(runs)
    ax = Axis(fig[2, n], aspect = DataAspect(), xlabel = "x", ylabel = n == 1 ? "z" : "",
              title = titles[n], titlesize = 13, titlecolor = palette[n])
    hm = heatmap!(ax, x, z, obs[n]; colorrange = (-TOL, 1 + TOL), colormap = :dense,
                  lowclip = :magenta, highclip = :red, nan_color = RGBAf(0.82, 0.82, 0.84, 1))
    scatter!(ax, under_obs[n]; marker = :circle, markersize = 11, color = :transparent,
             strokecolor = :magenta, strokewidth = 1.5)
    scatter!(ax, over_obs[n];  marker = :circle, markersize = 11, color = :transparent,
             strokecolor = :red, strokewidth = 1.5)
    n == length(runs) && !show_difference && Colorbar(fig[2, ncolumns], hm, label = "c")
end

if show_difference
    diff_at(n) = mask(runs[2].snapshots[n] .- runs[1].snapshots[n])
    dobs = Observable(diff_at(1))
    axd = Axis(fig[2, 3], aspect = DataAspect(), xlabel = "x",
               title = "$(runs[2].label) − $(runs[1].label)", titlesize = 13)
    dmax = maximum(maximum(v -> isnan(v) ? 0.0 : abs(v), diff_at(n)) for n in 1:nframes)
    hmd = heatmap!(axd, x, z, dobs; colorrange = (-dmax, dmax), colormap = :balance,
                   nan_color = RGBAf(0.82, 0.82, 0.84, 1))
    Colorbar(fig[2, ncolumns], hmd, label = "Δc")
end

#####
##### Dissipation and the extremum envelope, drawn up to the current frame
#####

t = runs[1].times[1:nframes]
now = Observable(t[1])

axv = Axis(fig[3, 1:max(1, ncolumns - 2)], xlabel = "t", ylabel = "variance lost (%)",
           title = "dissipation", titlesize = 13)
axe = Axis(fig[3, ncolumns-1:ncolumns], xlabel = "t", ylabel = "c",
           title = "running extrema", titlesize = 13)

for (n, r) in enumerate(runs)
    lines!(axv, t, 100 .* r.variance_lost[1:nframes]; color = palette[n], linewidth = 2, label = r.label)
    lines!(axe, t, r.hi[1:nframes]; color = palette[n], linewidth = 2, label = r.label)
    lines!(axe, t, r.lo[1:nframes]; color = palette[n], linewidth = 2)
end
hlines!(axe, [0, 1]; color = :gray50, linestyle = :dash)
vlines!(axv, now; color = :gray30, linestyle = :dot)
vlines!(axe, now; color = :gray30, linestyle = :dot)
axislegend(axv; position = :lt, labelsize = 10, framevisible = false)

title = Observable("t = 0.000")
Label(fig[1, 1:ncolumns], title, fontsize = 17, font = :bold)
Label(fig[4, 1:ncolumns],
      "colour pinned to the initial range: magenta is c < −$(TOL), red is c > 1 + $(TOL), ringed cells are counted in each title, grey is the staircase",
      fontsize = 11, color = :gray30)

rowsize!(fig.layout, 2, Relative(0.60))
rowsize!(fig.layout, 3, Relative(0.28))
colgap!(fig.layout, 14)
rowgap!(fig.layout, 8)

out = "hat_advection_" * join((replace(r.label, r"[^A-Za-z0-9]" => "") for r in runs), "_vs_") * ".mp4"
record(fig, out, 1:nframes; framerate = 24) do n
    for (m, r) in enumerate(runs)
        field = r.snapshots[n]
        obs[m][] = mask(field)
        under, over = excursions(field)
        under_obs[m][] = under
        over_obs[m][]  = over
        titles[m][] = @sprintf("%s\nmin %.4f   max %.4f   var lost %.3f%%\n%d under / %d over",
                               r.label, r.lo[n], r.hi[n], 100r.variance_lost[n],
                               length(under), length(over))
    end
    show_difference && (dobs[] = mask(runs[2].snapshots[n] .- runs[1].snapshots[n]))
    now[] = r_time = runs[1].times[n]
    title[] = @sprintf("immersed hat advection along a staircase      t = %.3f", r_time)
end

still = replace(out, ".mp4" => "_final.png")
save(still, fig)

println("wrote $out  ($nframes frames)  and $still")
