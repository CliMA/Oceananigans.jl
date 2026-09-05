#
# Render the snapshots dumped by `immersed_hat_advection.jl` as a movie.
#
# The colour range is pinned to the initial range of the tracer, [0, 1], with a tolerance, so
# every coloured pixel is a new extremum that the reconstruction invented: magenta below zero,
# red above one, grey for the staircase.
#
# Usage
# =====
#
#   # once per build, from the validation/advection directory
#   julia --project=<env> immersed_hat_advection.jl main      # cwenoz_boundary.jl disabled
#   julia --project=<env> immersed_hat_advection.jl cwenoz    # cwenoz_boundary.jl enabled
#
#   # any number of panels, each `file[:scheme-pattern]`
#   julia --project=<env> hat_advection_movie.jl main.jld2 cwenoz.jld2
#   julia --project=<env> hat_advection_movie.jl main.jld2 cwenoz.jld2 main.jld2:min_buffer
#
# The scheme pattern is matched against the saved scheme names and defaults to the plain
# seventh-order scheme. With exactly two panels a difference panel is added.
#

using JLD2
using CairoMakie
using Printf

# Only excursions bigger than this light up, so that roundoff-level negatives do not swamp the
# picture. Set to 0 to see every cell that leaves [0, 1].
const TOL = 0.02
const DEFAULT_SCHEME = "WENO(order=7)"

isempty(ARGS) && error("give at least one .jld2 written by immersed_hat_advection.jl")

struct Run
    label :: String
    snapshots :: Vector{Matrix{Float64}}
    times :: Vector{Float64}
    min :: Float64
    max :: Float64
    variance_lost :: Float64
end

parse_spec(spec) = (parts = split(spec, ':');
                    length(parts) == 1 ? (parts[1], DEFAULT_SCHEME) : (parts[1], parts[2]))

function pick_scheme(names, pattern)
    pattern in names && return pattern
    hits = filter(n -> occursin(pattern, n), names)
    length(hits) == 1 && return only(hits)
    isempty(hits) && error("no scheme matching \"$pattern\"; available: " * join(names, ", "))
    error("\"$pattern\" is ambiguous: " * join(hits, ", "))
end

short(name) = replace(name, "WENO(order=7)" => "W7", "WENO(order=5)" => "W5")

function load_run(spec)
    path, pattern = parse_spec(spec)
    jldopen(path, "r") do f
        names  = collect(keys(f["schemes"]))
        scheme = pick_scheme(names, pattern)
        stem   = splitext(basename(path))[1]
        label  = scheme == DEFAULT_SCHEME ? stem : stem * " / " * short(scheme)
        g = "schemes/" * scheme
        return Run(label, f[g * "/snapshots"], f[g * "/times"],
                   f[g * "/min"], f[g * "/max"], f[g * "/variance_lost"]),
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
noutside(field) = count(v -> !isnan(v) && (v < -TOL || v > 1 + TOL), mask(field))

nframes = minimum(length(r.snapshots) for r in runs)
show_difference = length(runs) == 2
npanels = length(runs) + (show_difference ? 1 : 0)
ncolumns = npanels + 1                      # panels plus the colourbar

fig = Figure(size = (400 * npanels + 110, 540), backgroundcolor = :white)

obs = [Observable(mask(r.snapshots[1])) for r in runs]
counts = [Observable("") for _ in runs]

for (n, r) in enumerate(runs)
    ax = Axis(fig[2, n], aspect = DataAspect(), xlabel = "x", ylabel = n == 1 ? "z" : "",
              title = @sprintf("%s\nmin %.4f   max %.4f   var lost %.2f%%",
                               r.label, r.min, r.max, 100r.variance_lost),
              titlesize = 13)
    hm = heatmap!(ax, x, z, obs[n]; colorrange = (-TOL, 1 + TOL), colormap = :dense,
                  lowclip = :magenta, highclip = :red, nan_color = RGBAf(0.82, 0.82, 0.84, 1))
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

title = Observable("t = 0.000")
Label(fig[1, 1:ncolumns], title, fontsize = 16, font = :bold)
Label(fig[3, 1:ncolumns],
      "colour range pinned to the initial [0, 1] — magenta is c < −$(TOL), red is c > 1+$(TOL), grey is the staircase",
      fontsize = 12, color = :gray30)

for n in 1:npanels
    colsize!(fig.layout, n, Aspect(2, 1.0))
end
colgap!(fig.layout, 10)
rowgap!(fig.layout, 5)

out = "hat_advection_" * join((replace(r.label, r"[^A-Za-z0-9]" => "") for r in runs), "_vs_") * ".mp4"
record(fig, out, 1:nframes; framerate = 24) do n
    for (m, r) in enumerate(runs)
        obs[m][] = mask(r.snapshots[n])
        counts[m][] = string(noutside(r.snapshots[n]))
    end
    show_difference && (dobs[] = mask(runs[2].snapshots[n] .- runs[1].snapshots[n]))
    title[] = @sprintf("t = %.3f      outside [0,1] by more than %.2f:   %s",
                       runs[1].times[n], TOL,
                       join((r.label * " " * counts[m][] for (m, r) in enumerate(runs)), "   |   "))
end

println("wrote $out  ($nframes frames)")
