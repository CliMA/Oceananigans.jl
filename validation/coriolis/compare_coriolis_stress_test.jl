# Compare Coriolis-only stress test results: Oceananigans vs MITgcm
#
# Both models run with NO momentum advection so the only active tendency
# is the Coriolis term (f × u).  This isolates the effect of the Coriolis
# discretisation near immersed boundaries.
#
# Prerequisites: run both stress test scripts first:
#   include("coriolis_immersed_stress_test.jl")
#   include("mitgcm_coriolis_stress_test.jl")

using Oceananigans
using Oceananigans.OutputReaders: FieldTimeSeries
using JLD2
using CairoMakie
using Printf

dir = @__DIR__

#####
##### Load MITgcm results
#####

mitgcm_file = joinpath(dir, "mitgcm_coriolis_results.jld2")
if !isfile(mitgcm_file)
    error("MITgcm results not found. Run mitgcm_coriolis_stress_test.jl first.")
end

mitgcm = load(mitgcm_file)
mitgcm_results = mitgcm["results"]
Nx = mitgcm["Nx"]
Ny = mitgcm["Ny"]
Δx = mitgcm["Δx"]
Δy = mitgcm["Δy"]

λ_mitgcm = [(i - 0.5) * Δx for i in 1:Nx]
φ_mitgcm = [30.0 + (j - 0.5) * Δy for j in 1:Ny]

#####
##### Load Oceananigans results
#####

ocean_dir = joinpath(dir, "coriolis_stress_test_output")
if !isdir(ocean_dir)
    error("Oceananigans results not found. Run coriolis_immersed_stress_test.jl first.")
end

ocean_labels = ["ES", "EN", "EEN", "AWES", "AWEN"]
ocean_ts = Dict{String, NamedTuple}()

for label in ocean_labels
    fp = joinpath(ocean_dir, "stress_test_$(label).jld2")
    isfile(fp) || continue
    ocean_ts[label] = (η = FieldTimeSeries(fp, "η"),
                       u = FieldTimeSeries(fp, "u"),
                       v = FieldTimeSeries(fp, "v"))
end

# Grid coordinates
first_ts = ocean_ts[first(keys(ocean_ts))]
ocean_grid = first_ts.η.grid
λc_ocean = λnodes(ocean_grid, Center())
φc_ocean = φnodes(ocean_grid, Center())

#####
##### Helpers
#####

function get_ocean_final(d, name)
    Nt = length(d.η.times)
    fts = getfield(d, Symbol(name))
    return interior(fts[Nt], :, :, 1)
end

function get_mitgcm_final(r, name)
    name == "η" && return r.etan
    name == "u" && return r.uvel[:, :, 1]
    name == "v" && return r.vvel[:, :, 1]
end

#####
##### Figure: 4 columns x 6 rows
#####
#
#  Columns: 4 schemes
#  Rows 1-3: Oceananigans (η, u, v)    — title on row 1
#  Rows 4-6: MITgcm       (η, u, v)    — title on row 4
#  Gap between row 3 and 4
#
#  OC columns (no EEN): ES, EN, AWES, AWEN
#  MG columns: Simple(0), Jamart(1), hFac(2), EnCons(3)

oc_schemes = [
    ("ES",   "Oceananigans: ES"),
    ("EN",   "Oceananigans: EN"),
    ("AWES", "Oceananigans: AWES"),
    ("AWEN", "Oceananigans: AWEN"),
]

mg_schemes = [
    ("Simple", "MITgcm: selectCoriScheme=0"),
    ("Jamart", "MITgcm: selectCoriScheme=1"),
    ("hFac",   "MITgcm: selectCoriScheme=2"),
    ("EnCons", "MITgcm: selectCoriScheme=3"),
]

var_names = ["u", "v"]
var_labels = ["u [m/s]", "v [m/s]"]

cmaps = Dict("u" => :balance, "v" => :balance)

Nc = length(oc_schemes)  # 4 columns

# Compute color limits across ALL data (OC + MG) per variable
clims = Dict{String, Float64}()
for vname in var_names
    cmax = 1e-10
    for (key, _) in oc_schemes
        haskey(ocean_ts, key) || continue
        vals = filter(isfinite, abs.(get_ocean_final(ocean_ts[key], vname)[:]))
        isempty(vals) || (cmax = max(cmax, maximum(vals)))
    end
    for (key, _) in mg_schemes
        haskey(mitgcm_results, key) || continue
        vals = filter(isfinite, abs.(get_mitgcm_final(mitgcm_results[key], vname)[:]))
        isempty(vals) || (cmax = max(cmax, maximum(vals)))
    end
    clims[vname] = cmax * 0.5
end

@info "Plotting comparison figure..."

fig = Figure(size = (350 * Nc + 60, 1100), fontsize=11)

last_hm = Dict{String, Any}()

# --- Rows 1-2: Oceananigans ---
for (col, (key, title)) in enumerate(oc_schemes)
    haskey(ocean_ts, key) || continue
    d = ocean_ts[key]

    for (vi, vname) in enumerate(var_names)
        row = vi
        clim = clims[vname]
        data = get_ocean_final(d, vname)

        show_title = vi == 1
        show_xlabel = false  # never on OC rows — gap below

        ax = Axis(fig[row, col];
                  title  = show_title ? title : "",
                  titlesize = show_title ? 14 : 11,
                  ylabel = col == 1 ? var_labels[vi] : "",
                  xticklabelsvisible = false,
                  yticklabelsvisible = col == 1)

        hm = heatmap!(ax, λc_ocean, φc_ocean, data;
                       colormap=cmaps[vname], colorrange=(-clim, clim))
        last_hm[vname] = hm
    end
end

# --- Rows 3-4: MITgcm ---
Nv = length(var_names)  # 2 (u, v)

for (col, (key, title)) in enumerate(mg_schemes)
    haskey(mitgcm_results, key) || continue
    r = mitgcm_results[key]

    for (vi, vname) in enumerate(var_names)
        row = vi + Nv
        clim = clims[vname]
        data = get_mitgcm_final(r, vname)

        show_title = vi == 1
        show_xlabel = vi == Nv

        ax = Axis(fig[row, col];
                  title  = show_title ? title : "",
                  titlesize = show_title ? 14 : 11,
                  ylabel = col == 1 ? var_labels[vi] : "",
                  xlabel = show_xlabel ? "λ [°]" : "",
                  xticklabelsvisible = show_xlabel,
                  yticklabelsvisible = col == 1)

        hm = heatmap!(ax, λ_mitgcm, φ_mitgcm, data;
                       colormap=cmaps[vname], colorrange=(-clim, clim))
        last_hm[vname] = hm
    end
end

# Colorbars — one per variable, spanning both OC and MG rows
for (vi, vname) in enumerate(var_names)
    Colorbar(fig[vi, Nc + 1]; label=var_labels[vi],
             colormap=cmaps[vname], colorrange=(-clims[vname], clims[vname]), width=12)
    Colorbar(fig[vi + Nv, Nc + 1]; label=var_labels[vi],
             colormap=cmaps[vname], colorrange=(-clims[vname], clims[vname]), width=12)
end

# Overarching title
Label(fig[0, 1:Nc], "Coriolis-only stress test (no momentum advection, 30 days)";
      fontsize=18, font=:bold)

# Gap between OC (row Nv) and MG (row Nv+1)
rowgap!(fig.layout, Nv, 40)

save("compare_coriolis_final.png", fig, px_per_unit=2)
@info "Saved compare_coriolis_final.png"
