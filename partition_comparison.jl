# Balanced partitions of a 1ᵒ global ocean: cartesian, jagged P × Q, and general rectangles with land excluded.
# The cost of a column is its number of wet cells, as in `create_cost_map`.

using Oceananigans
using Oceananigans.DistributedComputations: create_cost_map
using NCDatasets
using CairoMakie

#####
##### 1ᵒ bathymetry from ETOPO 2022, averaged over each grid cell
#####

etopo_path = "/Users/simonesilvestri/development/drakkar2025-demo-oceananigans/Downsampled_ETOPO_2022.nc"

Nx, Ny, Nz = 360, 160, 50
longitude = (0, 360)
latitude = (-80, 80)
z_faces = [-5000 * (1 - k / Nz)^2 for k in 0:Nz]

ds = NCDataset(etopo_path)
etopo_longitude = mod.(Float64.(ds["lon"][:]), 360)
etopo_latitude = Float64.(ds["lat"][:])
etopo_elevation = Float64.(ds["z"][:, :])
close(ds)

elevation_sum = zeros(Nx, Ny)
elevation_count = zeros(Nx, Ny)

for (n, φ) in enumerate(etopo_latitude), (m, λ) in enumerate(etopo_longitude)
    j = floor(Int, φ - latitude[1]) + 1
    1 ≤ j ≤ Ny || continue
    i = clamp(floor(Int, λ - longitude[1]) + 1, 1, Nx)
    elevation_sum[i, j] += etopo_elevation[m, n]
    elevation_count[i, j] += 1
end

bottom_height = min.(elevation_sum ./ elevation_count, 0)
bottom_height[bottom_height .> -10] .= 0

grid = LatitudeLongitudeGrid(size = (Nx, Ny, Nz); longitude, latitude, z = z_faces)
cost = Int.(create_cost_map(grid, GridFittedBottom(bottom_height))[:, :, 1])
ocean = cost .> 0
mean_cost(R) = sum(cost) / R

#####
##### Partitioning algorithms
#####

load(box) = sum(view(cost, box[1], box[2]))

exceeds(part_load, costs, n, B) = any(b -> part_load[b] + costs[b, n] > B, axes(costs, 1))

function count_parts(costs, B)
    parts = 1
    part_load = zeros(Int, size(costs, 1))
    for n in axes(costs, 2)
        if exceeds(part_load, costs, n, B)
            parts += 1
            part_load .= 0
        end
        part_load .+= view(costs, :, n)
    end
    return parts
end

# Split 1:N into R contiguous nonempty ranges minimizing the largest part, where the load of a part is the largest of
# its row sums of `costs` (one row per block of ranks in the other direction): exact, by bisection on that load.
function balanced_ranges(costs::AbstractMatrix, R)
    N = size(costs, 2)
    lower, upper = maximum(costs), maximum(sum(costs, dims=2))
    while lower < upper
        middle = (lower + upper) ÷ 2
        count_parts(costs, middle) ≤ R ? (upper = middle) : (lower = middle + 1)
    end

    ranges = UnitRange{Int}[]
    start = 1
    part_load = zeros(Int, size(costs, 1))
    for n in 1:N
        open_parts = R - length(ranges)
        if n > start && open_parts > 1 && (exceeds(part_load, costs, n, lower) || N - n + 1 < open_parts)
            push!(ranges, start:n-1)
            start = n
            part_load .= 0
        end
        part_load .+= view(costs, :, n)
    end
    push!(ranges, start:N)
    return ranges
end

balanced_ranges(costs::AbstractVector, R) = balanced_ranges(reshape(costs, 1, :), R)

# Manne & Sørevik (1996): alternate the x and y cuts, each optimal given the other
function cartesian_boxes(P, Q)
    x = balanced_ranges(vec(sum(cost, dims=2)), P)
    y = balanced_ranges(vec(sum(cost, dims=1)), Q)
    boxes = [(i, j) for i in x, j in y]
    for iteration in 1:20
        y = balanced_ranges([sum(view(cost, i, j)) for i in x, j in 1:Ny], Q)
        x = balanced_ranges([sum(view(cost, i, j)) for j in y, i in 1:Nx], P)
        new_boxes = [(i, j) for i in x, j in y]
        maximum(load, new_boxes) < maximum(load, boxes) || break
        boxes = new_boxes
    end
    return vec(boxes)
end

# x first, then every column of ranks balanced in y independently
function jagged_boxes(P, Q)
    columns = balanced_ranges(vec(sum(cost, dims=2)), P)
    return [(i, j) for i in columns for j in balanced_ranges(vec(sum(view(cost, i, :), dims=1)), Q)]
end

function ocean_extent(I, J)
    wet_columns = findall(>(0), vec(sum(view(cost, I, J), dims=2)))
    wet_rows = findall(>(0), vec(sum(view(cost, I, J), dims=1)))
    return I[first(wet_columns):last(wet_columns)], J[first(wet_rows):last(wet_rows)]
end

# Weighted recursive bisection on general rectangles; every box shrinks to the extent of its own ocean
function bisection_boxes(I, J, R)
    I, J = ocean_extent(I, J)
    R == 1 && return [(I, J)]
    R₁ = R ÷ 2
    R₂ = R - R₁
    best_load = Inf
    best_halves = ((I, J), (I, J))
    for (along_x, line) in ((true, vec(sum(view(cost, I, J), dims=2))), (false, vec(sum(view(cost, I, J), dims=1))))
        cumulative = cumsum(line)
        for c in 1:length(line)-1
            halves_load = max(cumulative[c] / R₁, (cumulative[end] - cumulative[c]) / R₂)
            if halves_load < best_load
                best_load = halves_load
                best_halves = along_x ? ((I[1:c], J), (I[c+1:end], J)) : ((I, J[1:c]), (I, J[c+1:end]))
            end
        end
    end
    (I₁, J₁), (I₂, J₂) = best_halves
    return vcat(bisection_boxes(I₁, J₁, R₁), bisection_boxes(I₂, J₂, R₂))
end

general_boxes(R) = bisection_boxes(1:Nx, 1:Ny, R)

#####
##### Diagnostics
#####

function check_partition(boxes; tiling)
    coverage = zeros(Int, Nx, Ny)
    for (i, j) in boxes
        coverage[i, j] .+= 1
    end
    @assert all(≤(1), coverage) "boxes overlap"
    @assert all(coverage[ocean] .== 1) "ocean left without a rank"
    tiling && @assert all(==(1), coverage) "the partition does not tile the domain"
    return nothing
end

# Ranks whose box meets the halo of width H around a box (periodic in x)
function neighbor_counts(boxes; H = 4)
    return map(boxes) do (I, J)
        halo_I = first(I)-H:last(I)+H
        halo_J = first(J)-H:last(J)+H
        count(boxes) do (I′, J′)
            (I′, J′) == (I, J) && return false
            meets_y = !isempty(intersect(halo_J, J′))
            meets_x = any(shift -> !isempty(intersect(halo_I, I′ .+ shift)), (-Nx, 0, Nx))
            meets_x && meets_y
        end
    end
end

allocated_fraction(boxes) = sum(length(i) * length(j) for (i, j) in boxes) / (Nx * Ny)
@info "Ocean fraction of the grid" count(ocean) / (Nx * Ny)

#####
##### The 64-rank partitions
#####

R = 64
partitions = (cartesian = cartesian_boxes(8, 8), jagged = jagged_boxes(8, 8), general = general_boxes(R))

check_partition(partitions.cartesian; tiling = true)
check_partition(partitions.jagged; tiling = true)
check_partition(partitions.general; tiling = false)

relative_loads(boxes) = [load(box) / mean_cost(length(boxes)) for box in boxes]

summaries = map(partitions) do boxes
    loads = relative_loads(boxes)
    (maximum_load = maximum(loads), land_only = count(iszero, loads),
     neighbors = maximum(neighbor_counts(boxes)), allocated = allocated_fraction(boxes))
end

foreach(name -> @info(name, summaries[name]...), keys(summaries))

#####
##### Best achievable time per step against the number of GPUs
#####

gpus = 16:160
time_per_step(boxes) = maximum(load, boxes) / mean_cost(R) # relative to a perfectly balanced 64-GPU run

cartesian_points = Tuple{Int, Float64}[]
cartesian_without_land_points = Tuple{Int, Float64}[]
jagged_points = Tuple{Int, Float64}[]

for P in 1:20, Q in 1:20
    P * Q ≤ last(gpus) || continue
    boxes = cartesian_boxes(P, Q)
    t = time_per_step(boxes)
    push!(cartesian_points, (P * Q, t))
    push!(cartesian_without_land_points, (P * Q - count(box -> load(box) == 0, boxes), t))
    push!(jagged_points, (P * Q, time_per_step(jagged_boxes(P, Q))))
end

general_points = [(n, time_per_step(general_boxes(n))) for n in gpus]

best_with_at_most(points) = [minimum((t for (n′, t) in points if n′ ≤ n); init = Inf) for n in gpus]

curves = (cartesian = best_with_at_most(cartesian_points),
          cartesian_without_land = best_with_at_most(cartesian_without_land_points),
          jagged = best_with_at_most(jagged_points),
          general = best_with_at_most(general_points),
          perfect = R ./ gpus)

gpus_needed(curve) = gpus[findfirst(t -> t ≤ 1.1, curve)]
foreach(name -> @info("GPUs within 10% of a perfectly balanced 64-GPU run", name, gpus_needed(curves[name])), keys(curves))

#####
##### Figure
#####

λ = range(longitude[1] + 0.5, longitude[2] - 0.5, length=Nx)
φ = range(latitude[1] + 0.5, latitude[2] - 0.5, length=Ny)

diverging = cgrad(["#184f95", "#6da7ec", "#f0efec", "#ec8a89", "#b8302f"])
allocated_land_color = "#9a9893"
free_land_color = "#d9d7d2"
boundary_color = "#1a1a19"
load_range = (0.5, 1.5)

function draw_partition!(ax, boxes)
    loads = relative_loads(boxes)
    load_map = fill(NaN, Nx, Ny)
    allocated = falses(Nx, Ny)
    for ((i, j), l) in zip(boxes, loads)
        load_map[i, j] .= l
        allocated[i, j] .= true
    end
    load_map[.!ocean] .= NaN
    land_map = [ocean[n] ? NaN : (allocated[n] ? 1.0 : 0.0) for n in CartesianIndices(ocean)]

    heatmap!(ax, λ, φ, load_map; colormap = diverging, colorrange = load_range, lowclip = "#184f95", highclip = "#b8302f")
    heatmap!(ax, λ, φ, land_map; colormap = [free_land_color, allocated_land_color], colorrange = (0, 1))

    for ((i, j), l) in zip(boxes, loads)
        λ₁, λ₂ = longitude[1] + first(i) - 1, longitude[1] + last(i)
        φ₁, φ₂ = latitude[1] + first(j) - 1, latitude[1] + last(j)
        lines!(ax, [λ₁, λ₂, λ₂, λ₁, λ₁], [φ₁, φ₁, φ₂, φ₂, φ₁]; color = boundary_color, linewidth = 1)
        l == 0 && linesegments!(ax, [λ₁, λ₂, λ₁, λ₂], [φ₁, φ₂, φ₂, φ₁]; color = boundary_color, linewidth = 1.5)
    end
    return nothing
end

percent(x) = string(round(Int, 100x), "%")

function subtitle(s)
    land_only = s.land_only == 0 ? "no land-only ranks" : "$(s.land_only) land-only ranks (crossed)"
    return "most loaded rank $(round(s.maximum_load, digits=2)) × mean · $land_only · " *
           "up to $(s.neighbors) neighbors · $(percent(s.allocated)) of the globe allocated"
end

fig = Figure(size = (2000, 980), fontsize = 16)

axis_options = (xlabel = "Longitude (ᵒ)", ylabel = "Latitude (ᵒ)", aspect = DataAspect(),
                limits = (longitude, latitude), xticks = 0:60:360, yticks = -80:40:80, subtitlesize = 15)

ax_cartesian = Axis(fig[1, 1]; title = "Cartesian, 8 × 8", subtitle = subtitle(summaries.cartesian), axis_options...)
ax_jagged = Axis(fig[1, 2]; title = "Jagged P × Q: 8 columns, each with its own 8 latitude cuts",
                 subtitle = subtitle(summaries.jagged), axis_options...)
ax_general = Axis(fig[2, 1]; title = "General rectangles, land excluded (weighted recursive bisection, 64 boxes)",
                  subtitle = subtitle(summaries.general), axis_options...)

draw_partition!(ax_cartesian, partitions.cartesian)
draw_partition!(ax_jagged, partitions.jagged)
draw_partition!(ax_general, partitions.general)

ax_gpus = Axis(fig[2, 2]; title = "Best time per step for a given number of GPUs",
               subtitle = "compute-bound estimate: most loaded rank, relative to a perfectly balanced 64-GPU run",
               subtitlesize = 15, xlabel = "GPUs", ylabel = "Time per step", xticks = 16:16:160, aspect = AxisAspect(Nx / Ny),
               yticks = 0.5:0.5:3, limits = ((first(gpus), last(gpus)), (0.25, 3)))

orange, blue, aqua = "#eb6834", "#2a78d6", "#1baf7a"

lines!(ax_gpus, gpus, curves.perfect; color = "#8c8b87", linewidth = 1.5, linestyle = :dash, label = "perfect balance")
lines!(ax_gpus, gpus, curves.cartesian; color = orange, linewidth = 2, label = "cartesian")
lines!(ax_gpus, gpus, curves.cartesian_without_land; color = orange, linewidth = 2, linestyle = :dot,
       label = "cartesian, land-only ranks removed")
lines!(ax_gpus, gpus, curves.jagged; color = blue, linewidth = 2, label = "jagged P × Q")
lines!(ax_gpus, gpus, curves.general; color = aqua, linewidth = 2, label = "general rectangles")
hlines!(ax_gpus, 1.1; color = "#c3c2be", linewidth = 1)
axislegend(ax_gpus; position = :rt, framevisible = false)

Colorbar(fig[1, 3]; colormap = diverging, limits = load_range, highclip = "#b8302f", lowclip = "#184f95",
         label = "Rank load / mean load (wet cells)")

Legend(fig[2, 3], [PolyElement(color = allocated_land_color), PolyElement(color = free_land_color)],
       ["land inside a rank's box\n(allocated, not computed)", "land outside every box\n(not allocated)"];
       framevisible = false, valign = :top)

save(joinpath(@__DIR__, "partition_comparison.png"), fig, px_per_unit = 2)
