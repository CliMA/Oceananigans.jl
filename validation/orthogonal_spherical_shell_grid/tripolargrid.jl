using Oceananigans
using Oceananigans.OrthogonalSphericalShellGrids: TripolarGrid
using Printf
using GLMakie

# tg = TripolarGrid(;
#     size = (12, 12, 2),
#     z = (-1000, 0),
#     southernmost_latitude = -70,
#     north_poles_latitude = 75,
#     first_pole_longitude = -180,
# )

# (; Nx, Ny, Nz, Hx, Hy, Hz) = tg
# # paint halos with NaNs for visualizing the grid
# function NaN_halo!(array, Nx, Ny, Hx, Hy; NEoffset = 1)
#     array[(1 - Hx):0, :] .= NaN
#     array[:, (1 - Hy):0] .= NaN
#     array[(Nx + NEoffset + 1):(Nx + Hx), :] .= NaN
#     array[:, (Ny + NEoffset + 1):(Ny + Hy)] .= NaN
#     return array
# end
# function no_lon_jumps(lon, Nx, Ny)
#     out = deepcopy(lon)
#     for j in 2:(Ny + 1)
#         for i in 2:(Nx + 1)
#             if (lon[i, j] < lon[i - 1, j] - 180) || (lon[i, j] < lon[i, j - 1] - 180)
#                 out[i, j] += 360
#             elseif (lon[i, j] > lon[i - 1, j] + 180) || (lon[i, j] > lon[i, j - 1] + 180)
#                 out[i, j] -= 360
#             end
#         end
#     end
#     return out
# end
# function no_lon_jumps_in_x(lon, Nx, Ny)
#     out = deepcopy(lon)
#     for j in 2:(Ny + 1)
#         for i in 2:(Nx + 1)
#             if lon[i, j] < lon[i - 1, j] - 180
#                 out[i, j] += 360
#             elseif lon[i, j] > lon[i - 1, j] + 180
#                 out[i, j] -= 360
#             end
#         end
#     end
#     return out
# end
# function no_lon_jumps_in_y(lon, Nx, Ny)
#     out = deepcopy(lon)
#     for j in 2:(Ny + 1)
#         for i in 2:(Nx + 1)
#             if lon[i, j] < lon[i, j - 1] - 180
#                 out[i, j] += 360
#             elseif lon[i, j] > lon[i, j - 1] + 180
#                 out[i, j] -= 360
#             end
#         end
#     end
#     return out
# end
# elat = NaN_halo!(tg.φᶠᶠᵃ, Nx, Ny, Hx, Hy; NEoffset = 1)
# elonx = no_lon_jumps_in_x(NaN_halo!(tg.λᶠᶠᵃ, Nx, Ny, Hx, Hy; NEoffset = 1), Nx, Ny)
# elony = no_lon_jumps_in_y(NaN_halo!(tg.λᶠᶠᵃ, Nx, Ny, Hx, Hy; NEoffset = 1), Nx, Ny)
# elon = no_lon_jumps(NaN_halo!(tg.λᶠᶠᵃ, Nx, Ny, Hx, Hy; NEoffset = 1), Nx, Ny)
# lat = NaN_halo!(tg.φᶜᶜᵃ, Nx, Ny, Hx, Hy; NEoffset = 0)
# lon = no_lon_jumps_in_x(NaN_halo!(tg.λᶜᶜᵃ, Nx, Ny, Hx, Hy; NEoffset = 0), Nx, Ny)

# fig = Figure(size = (1600, 800))
# ax = Axis(
#     fig[1, 1];
#     limits = ((-60, 420), (-91, 91)),
#     aspect = DataAspect(),
#     xticks = 0:30:360,
#     yticks = -90:30:90,
# )
# lines!(ax, elonx.parent[:], elat.parent[:])
# lines!(ax, elony.parent'[:], elat.parent'[:])
# scatter!(ax, lon.parent[:], lat.parent[:], color = :red)
# fig


function plot_u!(ax, u; k = size(u, 3), kwargs...)
    Nx, Ny, Nz = size(u)
    Hx, Hy, Hz = .-u.data.offsets
    i = (1 - Hx):(Nx + Hx)
    j = (1 - Hy):(Ny + Hy)
    us = [u[i, j, k] for i in i for j in j]
    vs = [0 for i in i for j in j]
    xs = [i - 0.5 for i in i for j in j]
    ys = [j for i in i for j in j]
    return arrows2d!(ax, xs, ys, us, vs; lengthscale = 1 / 2maximum(u), kwargs...)
end
function plot_v!(ax, v; k = size(v, 3), kwargs...)
    Nx, Ny, Nz = size(v)
    Hx, Hy, Hz = .-v.data.offsets
    i = (1 - Hx):(Nx + Hx)
    j = (1 - Hy):(Ny + Hy)
    us = [0 for i in i for j in j]
    vs = [v[i, j, k] for i in i for j in j]
    xs = [i for i in i for j in j]
    ys = [j - 0.5 for i in i for j in j]
    return arrows2d!(ax, xs, ys, us, vs; lengthscale = 1 / 2maximum(v), kwargs...)
end

# Create a small tripolar grid
tg = TripolarGrid(;
    size = (6, 6, 2),
    z = (-1000, 0),
    southernmost_latitude = -70,
    north_poles_latitude = 75,
    first_pole_longitude = -180,
)
# Create u and v fields with Zipper boundary conditions at the north edge
north_bc = Oceananigans.BoundaryCondition(Oceananigans.BoundaryConditions.Zipper(), -1)
ubcs = FieldBoundaryConditions(tg, (Face(), Center(), Center()), north = north_bc)
vbcs = FieldBoundaryConditions(tg, (Center(), Face(), Center()), north = north_bc)
u = XFaceField(tg; boundary_conditions = ubcs)
v = YFaceField(tg; boundary_conditions = vbcs)
u .= randn(size(tg))
v .= randn(size(tg))
# Zero out u at the poles
(; Nx, Ny, Nz, Hx, Hy, Hz) = tg
u[1, Ny, :] .= 0
u[Nx ÷ 2 + 1, Ny, :] .= 0
# Fill halos of u and v
Oceananigans.fill_halo_regions!(u)
Oceananigans.fill_halo_regions!(v)
# Compute divergence of (u, v)
c = [Oceananigans.div_xyᶜᶜᶜ(i, j, k, tg, u, v) for i in (1 - Hx):(Nx + Hx), j in (1 - Hy):(Ny + Hy), k in 1:Nz]
# Make figure
fig = Figure()
ax = Axis(
    fig[1, 1];
    aspect = DataAspect(),
    limits = (1 - Hx - 1, Nx + Hx + 1, 1 - Hy - 1, Ny + Hy + 1),
    # xticks = 0.5:(Nx + 0.5),
    # yticks = 0.5:(Ny + 0.5),
    xticks = 1-Hx:Nx+Hx, xlabel = "i (center)",
    yticks = 1-Hy:Ny+Hy, ylabel = "j (center)",
    xminorgridvisible = true, xminorticksvisible = true,
    xticksvisible = false, xgridvisible = false,
    xminorgridcolor = (:black, 0.2),
    )
colormap = cgrad(:RdBu, rev = true)
# Heatmap of divergence
hm = heatmap!(ax, (1 - Hx):(Nx + Hx), (1 - Hy):(Ny + Hy), c[:,:,1]; colormap, colorrange = 1.5e-6 .* (-1, 1))
translate!(hm, 0, 0, -20)
Colorbar(fig[2, 1], hm; label = "Divergence", width = Relative(0.666), tellwidth = false, vertical = false)
# Plot u and v
plot_u!(ax, u; color = :black, align = 0.5)#:teal)
plot_v!(ax, v; color = :black, align = 0.5)#:tomato)
# Show "Interior" domain
interior = poly!(ax, [0.5, Nx + 0.5, Nx + 0.5, 0.5], [0.5, 0.5, Ny + 0.5, Ny + 0.5]; color = (:black, 0.03), strokewidth = 1, strokecolor = :black)
translate!(interior, 0, 0, -1)
# North fold
fold = hlines!(ax, Ny, color = :red, linestyle = :dash)
# Poles
poles = scatter!(ax, [0.5, Nx ÷ 2 + 0.5, Nx + 0.5], [Ny, Ny, Ny]; color = :red)
# labels
text!(ax, [1, 1, Nx, Nx], Ny .+ [-1, 1, -1, 1]; text = ["A", "B", "B", "A"], align = (:center, :center))
save("/Users/benoitpasquier/tmp/Oceananigans_signchange_afterfix.png", fig)
fig
