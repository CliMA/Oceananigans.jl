using Oceananigans
using Oceananigans.Fields: regrid!
using GLMakie

longitude = (-180, 180)
latitude = (-60, 60)
z = (-1000, 0)
arch = GPU()

coarse_grid = LatitudeLongitudeGrid(arch, size=(90, 30, 2); longitude, latitude, z)

Δλ = 60
Δφ = 20
cᵢ(λ, φ, z) = exp(z / 200) * exp(-(φ+10)^2 / 2Δφ^2) * exp(-(λ+10)^2 / 2Δλ^2)

c = CenterField(coarse_grid)
set!(c, cᵢ)

fine_x_grid = LatitudeLongitudeGrid(arch, size=(360, 30, 2); longitude, latitude, z)
fine_xy_grid = LatitudeLongitudeGrid(arch, size=(360, 120, 2); longitude, latitude, z)
fine_xyz_grid = LatitudeLongitudeGrid(arch, size=(360, 120, 8); longitude, latitude, z)

c_x = CenterField(fine_x_grid)
c_xy = CenterField(fine_xy_grid)
c_xyz = CenterField(fine_xyz_grid)
                        
regrid!(c_x, c)
regrid!(c_xy, c_x)
regrid!(c_xyz, c_xy)

fig = Figure()

ax1 = Axis(fig[1, 1], xlabel="λ", ylabel="φ", title="Coarse grid")
ax2 = Axis(fig[2, 1], xlabel="λ", ylabel="φ", title="Fine grid in x, y")
ax3 = Axis(fig[3, 1], xlabel="λ", ylabel="φ", title="Fine grid in x, y, z")

ax4 = Axis(fig[1, 2], xlabel="c", ylabel="z", title="Coarse grid")
ax5 = Axis(fig[2, 2], xlabel="c", ylabel="z", title="Fine grid in x, y")
ax6 = Axis(fig[3, 2], xlabel="c", ylabel="z", title="Fine grid in x, y, z")

ylims!(ax4, -1000, 0)
ylims!(ax5, -1000, 0)
ylims!(ax6, -1000, 0)

λ, φ, z = nodes(c)
λ′, φ′, z′ = nodes(c_xyz)

colorrange=(0, 0.02)

heatmap!(ax1, λ, φ,   Array(interior(c, :, :, 1)); colorrange)
heatmap!(ax2, λ′, φ′, Array(interior(c_xy, :, :, 1)); colorrange)
heatmap!(ax3, λ′, φ′, Array(interior(c_xyz, :, :, 1)); colorrange)

scatter!(ax4, Array(interior(c, 45, 15, :)), z)
scatter!(ax5, Array(interior(c_xy, 180, 60, :)), z)
scatter!(ax6, Array(interior(c_xyz, 180, 60, :)), z′)

display(fig)

