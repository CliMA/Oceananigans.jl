using Oceananigans
using Oceananigans.Fields: regrid!
using GLMakie

x = (-180, 180)
y = (-60, 60)
z = (-1000, 0)
topology = (Periodic, Bounded, Bounded)
arch = GPU()

coarse_grid = RectilinearGrid(arch, size=(90, 30, 2); x, y, z, topology)

Δx = 60
Δy = 20
cᵢ(x, y, z) = exp(z / 200) * exp(-(y+10)^2 / 2Δy^2) * exp(-(x+10)^2 / 2Δx^2)

c = CenterField(coarse_grid)
set!(c, cᵢ)

fine_x_grid   = RectilinearGrid(arch, size=(360, 30, 2); x, y, z, topology)
fine_xy_grid  = RectilinearGrid(arch, size=(360, 120, 2); x, y, z, topology)
fine_xyz_grid = RectilinearGrid(arch, size=(360, 120, 8); x, y, z, topology)

c_x = CenterField(fine_x_grid)
c_xy = CenterField(fine_xy_grid)
c_xyz = CenterField(fine_xyz_grid)
                        
regrid!(c_x, c)
regrid!(c_xy, c_x)
regrid!(c_xyz, c_xy)

fig = Figure()

ax1 = Axis(fig[1, 1], xlabel="x", ylabel="y", title="Coarse grid")
ax2 = Axis(fig[2, 1], xlabel="x", ylabel="y", title="Fine grid in x, y")
ax3 = Axis(fig[3, 1], xlabel="x", ylabel="y", title="Fine grid in x, y, z")

ax4 = Axis(fig[1, 2], xlabel="c", ylabel="z", title="Coarse grid")
ax5 = Axis(fig[2, 2], xlabel="c", ylabel="z", title="Fine grid in x, y")
ax6 = Axis(fig[3, 2], xlabel="c", ylabel="z", title="Fine grid in x, y, z")

ylims!(ax4, -1000, 0)
ylims!(ax5, -1000, 0)
ylims!(ax6, -1000, 0)

x, y, z = nodes(c)
x′, y′, z′ = nodes(c_xyz)

heatmap!(ax1, x, y,   Array(interior(c, :, :, 1)))
heatmap!(ax2, x′, y′, Array(interior(c_xy, :, :, 1)))
heatmap!(ax3, x′, y′, Array(interior(c_xyz, :, :, 1)))

scatter!(ax4, Array(interior(c, 45, 15, :)), z)
scatter!(ax5, Array(interior(c_xy, 180, 60, :)), z)
scatter!(ax6, Array(interior(c_xyz, 180, 60, :)), z′)

display(fig)

