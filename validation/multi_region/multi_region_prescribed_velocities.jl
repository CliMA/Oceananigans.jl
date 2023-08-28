using Oceananigans
using Oceananigans.MultiRegion: getregion
using Oceananigans.Operators: div_xyᶜᶜᶜ # It is called with signature div_xyᶜᶜᶜ(i, j, k, grid, u, v).
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: replace_horizontal_velocity_halos!

include("multi_region_cubed_sphere.jl")

Nx, Ny, Nz = 10, 10, 1
u_advection = 0.5

# The following lines of code compute and print out the velocity divergence on a rectilinear grid.

grid = RectilinearGrid(size=(Nx, Ny, Nz), halo = (1, 1, 1), extent=(2π, 2π, 1))
Hx, Hy, Hz = grid.Hx, grid.Hy, grid.Hz

U(x, y, z) =  u_advection * cos(x) * sin(y)
V(x, y, z) = -u_advection * sin(x) * cos(y)

u = XFaceField(grid) 
set!(u, U)

v = YFaceField(grid) 
set!(v, V)

fill_halo_regions!(u)
fill_halo_regions!(v)
δ = Field(∂x(u) + ∂y(v))

k = Hz + (Nz + 1)÷2

@info "Zonal Velocity (u)"
display(rotl90(u.data[:, :, k]))

@info "Meridional Velocity (v)"
display(rotl90(v.data[:, :, k]))

@info "Divergence (δ)"
display(rotl90(δ.data[:, :, k]))
    
# The following lines of code compute the velocity divergence on a conformal cubed sphere grid and plot the velocities 
# on the panels.

grid = ConformalCubedSphereGrid(panel_size=(Nx, Ny, Nz), z=(-1, 0), radius=1, horizontal_direction_halo = 1, 
                                z_topology=Bounded)
Hx, Hy, Hz = grid.Hx, grid.Hy, grid.Hz

U(λ, φ, z) =   u_advection * cosd(λ) * sind(φ)
V(λ, φ, z) = - u_advection * sind(λ)
    
u = XFaceField(grid) 
set!(u, U)

v = YFaceField(grid) 
set!(v, V)

for _ in 1:2
    fill_halo_regions!(u)
    fill_halo_regions!(v)
    @apply_regionally replace_horizontal_velocity_halos!((; u = u, v = v, w = nothing), grid)
end

div_op = KernelFunctionOperation{Center, Center, Center}(div_xyᶜᶜᶜ, grid, u, v)
δ = compute!(Field(div_op))
    
k = grid.region_grids[1].Hz + (Nz + 1)÷2

panel_wise_visualization("zonal_velocity", u, "u", "nodes", k)
panel_wise_visualization("meridional_velocity", v, "v", "nodes", k)
panel_wise_visualization("divergence", δ, "c", "nodes", k)

tangential_velocities_around_cubed_sphere_panels(u, v, k)
normal_velocities_around_cubed_sphere_panels(u, v, k)