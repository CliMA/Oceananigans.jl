using Oceananigans
using Oceananigans.MultiRegion: getregion

region = 1
Nx, Ny, Nz = 5, 5, 1

grid = ConformalCubedSphereGrid(panel_size=(Nx, Ny, Nz), z=(-1, 0), radius=1, 
                                horizontal_direction_halo = 1, z_topology=Bounded)

c = CenterField(grid)
u = XFaceField(grid)
v = YFaceField(grid)

# Plotting longitude values using the set function

set!(c, (λ, φ, z) -> λ)
set!(u, (λ, φ, z) -> λ)
set!(v, (λ, φ, z) -> λ)
colorrange = (-180, 180)

@info "Plotting longitude values using the set function"

@info "c values"
display(rotl90(view(getregion(c, region).data, :, :, 1)))

@info "u values"
display(rotl90(view(getregion(u, region).data, :, :, 1)))

@info "v values"
display(rotl90(view(getregion(v, region).data, :, :, 1)))

# Plotting longitude values using λnodes

Hx, Hy, Hz = grid.Hx, grid.Hy, grid.Hz

for region in 1:6
        
    for i in 1-Hx:Nx+Hx, j in 1-Hy:Ny+Hy
        getregion(c, region).data[i, j, 1] = λnodes(getregion(grid, region), Center(), Center(), Center(); 
                                                    with_halos=true)[i, j, 1]
        getregion(u, region).data[i, j, 1] = λnodes(getregion(grid, region), Face(), Center(), Center(); 
                                                    with_halos=true)[i, j, 1]
        getregion(v, region).data[i, j, 1] = λnodes(getregion(grid, region), Center(), Face(), Center(); 
                                                    with_halos=true)[i, j, 1]
    end

end

@info "Plotting longitude values using λnodes"

@info "c values"
display(rotl90(view(getregion(c, region).data, :, :, 1)))

@info "u values"
display(rotl90(view(getregion(u, region).data, :, :, 1)))

@info "v values"
display(rotl90(view(getregion(v, region).data, :, :, 1)))