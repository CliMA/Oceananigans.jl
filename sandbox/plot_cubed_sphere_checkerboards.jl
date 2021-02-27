using PyCall

plt = pyimport("matplotlib.pyplot")
ccrs = pyimport("cartopy.crs")

include("conformal_cubed_sphere_grid.jl")

Nx_face = Ny_face = 21
grid = ConformalCubedSphereGrid(face_size=(Nx_face, Ny_face, 1), z=(-1, 0))

## Plot staggered grid points on a checkerboard for one face

transform = ccrs.PlateCarree()

fig = plt.figure(figsize=(16, 9))

cmaps = ("coolwarm", "viridis", "cividis", "plasma", "PiYG", "bone")

ax = fig.add_subplot(1, 2, 1, projection=ccrs.NearsidePerspective(central_longitude=270, central_latitude=45))

for (n, face) in enumerate(grid.faces)
    lons = face.λᶜᶜᶜ[1:Nx_face, 1:Ny_face]
    lats = face.ϕᶜᶜᶜ[1:Nx_face, 1:Ny_face]
    checkerboard = [(i+j)%2 for i in 1:Nx_face, j in 1:Ny_face]

    ax.pcolormesh(lons, lats, checkerboard, transform=transform, cmap=cmaps[n], alpha=0.4)
end

ax.gridlines(xlocs=range(-180, 180, step=20), ylocs=range(-80, 80, step=10))
ax.coastlines(resolution="50m")
ax.set_global()

ax = fig.add_subplot(1, 2, 2, projection=ccrs.NearsidePerspective(central_longitude=90, central_latitude=-45))

for (n, face) in enumerate(grid.faces)
    lons = face.λᶜᶜᶜ[1:Nx_face, 1:Ny_face]
    lats = face.ϕᶜᶜᶜ[1:Nx_face, 1:Ny_face]
    checkerboard = [(i+j)%2 for i in 1:Nx_face, j in 1:Ny_face]

    ax.pcolormesh(lons, lats, checkerboard, transform=transform, cmap=cmaps[n], alpha=0.4)
end

ax.gridlines(xlocs=range(-180, 180, step=20), ylocs=range(-80, 80, step=10))
ax.coastlines(resolution="50m")
ax.set_global()

plt.show()
# plt.savefig("cubed_sphere_staggered_grid.png", dpi=200)

plt.close("all")
