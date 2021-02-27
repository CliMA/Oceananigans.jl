using PyCall

plt = pyimport("matplotlib.pyplot")
ccrs = pyimport("cartopy.crs")

include("conformal_cubed_sphere_grid.jl")

Nx_face = Ny_face = 20
grid = ConformalCubedSphereGrid(face_size=(Nx_face, Ny_face, 1), z=(-1, 0))

## Plot points of each face

projection = ccrs.Mollweide()
transform = ccrs.PlateCarree()

fig = plt.figure(figsize=(16, 9))
ax = fig.add_subplot(1, 1, 1, projection=projection)

for (n, face) in enumerate(grid.faces)
    lons = face.λᶜᶜᶜ[1:Nx_face, 1:Ny_face]
    lats = face.ϕᶜᶜᶜ[1:Nx_face, 1:Ny_face]
    ax.scatter(lons, lats, s=4, label="face $n", transform=transform)
end

ax.legend(loc="lower right")
ax.coastlines(resolution="50m")
ax.set_global()

# plt.show()
plt.savefig("cubed_sphere_points.png", dpi=200)

plt.close("all")
