using PyCall

plt = pyimport("matplotlib.pyplot")
ccrs = pyimport("cartopy.crs")

include("conformal_cubed_sphere_grid.jl")

Nx_face = Ny_face = 20
grid = ConformalCubedSphereGrid(face_size=(Nx_face, Ny_face, 1), z=(-1, 0))

## Plot points of each face

fig = plt.figure(figsize=(16, 9))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.Mollweide())

for (n, face) in enumerate(grid.faces)
    lons = face.λᶜᶜᶜ[1:Nx_face, 1:Ny_face]
    lats = face.ϕᶜᶜᶜ[1:Nx_face, 1:Ny_face]
    ax.scatter(lons, lats, s=4, label="face $n", transform=ccrs.PlateCarree())
end

ax.legend(loc="lower right")
ax.coastlines(resolution="50m")
ax.set_global()

# plt.show()
plt.savefig("cubed_sphere_points.png", dpi=200)

## Plot staggered grid points on a checkerboard for one face

Nx_face = Ny_face = 10
grid = ConformalCubedSphereGrid(face_size=(Nx_face, Ny_face, 1), z=(-1, 0))

fig = plt.figure(figsize=(16, 9))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.Mollweide())

face = grid.faces[3]
lons = face.λᶜᶜᶜ[1:Nx_face, 1:Ny_face]
lats = face.ϕᶜᶜᶜ[1:Nx_face, 1:Ny_face]
checkerboard = [(i+j)%2 for i in 1:Nx_face, j in 1:Ny_face]

ax.pcolormesh(lons, lats, checkerboard, transform=ccrs.PlateCarree(), cmap="coolwarm", alpha=0.2)

ax.scatter(face.λᶜᶜᶜ[1:Nx_face,   1:Ny_face  ], face.ϕᶜᶜᶜ[1:Nx_face,   1:Ny_face  ], s=4, label="ccc (tracer)", transform=ccrs.PlateCarree())
ax.scatter(face.λᶠᶜᶜ[1:Nx_face+1, 1:Ny_face  ], face.ϕᶠᶜᶜ[1:Nx_face+1, 1:Ny_face  ], s=4, label="fcc (u)", transform=ccrs.PlateCarree())
ax.scatter(face.λᶜᶠᶜ[1:Nx_face,   1:Ny_face+1], face.ϕᶜᶠᶜ[1:Nx_face,   1:Ny_face+1], s=4, label="cfc (v)", transform=ccrs.PlateCarree())
ax.scatter(face.λᶠᶠᶜ[1:Nx_face+1, 1:Ny_face+1], face.ϕᶠᶠᶜ[1:Nx_face+1, 1:Ny_face+1], s=4, label="ffc (vorticity)", transform=ccrs.PlateCarree())

ax.legend(loc="lower right")
ax.coastlines(resolution="50m")
ax.set_global()

# plt.show()
plt.savefig("cubed_sphere_staggered_grid.png", dpi=200)

plt.close("all")
