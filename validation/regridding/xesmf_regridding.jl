using Oceananigans
using XESMF

arch = CPU()

z = (-1, 0)

radius = Oceananigans.defaults.planet_radius

llg_coarse = LatitudeLongitudeGrid(arch; z, radius,
                                   size = (180, 90, 1),
                                   longitude = (0, 360),
                                   latitude = (-90, 90))

llg_fine = LatitudeLongitudeGrid(arch; z, radius,
                                 size = (360, 180, 1),
                                 longitude = (0, 360),
                                 latitude = (-90, 90))

src_field = CenterField(llg_coarse)
dst_field = CenterField(llg_fine)

λ₀, φ₀ = 150, 30   # degrees
width = 12         # degrees
set!(src_field, (λ, φ, z) -> exp(-((λ - λ₀)^2 + (φ - φ₀)^2) / 2width^2) - 2exp(-((λ - 270)^2 + (φ + 20)^2) / 2width^2))

regridder = XESMF.Regridder(dst_field, src_field, method="conservative")

regrid!(dst_field, regridder, src_field)

println("integral src_field = ", first(Field(Integral(src_field, dims=(1, 2)))))
println("integral dst_field = ", first(Field(Integral(dst_field, dims=(1, 2)))))
