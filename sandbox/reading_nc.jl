using NetCDF, Plots

filename = pwd() * "/Windstress_Convection_Example_Constant" * "_meridional.nc"
ncinfo(filename)
y = Array(NetCDF.open(filename, "yC"))
z = Array(NetCDF.open(filename, "zF"))
t = Array(NetCDF.open(filename, "time"))
sim_day = t ./ 86400
z = (z[2:end] + z[1:end-1]) / 2
b = NetCDF.open(filename, "b")
u = NetCDF.open(filename, "u")
v = NetCDF.open(filename, "v")

for i in 1:1:240*10
    b_array = b[1, :, :, i]
    # p1 = contourf(y, z, b_array, fill = true, linewidth = 0, color = :ocean, clim = (-0.0013871555898403098, -3.3776441214941526e-6))
    day_label = @sprintf("%.2f ", sim_day[i])
    p1 = contourf(y, z, b_array', clim = (-0.0013871555898403098, -3.3776441214941526e-6), color = :plasma, title = "Buoyancy at Midchannel at day " * day_label , xlabel = "Meridional [m]", ylabel = "Depth [m]")
    display(p1)
end
###
gr()
b_array = b[1, :, :, end]
p1 = contour(z, y, b_array, fill = true, linewidth = 0, color = :ocean)

b_array = b[1, :, :, end]

plot(b_array[end, :], z)
