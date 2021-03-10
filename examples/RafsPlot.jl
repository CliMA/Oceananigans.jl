
file = jldopen(simulation.output_writers[:fields].filepath)
iterations = parse.(Int, keys(file["timeseries/t"]))
# "./rh.jld2"

##
file = jldopen("./rh.jld2")
iterations = parse.(Int, keys(file["timeseries/t"]))

λ, ϕ, r = nodes(model.free_surface.η, reshape=true)
ϕind = 140
nλ , _ = size(file["timeseries/u/" * string(0)])
u_time = zeros(nλ, length(iterations))
for (i,iter) in enumerate(iterations)
    u_time[:,i] = file["timeseries/u/" * string(iter)][:, ϕind, 1]
end
t = [file["timeseries/t/" * string(iter)] for iter in iterations] ./ 86400
iterations
##
file = jldopen("./rh_liner.jld2")
iterations = parse.(Int, keys(file["timeseries/t"]))

λ, ϕ, r = nodes(model.free_surface.η, reshape=true)
ϕind = 140
nλ , _ = size(file["timeseries/u/" * string(0)])
u_l_time = zeros(nλ, length(iterations))
for (i,iter) in enumerate(iterations)
    u_l_time[:,i] = file["timeseries/u/" * string(iter)][:, ϕind, 1]
end
t = [file["timeseries/t/" * string(iter)] for iter in iterations] ./ 86400
iterations
##
titlelabel = "Zonal Velocity at ϕ = " * @sprintf("%0.2f", ϕ[ϕind]) * " degrees"
fig = Figure(resolution  = (1466, 918))
ax1 = fig[1, 1] = Axis(fig, title = titlelabel, titlesize = 50)
heatmap!(ax1, λ[:], t, u_l_time - u_time, colormap = :balance)
ax1.xlabel = "Longidute in Degrees"
ax1.ylabel = "Time in Days"
ax1.xlabelsize = 30
ax1.ylabelsize = 30             