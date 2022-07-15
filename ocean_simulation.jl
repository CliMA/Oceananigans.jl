### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ 077a815f-418d-46d8-b3eb-b444574a4d70
# ╠═╡ show_logs = false
using Pkg; Pkg.add(name="MPICH_jll", version="4.0.1"); Pkg.build()

# ╔═╡ d5e580d4-6005-4813-8770-bef70b2def42
# ╠═╡ show_logs = false
using Oceananigans;

# ╔═╡ 57d45472-7696-4c1e-8b17-ea024aa1b24f
using Oceananigans.Units

# ╔═╡ f5adacde-59bf-4215-bddd-8c55696a7dcd
using Printf

# ╔═╡ 2cc23759-1396-4dd0-bcd5-fe85e99028e3
using Oceananigans.Utils

# ╔═╡ 886d15a5-11f9-4f00-b216-b2f4db5cc285
md"""
Lets start defining some parameters for our simulation
"""

# ╔═╡ 17ab16a5-3dfe-4acd-afc6-a5bd72e0f43b
N = (128, 60, 18)

# ╔═╡ 81d42573-eab7-4f29-83be-e10c4effdd84
md"""
Now we need to specify the extent of our earth: we want the grid to span the whole earth longitudinally and from antartica to the top of Greenland in the latitude direction
"""

# ╔═╡ 48c6241c-fc67-4741-8a44-fb6fcfbc7421
longitude = (-180, 180) # 

# ╔═╡ cd3d8ffb-e034-44e0-a12e-ce8f719dfdb5
latitude = (-84.375, 84.375) 

# ╔═╡ 9877fef3-b1d9-404b-80a7-786224903b9c
z = (-3600, 0)

# ╔═╡ ec4483ff-7eb7-42c4-bc7e-a6ce437643d2
grid = LatitudeLongitudeGrid(CPU(); latitude, longitude, z, size = N)

# ╔═╡ 223e3e78-3b0f-43a6-8c8d-df7b2f2fe5ba
model = HydrostaticFreeSurfaceModel(grid = grid, momentum_advection = VectorInvariant())

# ╔═╡ a167640a-158e-451a-9f94-c6516b33d329
u, v, w = model.velocities

# ╔═╡ a596f379-6478-46ba-ba16-72efebf781a7
simulation = Simulation(model, Δt = 10minutes, stop_time = 20days)

# ╔═╡ 6dec8617-aaa9-47be-bf9d-bb9eda8ffb60
progress(sim) = @sprintf("i: %04d, t: %s, Δt: %s, umax = (%.1e, %.1e) ms⁻¹, wall time: %s\n",
                   iteration(simulation),
                   prettytime(time(simulation)),
                   prettytime(simulation.Δt),
                   maximum(abs, u), maximum(abs, v),
                   prettytime(simulation.run_wall_time))


# ╔═╡ 88120fa8-66b7-4475-bbcd-7e96afd025d3
simulation.callbacks[:progress] = Callback(progress, IterationInterval(10))

# ╔═╡ 2348c880-ab35-4b34-a334-562f7c1255fc
run!(simulation)

# ╔═╡ 3d1b65e2-a3eb-4f36-86d6-f94ce301a17e
md"""
Nothing happened! 
we do not have any forcing, so let's start adding some wind to our earth
"""

# ╔═╡ 195e60a1-e53b-4779-b1e0-8ad0af7d72c7
parameters = (Ly = grid.Ly, τ₀ = 0.2, ρ = 1035)

@inline windshape(y, p) = sin( π * y / p.Ly)
@inline windstress(x, y, t, p) = - p.τ₀ / p.ρ * windshape(y, p)



# ╔═╡ dcac2953-805f-4b2e-bf72-e1d178478ca5


# ╔═╡ 3a04c7a9-f17e-440f-9fe4-cf265946c4f4


# ╔═╡ 35f994e7-b3cb-4349-993b-533b73a194bb


# ╔═╡ ea45c769-50e6-4376-975b-9441aa9d32ad


# ╔═╡ f43e9376-357c-4752-8721-371a4b00586e


# ╔═╡ 18d730cf-f416-4c71-83b2-e76208bed6f0


# ╔═╡ f7f0bd3c-b6fe-4d11-9c67-7b49b8a5a88e


# ╔═╡ Cell order:
# ╠═077a815f-418d-46d8-b3eb-b444574a4d70
# ╠═d5e580d4-6005-4813-8770-bef70b2def42
# ╟─886d15a5-11f9-4f00-b216-b2f4db5cc285
# ╠═17ab16a5-3dfe-4acd-afc6-a5bd72e0f43b
# ╟─81d42573-eab7-4f29-83be-e10c4effdd84
# ╠═48c6241c-fc67-4741-8a44-fb6fcfbc7421
# ╠═cd3d8ffb-e034-44e0-a12e-ce8f719dfdb5
# ╠═9877fef3-b1d9-404b-80a7-786224903b9c
# ╠═ec4483ff-7eb7-42c4-bc7e-a6ce437643d2
# ╠═223e3e78-3b0f-43a6-8c8d-df7b2f2fe5ba
# ╠═a167640a-158e-451a-9f94-c6516b33d329
# ╠═57d45472-7696-4c1e-8b17-ea024aa1b24f
# ╠═a596f379-6478-46ba-ba16-72efebf781a7
# ╠═f5adacde-59bf-4215-bddd-8c55696a7dcd
# ╠═2cc23759-1396-4dd0-bcd5-fe85e99028e3
# ╠═6dec8617-aaa9-47be-bf9d-bb9eda8ffb60
# ╠═88120fa8-66b7-4475-bbcd-7e96afd025d3
# ╠═2348c880-ab35-4b34-a334-562f7c1255fc
# ╟─3d1b65e2-a3eb-4f36-86d6-f94ce301a17e
# ╠═195e60a1-e53b-4779-b1e0-8ad0af7d72c7
# ╠═dcac2953-805f-4b2e-bf72-e1d178478ca5
# ╠═3a04c7a9-f17e-440f-9fe4-cf265946c4f4
# ╠═35f994e7-b3cb-4349-993b-533b73a194bb
# ╠═ea45c769-50e6-4376-975b-9441aa9d32ad
# ╠═f43e9376-357c-4752-8721-371a4b00586e
# ╠═18d730cf-f416-4c71-83b2-e76208bed6f0
# ╠═f7f0bd3c-b6fe-4d11-9c67-7b49b8a5a88e
