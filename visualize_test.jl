using Oceananigans
using JLD2
using GLMakie

domain = Bounded

domain_size = 20
rη1 = 1
rη2 = 36

ru1 = 1
ru2 = 26

plus_one = domain == Bounded ? 1 : 0

folder = "gpu-2-cores-periodic/"

u0gpu = jldopen(folder * "/variables_rank0.jld2")["uarr"];
u1gpu = jldopen(folder * "/variables_rank1.jld2")["uarr"];
u2gpu = jldopen(folder * "/variables_rank0.jld2")["uarr"];
u3gpu = jldopen(folder * "/variables_rank1.jld2")["uarr"];

η0gpu = jldopen(folder * "/variables_rank0.jld2")["ηarr"];
η1gpu = jldopen(folder * "/variables_rank1.jld2")["ηarr"];
η2gpu = jldopen(folder * "/variables_rank0.jld2")["ηarr"];
η3gpu = jldopen(folder * "/variables_rank1.jld2")["ηarr"];

u0cpu = jldopen("variables_rank0.jld2")["uarr"];
u1cpu = jldopen("variables_rank1.jld2")["uarr"];
u2cpu = jldopen("variables_rank0.jld2")["uarr"];
u3cpu = jldopen("variables_rank1.jld2")["uarr"];

η0cpu = jldopen("variables_rank0.jld2")["ηarr"];
η1cpu = jldopen("variables_rank1.jld2")["ηarr"];
η2cpu = jldopen("variables_rank0.jld2")["ηarr"];
η3cpu = jldopen("variables_rank1.jld2")["ηarr"];

g0 = η0cpu[1].grid.xᶜᵃᵃ[1:domain_size]
g1 = η1cpu[1].grid.xᶜᵃᵃ[1:domain_size]
g2 = η2cpu[1].grid.xᶜᵃᵃ[1:domain_size]
g3 = η3cpu[1].grid.xᶜᵃᵃ[1:domain_size]

g0f = η0cpu[1].grid.xᶠᵃᵃ[1:domain_size]
g1f = η1cpu[1].grid.xᶠᵃᵃ[1:domain_size]
g2f = η2cpu[1].grid.xᶠᵃᵃ[1:domain_size]
g3f = η3cpu[1].grid.xᶠᵃᵃ[1:domain_size + plus_one]

iter = Observable(1)

ηi0g = @lift(η0gpu[$iter][rη1:rη2, 3, 1] .* 1e-1)
ηi1g = @lift(η1gpu[$iter][rη1:rη2, 3, 1] .* 1e-1)
ηi2g = @lift(η2gpu[$iter][rη1:rη2, 3, 1] .* 1e-1)
ηi3g = @lift(η3gpu[$iter][rη1:rη2, 3, 1] .* 1e-1)

ui0g = @lift(u0gpu[$iter][ru1:ru2, 3, 4])
ui1g = @lift(u1gpu[$iter][ru1:ru2, 3, 4])
ui2g = @lift(u2gpu[$iter][ru1:ru2, 3, 4])
ui3g = @lift(u3gpu[$iter][ru1:ru2 + plus_one, 3, 4])

ηi0c =  @lift(interior(η0cpu[$iter], :, 1, 1) .* 1e-1)
ηi1c =  @lift(interior(η1cpu[$iter], :, 1, 1) .* 1e-1)
ηi2c =  @lift(interior(η2cpu[$iter], :, 1, 1) .* 1e-1)
ηi3c =  @lift(interior(η3cpu[$iter], :, 1, 1) .* 1e-1)

ui0c =  @lift(interior(u0cpu[$iter], :, 1, 1))
ui1c =  @lift(interior(u1cpu[$iter], :, 1, 1))
ui2c =  @lift(interior(u2cpu[$iter], :, 1, 1))
ui3c =  @lift(interior(u3cpu[$iter], :, 1, 1))

fig = Figure()
ax  = Axis(fig[1, 1]) 
lines!(ax, g0,  ηi0c, color = :red , linestyle = :solid)
lines!(ax, g0,  ηi0g, color = :blue, linestyle = :solid)
lines!(ax, g0f, ui0c, color = :red , linestyle = :dash)
lines!(ax, g0f, ui0g, color = :blue, linestyle = :dash)
lines!(ax, g1,  ηi1c, color = :red , linestyle = :solid)
lines!(ax, g1,  ηi1g, color = :blue, linestyle = :dash)
lines!(ax, g1f, ui1c, color = :red , linestyle = :dash)
lines!(ax, g1f, ui1g, color = :blue, linestyle = :dash)
lines!(ax, g2,  ηi2c, color = :red , linestyle = :solid)
lines!(ax, g2,  ηi2g, color = :blue, linestyle = :dashdot)
lines!(ax, g2f, ui2c, color = :red , linestyle = :dash)
lines!(ax, g2f, ui2g, color = :blue, linestyle = :dash)
lines!(ax, g3,  ηi3c, color = :red , linestyle = :solid)
lines!(ax, g3,  ηi3g, color = :blue, linestyle = :dot)
lines!(ax, g3f, ui3c, color = :red , linestyle = :dash)
lines!(ax, g3f, ui3g, color = :blue, linestyle = :dash)

GLMakie.record(fig, "bounded-cpu-vs-gpu-4cores-nobuffers.mp4", 1:1000, framerate = 12) do i
    @info "Plotting iteration $i of 600..."
    iter[] = i
end


# fig = Figure()
# ax  = Axis(fig[1, 1]) 
# lines!(ax, (-7:28), ηi0g, color = :blue, linestyle = :solid)
# lines!(ax, (13:48), ηi1g, color = :blue, linestyle = :dash)
# lines!(ax, (33:68), ηi2g, color = :blue, linestyle = :dashdot)
# lines!(ax, (53:88), ηi3g, color = :blue, linestyle = :dot)
