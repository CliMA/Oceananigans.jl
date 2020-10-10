using Plots

ENV["GKSwstype"] = "100"

u(x, t; uₗ, uᵣ, c, ν) = (uᵣ + uₗ) / 2 - (uₗ - uᵣ) / 2 * tanh((x - c*t) * (uₗ - uᵣ) / 4ν)

uₗ, uᵣ = 1/4, 3/4
c = 1
ν = 1e-3

N = 128
L = 20
T = 10
Δt = 1e-1
x = range(-L/2, L/2, length=N)

anim = @animate for n in 0:Int(T/Δt)
    t = n*Δt
    @info "t=$t"
    uₙ = u.(x, t, uₗ=uₗ, uᵣ=uᵣ, c=c, ν=ν)
    plot(x, uₙ, label="")
end

mp4(anim, "burgers_analytic.mp4", fps=30)