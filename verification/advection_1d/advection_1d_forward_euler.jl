using OffsetArrays

# Second order advection
@inline advective_flux(i, N, u, ϕ) = u[i] * (ϕ[i-1] + ϕ[i]) / 2
@inline ∂x_advective_flux(i, N, Δx, u, ϕ) = (advective_flux(i+1, N, u, ϕ) - advective_flux(i, N, u, ϕ)) / Δx

N = 128
L = 1
Δx = L/N

x = range(-L/2 + Δx/2, L/2 - Δx/2; length=N)
ϕ₀ = @. exp(-25x^2)

u = OffsetArray(ones(N+2), 0:N+1)
ϕ = OffsetArray([0, ϕ₀..., 0], 0:N+1)

T = 1 # end time
CFL = 0.1
Δt = CFL * Δx / maximum(u)
nt = ceil(T/Δt) # number of time steps

for iter in 1:nt
    ϕ[0], ϕ[N+1] = ϕ[N], ϕ[1]  # Fill ghost points.
    for i in 1:N
        ϕ[i] = ϕ[i] - Δt * ∂x_advective_flux(i, N, Δx, u, ϕ)
    end
end

using Plots
plot(x, ϕ₀, label="t=0")
plot!(x, ϕ[1:N], label="t=1")
