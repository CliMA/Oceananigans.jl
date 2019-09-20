using Oceananigans, Printf, PyPlot

include("utils.jl")

function informative_message(model, u, w, ℕ)
    return @sprintf("""
        This is an informative message.

         model vertical resolution : %d
                   model iteration : %d
        kinetic + potential energy : %.2e
                    kinetic energy : %.2e
               relative error in w : %.2e
               relative error in u : %.2e
        """, model.grid.Nz, model.clock.iteration, total_energy(model, ℕ),
        total_kinetic_energy(model), w_relative_error(model, w), u_relative_error(model, u))
end

function makeplot(axs, model, w)
    w_num = model.velocities.w

    w_ans = FaceFieldZ(w.(xnodes(w_num), ynodes(w_num), znodes(w_num),
                        model.clock.time), model.grid)

    wmax = maximum(abs.(w_num.data))

    sca(axs[1, 1])

    PyPlot.plot(data(w_ans)[1, 1, :], view(znodes(w_num), 1, 1, :),  "-", linewidth=2, alpha=0.4)
    PyPlot.plot(data(w_num)[1, 1, :], view(znodes(w_num), 1, 1, :), "--", linewidth=1)

    xlim(-wmax, wmax)

    axs[1, 1].spines["top"].set_visible(false)
    axs[1, 1].spines["right"].set_visible(false)

    xlabel(L"w")
    ylabel(L"z")

    sca(axs[2, 1])
    PyPlot.plot(data(w_ans)[1, 1, :] .- data(w_num)[1, 1, :], view(znodes(w_num), 1, 1, :), "-")

    xlim(-wmax, wmax)

    axs[2, 1].spines["top"].set_visible(false)
    axs[2, 1].spines["right"].set_visible(false)

    xlabel(L"\mathrm{mean} \left [ \left ( w_\mathrm{model} - w_\mathrm{analytical} \right )^2 \right ] / \mathrm{mean} \left ( w_\mathrm{analytical}^2 \right )")
    ylabel(L"z")

    sca(axs[1, 2])
    plotxzslice(w_num)
    axis("off")
    title("Model vertical velocity")
    xlabel(L"x")
    ylabel(L"z")

    sca(axs[2, 2])
    plotxzslice(w_ans)
    axis("off")
    title("Analytical vertical velocity")
    xlabel(L"x")
    ylabel(L"z")

    return nothing
end

# Internal wave parameters
 ν = κ = 1e-9
 L = 2π
z₀ = -L/3
 δ = L/20
a₀ = 1e-3
 m = 16
 k = 1
 f = 0.2
 ℕ = 1.0
 σ = sqrt( (ℕ^2*k^2 + f^2*m^2) / (k^2 + m^2) )

# Numerical parameters
 N = 128
Δt = 0.01 * 1/σ

@show σ/f
cᵍ = m * σ / (k^2 + m^2) * (f^2/σ^2 - 1)
 U = a₀ * k * σ   / (σ^2 - f^2)
 V = a₀ * k * f   / (σ^2 - f^2)
 W = a₀ * m * σ   / (σ^2 - ℕ^2)
 Θ = a₀ * m * ℕ^2 / (σ^2 - ℕ^2)

a(x, y, z, t) = exp( -(z - cᵍ*t - z₀)^2 / (2*δ)^2 )

u(x, y, z, t) =           a(x, y, z, t) * U * cos(k*x + m*z - σ*t)
v(x, y, z, t) =           a(x, y, z, t) * V * sin(k*x + m*z - σ*t)
w(x, y, z, t) =           a(x, y, z, t) * W * cos(k*x + m*z - σ*t)
T(x, y, z, t) = ℕ^2 * z + a(x, y, z, t) * Θ * sin(k*x + m*z - σ*t)

u₀(x, y, z) = u(x, y, z, 0)
v₀(x, y, z) = v(x, y, z, 0)
w₀(x, y, z) = w(x, y, z, 0)
T₀(x, y, z) = T(x, y, z, 0)

# Create a model where temperature = buoyancy.
model = BasicModel(N=(N, 1, N), L=(L, L, L), ν=ν, κ=κ,
                    eos=LinearEquationOfState(βT=1.0),
                    constants=PlanetaryConstants(f=f, g=1.0))

set_ic!(model, u=u₀, v=v₀, w=w₀, T=T₀)
println(informative_message(model, u, w, ℕ))

fig, axs = subplots(ncols=2, nrows=2, figsize=(8, 8))

for Nt in (1, 10, 100)
    time_step!(model, Nt, Δt)
    makeplot(axs, model, w)
    println(informative_message(model, u, w, ℕ))
end

gcf()
