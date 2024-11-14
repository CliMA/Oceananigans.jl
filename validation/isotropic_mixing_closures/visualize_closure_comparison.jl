using Oceananigans
using GLMakie

r = "20"

filenames = [
    "wind_driven_AMD_$r.jld2",
    "wind_driven_WENO5_$r.jld2",
    "wind_driven_WENO9_$r.jld2",
    "wind_driven_constant_smagorinsky_$r.jld2",
    "wind_driven_smagorinsky_lilly_$r.jld2",
    "wind_driven_directional_smagorinsky_$r.jld2",
    "wind_driven_lagrangian_smagorinsky_$r.jld2",
]

labels = [
    "AMD",
    "WENO(order=5)",
    "WENO(order=9)",
    "Constant Smagorinsky",
    "Smagorinsky-Lilly",
    "Dynamic Smagorinsky (directional average)",
    "Dynamic Smagorinsky (Lagrangian average)",
]

hasnu = [
    true,
    false,
    false,
    false,
    true,
    true,
    true,
]

Bs = []
Us = []
w²s = []
νs = []
for (hasν, name) in zip(hasnu, filenames)
    bt = FieldTimeSeries(name, "b", backend=OnDisk())
    ut = FieldTimeSeries(name, "u", backend=OnDisk())
    wt = FieldTimeSeries(name, "w", backend=OnDisk())

    bn = bt[end]
    un = ut[end]
    wn = wt[end]

    Bn = compute!(Field(Average(bn, dims=(1, 2))))
    Un = compute!(Field(Average(un, dims=(1, 2))))
    w²n = compute!(Field(Average(wn^2, dims=(1, 2))))

    push!(Bs, Bn)
    push!(Us, Un)
    push!(w²s, w²n)

    if hasν
        νt = FieldTimeSeries(name, "νₑ", backend=OnDisk())
        νn = νt[end]
        Nun = compute!(Field(Average(νn, dims=(1, 2))))
        push!(νs, Nun)
    else
        push!(νs, nothing)
    end
end

fig = Figure(size=(1400, 800))

axb = Axis(fig[1, 1], ylabel="z (m)", xlabel="Buoyancy (m s⁻²)")
axu = Axis(fig[1, 2], ylabel="z (m)", xlabel="x-velocity (m s⁻¹)")
axw = Axis(fig[1, 3], ylabel="z (m)", xlabel="Vertical velocity variance, w² (m² s⁻²)")
axν = Axis(fig[1, 4], ylabel="z (m)", xlabel="Eddy viscosity (m² s⁻¹)")

for (label, Bn) in zip(labels, Bs)
    lines!(axb, Bn; label)
end

for Un in Us
    lines!(axu, Un, label="u")
end

for w²n in w²s
    lines!(axw, w²n)
end

for i = 1:length(hasnu)
    hasν = hasnu[i]
    νn = νs[i]
    label = labels[i]
    if hasν
        lines!(axν, νn; label)
    else
        lines!(axν, [NaN], [NaN])
    end
end

axislegend(axb, position=:lt)
axislegend(axν, position=:rb)

display(fig)

save("closure_comparison_$r.png", fig)
