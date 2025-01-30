using GLMakie
using Oceananigans

function visualize_simulation(var)
    iter = Observable(1)

    v = Vector(undef, 4)
    V = Vector(undef, 4)
    x = Vector(undef, 4)
    y = Vector(undef, 4)

    for r in 1:4
        v[r] = FieldTimeSeries("mpi_hydrostatic_turbulence_rank$(r-1).jld2", var; boundary_conditions=nothing)
        nx, ny, _ = size(v[r])
        V[r] = @lift(interior(v[r][$iter], 1:nx, 1:ny, 1))

        x[r] = xnodes(v[r])
        y[r] = ynodes(v[r])
    end

    fig = Figure()
    ax = Axis(fig[1, 1])
    for r in 1:4
        heatmap!(ax, x[r], y[r], V[r], colorrange = (-1.0, 1.0))
    end

    GLMakie.record(fig, "hydrostatic_test_" * var * ".mp4", 1:length(v[1].times), framerate = 11) do i
        @info "step $i"; 
        iter[] = i; 
    end
end

visualize_simulation("u")
visualize_simulation("v")
visualize_simulation("c")
