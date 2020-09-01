using Test

location(s::Symbol) = (s === :u ? (Face, Cell, Cell) :
                       s === :v ? (Cell, Face, Cell) :
                       s === :w ? (Cell, Cell, Face) :
                                  (Cell, Cell, Cell))

print_min_max_mean(ψ, name="") =
    @info @sprintf("%s min: %.9e, max: %.9e, mean: %.9e\n", name, minimum(ψ), maximum(ψ), mean(ψ))

function extract_two_solutions(analytical_solution, filename; name=:u)
    grid = RegularCartesianGrid(filename)
    iters = iterations(filename)
    loc = location(name)

    ψ_raw = field_data(filename, name, iters[end])

    ψ_raw = ψ_raw[
                  1 : grid.Nx + 2 * grid.Hx,
                  1 : grid.Ny + 2 * grid.Hy,
                  1 : grid.Nz + 2 * grid.Hz
                 ]

    tx, ty, tz = size(ψ_raw)
    ψ_data = OffsetArray(ψ_raw, 0:tx-1, 0:ty-1, 0:tz-1)
    ψ_simulation = Field{loc[1], loc[2], loc[3]}(ψ_data, grid, FieldBoundaryConditions(grid, loc))

    x, y, z = nodes(ψ_simulation, reshape=true)

    ψ_simulation = interior(ψ_simulation)

    t = iteration_time(filename, iters[end])
    ψ_analytical = analytical_solution.(x, y, z, t)

    print_min_max_mean(ψ_simulation, "simulation")
    print_min_max_mean(ψ_analytical, "analytical")

    return ψ_simulation, ψ_analytical
end

function compute_error(u_simulation, u_analytical)
    absolute_error = @. abs(u_simulation - u_analytical)
    absolute_truth = abs.(u_analytical)
    L₁ = mean(absolute_error)
    L₂ = mean(absolute_error.^2)
    L∞ = maximum(absolute_error)

    return (L₁=L₁, L∞=L∞)
end

function compute_error(analytical_solution, filename::String; kwargs...)
    u_simulation, u_analytical = extract_two_solutions(analytical_solution, filename; kwargs...)
    return compute_error(u_simulation, u_analytical)
end

compute_errors(analytical_solution, filenames::String...; kwargs...) =
    [compute_error(analytical_solution, filename; kwargs...) for filename in filenames]

extract_sizes(filenames...) = [size(RegularCartesianGrid(filename)) for filename in filenames]

function test_rate_of_convergence(error, Δ; name="", data_points=length(error), expected, atol)
    d = data_points
    ROC = log10(error[1] / error[d]) / log10(Δ[1] / Δ[d])
    @info (name == "" ? "" : name * " ") * "rate of convergence = $ROC (expected ≈ $expected, atol = $atol)"
    @test isapprox(ROC, expected, atol=atol)
    return ROC
end
