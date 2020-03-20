location(s::Symbol) = (s === :u ? (Face, Cell, Cell) :
                       s === :v ? (Cell, Face, Cell) :
                       s === :w ? (Cell, Cell, Face) :
                                  (Cell, Cell, Cell))

print_min_max_mean(ψ, name="") =
    @printf("%s max: %.9e, min: %.9e, mean: %.9e\n", name, minimum(ψ), maximum(ψ), mean(ψ))

function extract_two_solutions(analytical_solution, filename; name=:u)
    grid = RegularCartesianGrid(filename)
    iters = iterations(filename)
    loc = location(name)

    ψ_data = field_data(filename, name, iters[end])
    ψ_simulation = Field{loc[1], loc[2], loc[3]}(ψ_data, grid, FieldBoundaryConditions(grid, loc))

    x, y, z = nodes(ψ_simulation)

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
    L₁ = mean(absolute_error) / mean(absolute_truth)
    L₂ = mean(absolute_error.^2) / mean(absolute_truth.^2)
    L∞ = maximum(absolute_error) / maximum(absolute_truth)

    return (L₁=L₁, L∞=L∞)
end

function compute_error(analytical_solution, filename::String; kwargs...)
    u_simulation, u_analytical = extract_two_solutions(analytical_solution, filename; kwargs...)
    return compute_error(u_simulation, u_analytical)
end

compute_errors(analytical_solution, filenames::String...; kwargs...) = 
    [compute_error(analytical_solution, filename; kwargs...) for filename in filenames]

extract_sizes(filenames...) = [size(RegularCartesianGrid(filename)) for filename in filenames]
