import Oceananigans.Fields: location

using Oceananigans: Face, Cell

location(s::Symbol) = s === :u ? (Face, Cell, Cell) :
                      s === :v ? (Cell, Face, Cell) :
                      s === :w ? (Cell, Cell, Face) :
                                 (Cell, Cell, Cell)

function extract_two_solutions(analytical_solution, filename; name=:u)
    grid = RegularCartesianGrid(filename)
    iters = iterations(filename)
    loc = location(name)

    ψ_data = field_data(filename, name, iters[end])
    ψ_simulation = Field{loc[1], loc[2], loc[3]}(u_data, grid, FieldBoundaryConditions(grid, loc))

    Nx, Ny, Nz = size(grid)

    ψ_simulation = interior(ψ_simulation)

    x, y, z = nodes(ψ_simulation)
    t = iteration_time(filename, iters[end])
    ψ_analytical = analytical_solution.(x, y, z, t)

    return u_simulation, u_analytical
end

function compute_error(u_simulation, u_analytical)
    absolute_error = @. abs(u_simulation - u_analytical)
    L₁ = mean(absolute_error)
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
