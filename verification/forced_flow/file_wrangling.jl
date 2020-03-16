function iteration_time(filename, iter)
    file = jldopen(filename)
    time = file["timeseries/t/$iter"]
    close(file)
    return time
end

function field_data(filename, field_name, iter)
    file = jldopen(filename)
    data = file["timeseries/$field_name/$iter"]
    close(file)
    return data
end

function RegularCartesianGrid(filename)
    file = jldopen(filename)
    Nx = file["grid/Nx"]
    Ny = file["grid/Ny"]
    Nz = file["grid/Nz"]
    close(file)

    grid = RegularCartesianGrid(size=(Nx, 1, Nz), x=(0, 2π), y=(0, 1), z=(0, π), 
                                topology=(Periodic, Periodic, Bounded))

    return grid
end

function iterations(filename)
    file = jldopen(filename)
    iters = parse.(Int, keys(file["timeseries/t"]))
    close(file)

    return iters
end
