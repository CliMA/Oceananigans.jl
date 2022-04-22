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

function RectilinearGrid(filename)
    file = jldopen(filename)
    Nx = file["grid/Nx"]
    Ny = file["grid/Ny"]
    Nz = file["grid/Nz"]
    Lx = file["grid/Lx"]
    Ly = file["grid/Ly"]
    Lz = file["grid/Lz"]
    close(file)

    grid = RectilinearGrid(size=(Nx, Ny, Nz), x=(0, Lx), y=(0, Ly), z=(0, Lz),
                                topology=(Periodic, Periodic, Bounded))

    return grid
end

function iterations(filename)
    file = jldopen(filename)
    iters = parse.(Int, keys(file["timeseries/t"]))
    close(file)

    return iters
end
