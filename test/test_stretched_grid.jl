using Revise

using Oceananigans

@inline    linear(x, L, N) = x .+ L*collect(range(0, 1, length=N+1))
@inline quadratic(x, L, N) = x .+ L*collect(range(0, 1, length=N+1)).^2
@inline     cubic(x, L, N) = x .+ L*collect(range(0, 1, length=N+1)).^3

stretch = Dict([("x", linear), ("y", quadratic), ("z", cubic)])

grid1 = StretchedRectilinearGrid(size=(8, 8, 8), 
                                    x=(0,8), y=(0,8), z=(0,8), 
                                    halo = (1,1,1),
                             topology=(Bounded, Bounded, Bounded),
                              stretch=stretch)
show(grid1)

grid2 = StretchedRectilinearGrid(size=(8, 8), 
                                    x=(0,8), y=(0,8), 
                             topology=(Bounded, Bounded, Flat),
                              stretch=stretch)
show(grid2)

grid3 = StretchedRectilinearGrid(size=(8), 
                                    x=(0,8), 
                             topology=(Bounded, Flat, Flat),
                              stretch=stretch)
show(grid3)

grid4 = StretchedRectilinearGrid(size=(),  
                             topology=(Flat, Flat, Flat),
                              stretch=stretch)
show(grid4)
