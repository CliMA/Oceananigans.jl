using Revise

using Oceananigans
#using Oceananigans.Grids: validate_size

#print("Test vertically stretched grid\n")
#grid2 = VerticallyStretchedRectilinearGrid(size=(1, 1, 8), x=(0,1), y=(0,1), zF=collect(0:8))

print("Test zonally stretched grid in 3D\n")
#grid3 = ZonallyStretchedRectilinearGrid(size=(8, 1, 1), xF=collect(0:8), y=(0,1), z=(0,1), topology=(Bounded, Periodic, Periodic))
#grid3 = ZonallyStretchedRectilinearGrid(size=(8, 8, 8), x=(0,8), y=(0,8), z=(0,8), topology=(Bounded, Bounded, Periodic))
grid4 = ZonallyStretchedRectilinearGrid(size=(8, 8), x=(0,8), y=(0,8), topology=(Bounded, Bounded, Flat))

#print("Test zonally stretched grid in 2D\n")
#grid4 = ZonallyStretchedRectilinearGrid(size=(8, 1), xF=collect(0:8), y=(0,1), topology=(Bounded, Periodic, Flat))
##grid4 = ZonallyStretchedRectilinearGrid(size=(8, 1), xF=collect(0:8), extent=(1), topology=(Bounded, Periodic, Flat))

#print("Test zonally stretched grid in 1D\n")
#grid4 = ZonallyStretchedRectilinearGrid(size=(8),  x=(0,8), topology=(Bounded, Flat, Flat))
#grid4 = ZonallyStretchedRectilinearGrid(size=(8), xF=collect(0:8), topology=(Bounded, Flat, Flat))
#grid4 = ZonallyStretchedRectilinearGrid(size=(8), xF=collect(0:8), topology=(Periodic, Flat, Flat))