using Oceananigans.Grids

import Oceananigans.Grids: topology, domain_string

struct MultiRegionGrid{FT, G, F, C} <: AbstractGrid{FT, Connected, Connected, Bounded}
         regions :: F
    connectivity :: C

    """
        MultiRegionGrid(regional_grids::F, connectivity::C) where {F <: Tuple, C}

    Contains the grids for a collection of connected `regions` and `connectivity`
    that describes the overall domain topology and how the regional grids are
    connected to one another.
    """
    function MultiRegionGrid(regional_grids::F, connectivity::C) where {F <: Tuple, C}
        FT = eltype(regional_grids[1])
        G = typeof(regional_grids[1])
        return new{FT, G, F, C}(regional_grids, connectivity)
    end
end

function Base.show(io::IO, grid::MultiRegionGrid{FT, G}) where {FT, G}
    Nr = length(grid.regions)
    sizes = Tuple(size(region) for region in grid.regions)
    print(io, "MultiRegionGrid{$FT, $G}: $Nr regions with size = $sizes")
end

##### 
##### Connectivity for each boundary of a connected region.
##### Regions can be connected in horizontal directions only.
##### 

struct RegionConnectivity{W, E, S, N}
     west :: W
     east :: E
    south :: S
    north :: N
end

RegionConnectivity(; west, east, south, north) =
    RegionConnectivity(west, east, south, north)

function Base.show(io::IO, connectivity::RegionConnectivity)
    print(io, "RegionConnectivity:\n",
              "├── west: $(short_string(connectivity.west))\n",
              "├── east: $(short_string(connectivity.east))\n",
              "├── south: $(short_string(connectivity.south))\n",
              "└── north: $(short_string(connectivity.north))")
end

#####
##### Connectivity "details" (one for each boundary)
#####

struct RegionConnectivityDetails{F, S}
    face :: F
    side :: S
end

short_string(deets::RegionConnectivityDetails) = "face $(deets.face) $(deets.side) side"

Base.show(io::IO, deets::RegionConnectivityDetails) =
    print(io, "RegionConnectivityDetails: $(short_string(deets))")

#####
##### Grid utils
#####

Base.eltype(grid::MultiRegionGrid{FT}) where FT = FT

topology(::MultiRegionGrid) = (Connected, Connected, Bounded)

# Not sure what to put. Gonna leave it blank so that Base.show(io::IO, operation::AbstractOperation) doesn't error.
domain_string(grid::MultiRegionGrid) = ""
