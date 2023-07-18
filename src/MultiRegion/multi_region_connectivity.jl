"""
    struct RegionalConnectivity{S <: AbstractRegionSide, FS <: AbstractRegionSide} <: AbstractConnectivity

The connectivity among various regions in a multi-region partition.

$(TYPEDFIELDS)
"""
struct RegionalConnectivity{S <: AbstractRegionSide, FS <: AbstractRegionSide} <: AbstractConnectivity
    "the current region rank"
         rank :: Int
    "the region from which boundary condition comes from"
    from_rank :: Int
    "the current region side"
         side :: S
    "the side of the region from which boundary condition comes from"
    from_side :: FS
end
