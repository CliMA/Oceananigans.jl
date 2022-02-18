import Oceananigans.Fields: set!

struct MultiRegionField{TX, TY, TZ, G, F, T} <: AbstractMultiField{TX, TY, TZ, G, F, T}
    multi_grid :: G
    local_fields :: F

    function MultiRegionField{TX, TY, TZ}(multi_grid::G, local_fields::F) where {TX, TY, TZ, G, F}
        T = eltype(multi_grid)
        return new{TX, TY, TZ, G, F, T}(multi_grid, local_fields)
    end
end

@inline assoc_field(mrf::MultiRegionField, idx)  = mrf.local_fields[idx]
@inline assoc_device(mrf::MultiRegionField, idx) = assoc_device(mrf.multi_grid, idx)
@inline regions(mrf::MultiRegionField)           = regions(mrf.multi_grid)
@inline fields(mrf::MultiRegionField)            = mrf.local_fields

function MultiRegionField(loc::Tuple,
                          mrg::MultiRegionGrid)
    
    loc_fields = multi_region_object(mrg, Field, (loc, grids(mrg)), (0, 1))
    return MultiRegionField{loc[1], loc[2], loc[3]}(mrg, loc_fields)
end

Base.show(io::IO, mrf::MultiRegionField{LX, LY, LZ}) where {LX, LY, LZ} =  
    print(io, "MultiRegionalField on ($LX, $LY, $LZ): \n",
              "├── grid: $(summary(mrf.multi_grid)) \n",
              "└── total: $(length(fields(mrf))) fields on $(arch_summary(mrf.multi_grid))\n",
              "     └── field: $(summary(mrf.local_fields[1]))")

@inline set!(mrf::MultiRegionField, f::Function) = multi_region_function!(mrf, set!, (fields(mrf), f), (1, 0))