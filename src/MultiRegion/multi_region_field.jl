import Oceananigans.Fields: set!, validate_field_data

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
    
    args       = (loc, grids(mrg))
    iter_args  = (0, 1)
    loc_fields = multi_region_object(mrg, Field, args, iter_args, nothing, nothing)
    return MultiRegionField{loc[1], loc[2], loc[3]}(mrg, loc_fields)
end

# function MultiRegionField(loc::Tuple,
#                           mrg::MultiRegionGrid;
#                           boundary_conditions = FieldBoundaryConditions(mrg, loc))

#     args        = (regions(mrg), partition(mrg))
#     iter_args   = (1, 0)
#     kwargs      = (bcs = boundary_conditions, ) 
#     iter_kwargs = (0, )
#     loc_bcs     = multi_region_object(mrg, inject_multi_bc, args, iter_args, kwargs, iter_kwargs) 
#     args        = (loc, grids(mrg))
#     iter_args   = (0, 1)
#     kwargs      = (boundary_conditions = loc_bcs, )
#     iter_kwargs = Tuple(1)
#     loc_fields  = multi_region_object(mrg, Field, args, iter_args, kwargs, iter_kwargs)

#     return MultiRegionField{loc[1], loc[2], loc[3]}(mrg, loc_fields)
# end

Base.show(io::IO, mrf::MultiRegionField{LX, LY, LZ}) where {LX, LY, LZ} =  
    print(io, "MultiRegionField on ($LX, $LY, $LZ): \n",
              "├── grid: $(summary(mrf.multi_grid)) \n",
              "└── total: $(length(fields(mrf))) fields on $(arch_summary(mrf.multi_grid))\n",
              "     └── field: $(summary(mrf.local_fields[1]))")

@inline set!(mrf::MultiRegionField, f::Function) = multi_region_function!(mrf, set!, (fields(mrf), f), (1, 0), nothing, nothing)

function apply_regionally!(mrg, func, args...; kwargs...)
    for r in regions(mrg)
        switch_device!(mrg, r)
        region_args = Tuple(getregion(arg, r) for arg in args)
        region_kwargs = Tuple(getregion(kwarg, r) for kwarg in kwargs)
        func(region_args...; region_kwargs...)
    end
end

getregion(f, i) = i

f :: MultiField = Field{<:Any, <:Any, <:Any, <:MultiRegionGrid}

function getregion(f :: MultiField{LX, LY, LZ}, i) where {LX, LY, LZ} 
    switch_device!(f.grid, i)
    return Field((LX, LY, LZ), getregion(f.grid, i), getregion(f.boundary_conditions, i))
end

new_data(FT, grid, loc) = apply_regionally!(grid, new_data, FT, grid.regional_grids, loc)
set!(f, func) = apply_regionally!(f.grid, set!, f, func)

function 