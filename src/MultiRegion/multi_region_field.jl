
struct MultiRegionField{TX, TY, TZ, G, F, T} <: AbstractField{TX, TY, TZ, G, T, 3}
    multi_grid :: G
    local_fields :: F

    function MultiRegionField{TX, TY, TZ}(multi_grid::G, local_fields::F) where {TX, TY, TZ, G, F}
        T = eltype(multi_grid)
        return new{TX, TY, TZ, G, F, T}(multi_grid, local_fields)
    end
end