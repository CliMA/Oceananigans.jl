using Oceananigans.ImmersedBoundaries
using Oceananigans.Grids: get_active_cells_map, active_cell
using Oceananigans.Architectures: CPU
import Oceananigans.Architectures as AC
using Oceananigans.Fields: Field, interior
using KernelAbstractions: @kernel, @index

function generate_condition_maps(grid,
                                 advection;
                                 condition_momentum_advection=false,
                                 condition_tracer_advection=false)

    active_cells_map = get_active_cells_map(grid, Val(:interior))

    map_keys = keys(advection)
    condition_maps = NamedTuple{map_keys}

    for key in keys(advection)
      if key == :momentum
        if condition_momentum_advection
          condition_maps[:momentum] = compute_advection_conditioned_map(advection.momentum,
                                                                        grid;
                                                                        active_cells_map)
        end
      else
        if condition_tracer_advection
          condition_maps[key] = compute_advection_conditioned_map(advection[:key],
                                                                  grid;
                                                                  active_cells_map)
        end
      end
    end 

    return condition_maps
end

function compute_advection_conditioned_map(scheme,
                                           grid::ImmersedBoundaryGrid;
                                           active_cells_map=nothing)
    # Field is true if the max scheme can be used for computing u advection
    max_scheme_field = Field{Center, Center, Center}(grid, Bool)
    fill!(max_scheme_field, false)
    launch!(architecture(grid),
            grid,
            :xyz,
            condition_map!,
            max_scheme_field,
            grid,
            scheme;
            active_cells_map)

    return NamedTuple{(:interior, :boundary)}(split_indices(max_scheme_field, grid; active_cells_map))
end

@kernel function condition_map!(max_scheme_field, ibg, scheme)
    i, j, k = @index(Global, NTuple)

    @inbounds max_scheme_field[i, j, k] = check_interior_xyz(i, j, k, ibg, scheme)
end

function split_indices(field, grid; active_cells_map=nothing)
    if isnothing(active_cells_map)
        return split_indices_full(field, grid)
    else
        return split_indices_mapped(field, grid, active_cells_map)
    end
end


check_interior_xyz(i, j, k, ibg, scheme) = reduce(&, 
                                                 (check_interior_x(i, j, k, ibg, scheme),
                                                  check_interior_y(i, j, k, ibg, scheme),
                                                  check_interior_z(i, j, k, ibg, scheme)))

function check_interior_x(i, j, k, ibg, scheme::AbstractAdvectionScheme{N}) where N
    interior = true
    
    buffer = N + 1
    for di in -buffer:buffer
        interior &= active_cell(i + di, j, k, ibg)
    end
    return interior
end

function check_interior_y(i, j, k, ibg, scheme::AbstractAdvectionScheme{N}) where N
    interior = true
    
    buffer = N + 1
    for dj in -buffer:buffer
        interior &= active_cell(i, j + dj, k, ibg)
    end
    return interior
end

function check_interior_z(i, j, k, ibg, scheme::AbstractAdvectionScheme{N}) where N
    interior = true
    
    buffer = N + 1
    for dk in -buffer:buffer
        interior &= active_cell(i, j, k + dk, ibg)
    end
    return interior
end

function split_indices_mapped(field, grid, active_cells_map)
    IndexType = Tuple{Int64,Int64,Int64}
    vals = AC.on_architecture(CPU(), interior(field))
    active_indices = AC.on_architecture(CPU(), active_cells_map)
    map1 = Vector{eltype(active_indices)}()
    map2 = Vector{eltype(active_indices)}()
    for index in active_indices
        val = vals[convert(IndexType, index)...]
	if val
	    push!(map1, index)
	else
	    push!(map2, index)
	end
    end
    map1 = AC.on_architecture(architecture(grid), map1) 
    map2 = AC.on_architecture(architecture(grid), map2) 
    return (map1, map2)
end

function split_indices_full(field, grid)
    IndicesType = Tuple{Int16, Int16, Int16}
    map1 = IndicesType[]
    map2 = IndicesType[]
    for k in 1:size(grid, 3)
        vals = AC.on_architecture(CPU(), interior(field, :, :, k)) 
	map1 = vcat(map1, convert_interior_indices(findall(x->x, vals), k, IndicesType))
	map2 = vcat(map2, convert_interior_indices(findall(x->!x, vals), k, IndicesType))
	GC.gc()
    end
    map1 = AC.on_architecture(architecture(grid), map1) 
    map2 = AC.on_architecture(architecture(grid), map2) 
    return (map1, map2)
end


function convert_interior_indices(interior_indices, k, IndicesType)
    interior_indices =   getproperty.(interior_indices, :I)
    interior_indices = add_3rd_index.(interior_indices, k) |> Array{IndicesType}
    return interior_indices
end

add_3rd_index(ij, k) = (ij[1], ij[2], k)

