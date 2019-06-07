using Printf

using CuArrays
import MPI

using Oceananigans

# Source: https://github.com/JuliaCI/BenchmarkTools.jl/blob/master/src/trials.jl
function prettytime(t)
    if t < 1e3
        value, units = t, "ns"
    elseif t < 1e6
        value, units = t / 1e3, "μs"
    elseif t < 1e9
        value, units = t / 1e6, "ms"
    else
        s = t / 1e9
        if s < 60
            value, units = s, "s"
        else
            value, units = (s / 60), "min"
        end
    end
    return string(@sprintf("%.3f", value), " ", units)
end

function prettybandwidth(b)
    if b < 1024
        val, units = b, "B/s"
    elseif b < 1024^2
        val, units = b / 1024, "KiB/s"
    elseif b < 1024^3
        val, units = b / 1024^2, "MiB/s"
    else
        val, units = b / 1024^3, "GiB/s"
    end
    return string(@sprintf("%.3f", val), " ", units)
end

@inline index2rank(I, J, Mx, My) = J*My + I
@inline rank2index(r, Mx, My) = mod(r, Mx), div(r, My)

@inline north_halo(tile) = @views @inbounds tile.data.parent[1:tile.grid.Hx, :, :]
@inline south_halo(tile) = @views @inbounds tile.data.parent[tile.grid.Nx+tile.grid.Hx+1:tile.grid.Nx+2tile.grid.Hx, :, :]
@inline  west_halo(tile) = @views @inbounds tile.data.parent[:, 1:tile.grid.Hy, :]
@inline  east_halo(tile) = @views @inbounds tile.data.parent[:, tile.grid.Ny+tile.grid.Hy+1:tile.grid.Ny+2tile.grid.Hy, :]

@inline north_data(tile) = @views @inbounds tile.data.parent[1+tile.grid.Hx:2tile.grid.Hx,   :, :]
@inline south_data(tile) = @views @inbounds tile.data.parent[tile.grid.Nx+1:tile.grid.Nx+tile.grid.Hx, :, :]
@inline  west_data(tile) = @views @inbounds tile.data.parent[:, 1+tile.grid.Hy:2tile.grid.Hy,   :]
@inline  east_data(tile) = @views @inbounds tile.data.parent[:, tile.grid.Ny+1:tile.grid.Ny+tile.grid.Hy, :]

@inline distribute_tag(rank) = 100 + rank
@inline  send_west_tag(rank) = 200 + rank
@inline  send_east_tag(rank) = 300 + rank
@inline send_north_tag(rank) = 400 + rank
@inline send_south_tag(rank) = 500 + rank

function send_halo_data(tile, Mx, My, comm)
    rank = MPI.Comm_rank(comm)

    I, J = rank2index(rank, Mx, My)
    I⁻, I⁺ = mod(I-1, Mx), mod(I+1, Mx)
    J⁻, J⁺ = mod(J-1, My), mod(J+1, My)

    north_rank = index2rank(I,  J⁻, Mx, My)
    south_rank = index2rank(I,  J⁺, Mx, My)
    east_rank  = index2rank(I⁺, J,  Mx, My)
    west_rank  = index2rank(I⁻, J,  Mx, My)

    west_data_buf = zeros(size(west_data(tile)))
    east_data_buf = zeros(size(east_data(tile)))
   north_data_buf = zeros(size(north_data(tile)))
   south_data_buf = zeros(size(south_data(tile)))

    west_data_buf .= copy(west_data(tile))
    east_data_buf .= copy(east_data(tile))
   north_data_buf .= copy(north_data(tile))
   south_data_buf .= copy(south_data(tile))

   se_req = MPI.Isend(east_data_buf,  east_rank,  send_east_tag(rank),  comm)
   sw_req = MPI.Isend(west_data_buf,  west_rank,  send_west_tag(rank),  comm)
   sn_req = MPI.Isend(north_data_buf, north_rank, send_north_tag(rank), comm)
   ss_req = MPI.Isend(south_data_buf, south_rank, send_south_tag(rank), comm)

   @debug "[rank $rank] sending #$(send_east_tag(rank)) to rank $east_rank"
   @debug "[rank $rank] sending #$(send_west_tag(rank)) to rank $west_rank"
   @debug "[rank $rank] sending #$(send_north_tag(rank)) to rank $north_rank"
   @debug "[rank $rank] sending #$(send_south_tag(rank)) to rank $south_rank"
end

function receive_halo_data(tile, Mx, My, comm)
    rank = MPI.Comm_rank(comm)

    I, J = rank2index(rank, Mx, My)
    I⁻, I⁺ = mod(I-1, Mx), mod(I+1, Mx)
    J⁻, J⁺ = mod(J-1, My), mod(J+1, My)

    north_rank = index2rank(I,  J⁻, Mx, My)
    south_rank = index2rank(I,  J⁺, Mx, My)
    east_rank  = index2rank(I⁺, J,  Mx, My)
    west_rank  = index2rank(I⁻, J,  Mx, My)
    
    west_halo_buf = zeros(size(west_halo(tile)))
    east_halo_buf = zeros(size(east_halo(tile)))
   north_halo_buf = zeros(size(north_halo(tile)))
   south_halo_buf = zeros(size(south_halo(tile)))

   re_req = MPI.Irecv!(west_halo_buf,  west_rank,  send_east_tag(west_rank),  comm)
   rw_req = MPI.Irecv!(east_halo_buf,  east_rank,  send_west_tag(east_rank),  comm)
   rn_req = MPI.Irecv!(south_halo_buf, south_rank, send_north_tag(south_rank), comm)
   rs_req = MPI.Irecv!(north_halo_buf, north_rank, send_south_tag(north_rank), comm)

   @debug "[rank $rank] waiting for #$(send_east_tag(west_rank)) from rank $west_rank..."
   @debug "[rank $rank] waiting for #$(send_west_tag(east_rank)) from rank $east_rank..."
   @debug "[rank $rank] waiting for #$(send_north_tag(south_rank)) from rank $south_rank..."
   @debug "[rank $rank] waiting for #$(send_south_tag(north_rank)) from rank $north_rank..."

   MPI.Waitall!([re_req, rw_req, rn_req, rs_req])

     east_halo(tile) .= CuArray(east_halo_buf)
     west_halo(tile) .= CuArray(west_halo_buf)
    north_halo(tile) .= CuArray(north_halo_buf)
    south_halo(tile) .= CuArray(south_halo_buf)
end

function fill_halo_regions_mpi!(FT, arch, Nx, Ny, Nz, Mx, My)
    Lx, Ly, Lz = 10, 10, 10

    Nx′, Ny′, Nz′ = Int(Nx/Mx), Int(Ny/My), Nz
    Lx′, Ly′, Lz′ = Lx/Mx, Ly/My, Lz

    comm = MPI.COMM_WORLD
    
    MPI.Barrier(comm)

    rank = MPI.Comm_rank(comm)
       R = MPI.Comm_size(comm)

    I, J = rank2index(rank, Mx, My)
    I⁻, I⁺ = mod(I-1, Mx), mod(I+1, Mx)
    J⁻, J⁺ = mod(J-1, My), mod(J+1, My)
    Nx′, Ny′, Nz′ = Int(Nx/Mx), Int(Ny/My), Nz
    Lx′, Ly′, Lz′ = Lx/Mx, Ly/My, Lz

    north_rank = index2rank(I,  J⁻, Mx, My)
    south_rank = index2rank(I,  J⁺, Mx, My)
    east_rank  = index2rank(I⁺, J,  Mx, My)
    west_rank  = index2rank(I⁻, J,  Mx, My)

    tile_grid = RegularCartesianGrid((Nx′, Ny′, Nz′), (Lx′, Ly′, Lz′))
    tile = CellField(FT, arch, tile_grid)
    
    send_reqs = MPI.Request[]
    if rank == 0
        rands = rand(Nx, Ny, Nz)

        for r in 1:Mx*My-1
            I′, J′ = rank2index(r, Mx, My)
            i1, i2 = I′*Nx′+1, (I′+1)*Nx′
            j1, j2 = J′*Ny′+1, (J′+1)*Ny′
            send_mesg = rands[i1:i2, j1:j2, :]

            println("[rank $rank] Sending rands[$i1:$i2, $j1:$j2, :] to rank $r...")
            sreq = MPI.Isend(send_mesg, r, distribute_tag(r), comm)
            push!(send_reqs, sreq)
        end

        data(tile) .= rands[1:Nx′, 1:Ny′, :]

        MPI.Waitall!(send_reqs)
    end

    if rank != 0
        println("[rank $rank] Receiving tile from rank 0...")
        recv_mesg = zeros(FT, Nx′, Ny′, Nz′)
        rreq = MPI.Irecv!(recv_mesg, 0, distribute_tag(rank), comm)

        stats = MPI.Wait!(rreq)
        data(tile) .= recv_mesg
    end
    
    println("[rank $rank] Sending halo data...")
    send_halo_data(tile, Mx, My, comm)

    println("[rank $rank] Receiving halo data...")
    receive_halo_data(tile, Mx, My, comm)
    
    MPI.Barrier(comm)
    if rank == 0
        tic = time_ns() 
    end

    println("[rank $rank] Sending halo data...")
    send_halo_data(tile, Mx, My, comm)

    println("[rank $rank] Receiving halo data...")
    receive_halo_data(tile, Mx, My, comm)

	MPI.Barrier(comm)
	if rank == 0
		t = (time_ns() - tic)
		ts = t / 1e9
        @info "$R ranks halo communication time: $(prettytime(t))"
        
        Hx, Hy = 1, 1
        data_size = sizeof(FT) * 2Nz*(Hx*Nx + Hy*Ny)
        @info "$R ranks halo communication bandwidth: $(prettybandwidth(data_size/ts))"
	end
end

MPI.Init()
fill_halo_regions_mpi!(Float64, GPU(), 192, 192, 192, 3, 3)
MPI.Finalize()
