import MPI

using Oceananigans

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

function send_halo_data(tile)
    rank = MPI.Comm_rank(comm)

    I, J = rank2index(rank, Mx, My)
    I⁻, I⁺ = mod(I-1, Mx), mod(I+1, Mx)
    J⁻, J⁺ = mod(J-1, My), mod(J+1, My)
    Nx′, Ny′, Nz′ = Int(Nx/Mx), Int(Ny/My), Nz
    Lx′, Ly′, Lz′ = Lx/Mx, Ly/My, Lz

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

   MPI.Waitall!([se_req, sw_req, sn_req, ss_req])
end

function receive_halo_data(tile)
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

    east_halo(tile) .=  east_halo_buf
    west_halo(tile) .=  west_halo_buf
   north_halo(tile) .= north_halo_buf
   south_halo(tile) .= south_halo_buf
end

function fill_halo_regions_mpi!(FT, arch, Nx, Ny, Nz, Mx, My)
    Lx, Ly, Lz = 10, 10, 10

    Nx′, Ny′, Nz′ = Int(Nx/Mx), Int(Ny/My), Nz
    Lx′, Ly′, Lz′ = Lx/Mx, Ly/My, Lz

    comm = MPI.COMM_WORLD

    MPI.Barrier(comm)

    rank = MPI.Comm_rank(comm)

    I, J = rank2index(rank, Mx, My)
    I⁻, I⁺ = mod(I-1, Mx), mod(I+1, Mx)
    J⁻, J⁺ = mod(J-1, My), mod(J+1, My)
    Nx′, Ny′, Nz′ = Int(Nx/Mx), Int(Ny/My), Nz
    Lx′, Ly′, Lz′ = Lx/Mx, Ly/My, Lz

    north_rank = index2rank(I,  J⁻, Mx, My)
    south_rank = index2rank(I,  J⁺, Mx, My)
    east_rank  = index2rank(I⁺, J,  Mx, My)
    west_rank  = index2rank(I⁻, J,  Mx, My)

    send_reqs = MPI.Request[]
    if rank == 0
        rands = rand(Nx, Ny, Nz)

        for r in 0:Mx*My-1
            I′, J′ = rank2index(r, Mx, My)
            i1, i2 = I′*Nx′+1, (I′+1)*Nx′
            j1, j2 = J′*Ny′+1, (J′+1)*Ny′
            send_mesg = rands[i1:i2, j1:j2, :]

            println("[rank $rank] Sending R[$i1:$i2, $j1:$j2, :] to rank $r...")
            sreq = MPI.Isend(send_mesg, r, distribute_tag(r), comm)
            push!(send_reqs, sreq)
        end

        MPI.Waitall!(send_reqs)
    end

    tile_grid = RegularCartesianGrid((Nx′, Ny′, Nz′), (Lx′, Ly′, Lz′))
    tile = CellField(FT, arch, tile_grid)

    println("[rank $rank] Receiving tile from rank 0...")
    recv_mesg = zeros(FT, Nx′, Ny′, Nz′)
    rreq = MPI.Irecv!(recv_mesg, 0, distribute_tag(rank), comm)

    stats = MPI.Wait!(rreq)
    data(tile) .= recv_mesg

    println("[rank $rank] Sending halo data...")
    send_halo_data(tile)

    println("[rank $rank] Receiving halo data...")
    receive_halo_data(tile)

    println("[rank $rank] Sending halo data...")
    send_halo_data(tile)

    println("[rank $rank] Receiving halo data...")
    receive_halo_data(tile)

    if rank == 3
        display(tile.data)
    end
end

MPI.Init()
fill_halo_regions_mpi!(Float64, CPU(), 16, 16, 16, 2, 2)
MPI.Finalize()
