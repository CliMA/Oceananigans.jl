using JLD2
using Plots
#
#
# Read in saved times series of rhs and solution (η).
# Form consistent arrays and plot
#

# Halo width
hw=3;
η_fk="η";
rhs_fk="rhs";

fh=jldopen("tdata.jld2","r");
fk=keys(fh);

# Get sizes
k1=fk[1];
η=fh[k1][k1][η_fk][1+hw:end-hw,1+hw:end-hw];
nx=size(η)[1]
ny=size(η)[2]
nrec=size(fk)[1]

# Now read some records
net_input =zeros(nx,ny,nrec);
net_output=zeros(nx,ny,nrec);
i=0;
for k in fk
    global i=i+1;
    net_input[:,:,i] =fh[k][k][η_fk][1+hw:end-hw,1+hw:end-hw];
    net_output[:,:,i]=reshape(fh[k][k][rhs_fk],(nx,ny));
end     
# infld
#
for i in 1:nrec
    p1=heatmap( net_input[:,:,i],title="Input $i")
    p2=heatmap(net_output[:,:,i],title="Output $i")
    plot(p1,p2)
    fname="plot_$i.png"
    savefig(fname)
end
