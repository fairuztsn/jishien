from torch_geometric.nn.conv import RGCNConv
class RGCN(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.atom_embedding=nn.Embedding(len(ATOMIC_SYMBOL), 15)
        self.num_relation=len(BONDTYPE)
        rgcns=[]
        for i in range(len(layers)-1):
            rgcns.append(RGCNConv(layers[i], layers[i+1], self.num_relation))
        self.rgcns = nn.ModuleList(rgcns)
    def forward(self, x, edge_index, edge_type, batch=None):
        h=self.atom_embedding(x)
        if len(edge_type.shape)>1:
            edge_type = edge_type[:,0].long()
        for i,rgcn in enumerate(self.rgcns):
            h=rgcn(h, edge_index, edge_type)
            if i+1<len(self.rgcns):
                h=h.tanh()
        if batch is not None:
            h= scatter(h,batch,dim=0,reduce='mean').mean(-1)
        else:
            h= h.mean((-2,-1))
        return h#*width+offset
