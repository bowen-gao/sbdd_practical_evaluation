from ssl import ALERT_DESCRIPTION_UNEXPECTED_MESSAGE
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionCore(nn.Module):
    def __init__(self, config, fusion_type,h_dim,ret_feat_dim,n_heads):
        super(CrossAttentionCore, self).__init__()
        self.config=config
        self.fusion_type=fusion_type
        self.h_dim=h_dim
        self.ret_feat_dim=ret_feat_dim
        self.n_heads=n_heads
        self.linear_query = nn.Linear(ret_feat_dim, h_dim)
        self.attention = nn.MultiheadAttention(h_dim, n_heads)
        if "cat" in fusion_type:
            self.linear_cat=nn.Linear(h_dim+h_dim,h_dim)

    def forward(self, h, ret_feat,batch,mask_ligand):
        # h is the [all_atom_num, h_dim] tensor, batch is the [all_atom_num] tensor, indicating the batch id of each atom

        # get the batch id of each atom
        batch_size = batch.max().item() + 1
        num_atom_each_batch = torch.bincount(batch).to(h.device)
        h_list=list(h.split(num_atom_each_batch.tolist(), dim=0))
        # append to same length
        max_num_atom = max(num_atom_each_batch)
        for i in range(batch_size):
            h_list[i] = torch.cat([h_list[i], torch.zeros(max_num_atom - num_atom_each_batch[i], h.shape[1]).to(h.device)], dim=0)
        h_tensor = torch.stack(h_list, dim=0)
        # h_tensor is the [batch_size, max_num_atom, h_dim] tensor
        arr = torch.arange(max_num_atom).to(h_tensor.device)
        arr_expand=arr.expand(batch_size, max_num_atom)
        mask = arr_expand >= num_atom_each_batch.unsqueeze(1)

        query = h_tensor.permute(1, 0, 2)
        key_value = self.linear_query(ret_feat).permute(1, 0, 2)
        attention_output, attn_weight = self.attention(query, key_value, key_value)
        if hasattr(self.config,"debug_attn_weight") and self.config.debug_attn_weight==True:
            torch.save(attn_weight, '/nfs/data/targetdiff_data/attn_weight.pt')
            print("attn_weight has been saved to /nfs/data/targetdiff_data/attn_weight.pt")
            exit()
        # attention_output is the [max_num_atom, batch_size, h_dim] tensor
        # reverse the padding
        attention_output = attention_output.permute(1, 0, 2).reshape(-1, self.h_dim)
        # attention_output is the [all_atom_num, h_dim] tensor
        mask=mask.view(-1)
        # mask is the [all_atom_num] tensor
        output=attention_output[~mask]

        if self.fusion_type=="xattn_cat":
            output=self.linear_cat(torch.cat([h,output],dim=1))

        return output

class CrossAttentionRespectivelyModule(nn.Module):
    def __init__(self, config, fusion_type,h_dim,ret_feat_dim,n_heads):
        super(CrossAttentionRespectivelyModule, self).__init__()
        self.config=config
        self.fusion_type=fusion_type
        self.h_dim=h_dim
        self.ret_feat_dim=ret_feat_dim
        self.n_heads=n_heads

        # Hardcode ret_feat split here
        if ret_feat_dim==512:
            self.pocket_ret_feat_dim=0
            self.ligand_ret_feat_dim=512
        elif ret_feat_dim==1024:
            self.pocket_ret_feat_dim=512
            self.ligand_ret_feat_dim=512
        else:
            raise NotImplementedError

        if self.pocket_ret_feat_dim != 0:
            self.pocket_attention=CrossAttentionCore(config,fusion_type,h_dim,self.pocket_ret_feat_dim,n_heads)
        if self.ligand_ret_feat_dim != 0:
            self.ligand_attention=CrossAttentionCore(config,fusion_type,h_dim,self.ligand_ret_feat_dim,n_heads)

    def forward(self,h,ret_feat,batch,mask_ligand):
        # h is the [all_atom_num, h_dim] tensor, batch is the [all_atom_num] tensor, indicating the batch id of each atom
        # ret_feat is the [batch_size, 10 ,ret_feat_dim] tensor
        # mask_ligand is the [all_atom_num] tensor, indicating whether the atom is in the ligand or not

        # split h into pocket and ligand
        pocket_h=h[~mask_ligand]
        ligand_h=h[mask_ligand]

        # split ret_feat into pocket and ligand
        if self.pocket_ret_feat_dim != 0:
            pocket_ret_feat,ligand_ret_feat=torch.split(ret_feat,self.pocket_ret_feat_dim,dim=2)
        else:
            ligand_ret_feat=ret_feat

        if self.pocket_ret_feat_dim != 0:
            pocket_h=self.pocket_attention(pocket_h,pocket_ret_feat,batch[~mask_ligand],mask_ligand[~mask_ligand])
        if self.ligand_ret_feat_dim != 0:
            ligand_h=self.ligand_attention(ligand_h,ligand_ret_feat,batch[mask_ligand],mask_ligand[mask_ligand])

        # concat pocket and ligand
        h_new=h.clone()
        h_new[~mask_ligand]=pocket_h
        h_new[mask_ligand]=ligand_h

        return h
        

class CrossAttentionModule(nn.Module):
    def __init__(self, config, fusion_type,h_dim,ret_feat_dim,n_heads):
        super(CrossAttentionModule, self).__init__()
        self.config=config
        self.fusion_type=fusion_type
        self.h_dim=h_dim
        self.ret_feat_dim=ret_feat_dim
        self.n_heads=n_heads
        if not "respectively" in fusion_type:
            self.cross_attention=CrossAttentionCore(config,fusion_type,h_dim,ret_feat_dim,n_heads)
        else:
            self.cross_attention=CrossAttentionRespectivelyModule(config,fusion_type,h_dim,ret_feat_dim,n_heads)

    def forward(self,h,ret_feat,batch,mask_ligand):
        return self.cross_attention(h,ret_feat,batch,mask_ligand)

class CrossAttentionFusionModules(nn.Module):
    def __init__(self, config,fusion_type,fusion_pos ,h_dim, ret_feat_dim, n_heads,num_gnn_layers):
        super(CrossAttentionFusionModules, self).__init__()
        self.config=config
        self.fusion_type=fusion_type
        self.h_dim=h_dim
        self.ret_feat_dim=ret_feat_dim
        self.n_heads=n_heads
        self.fusion_pos=fusion_pos
        self.num_gnn_layers=num_gnn_layers
        
        if "init" in fusion_pos:
            self.init_cross_attention_module=CrossAttentionModule(config,self.fusion_type,h_dim,ret_feat_dim,n_heads)
        
        if "all_layers" in fusion_pos or "a_third_layers" in fusion_pos:
            self.cross_attention_modules=nn.ModuleList([CrossAttentionModule(config,self.fusion_type,h_dim,ret_feat_dim,n_heads) for i in range(num_gnn_layers)])

        if "ending" in fusion_pos:
            self.ending_cross_attention_module=CrossAttentionModule(config,self.fusion_type,h_dim,ret_feat_dim,n_heads)

    def forward(self,h,ret_feat,batch,mask_ligand,cur_pos):
        if cur_pos=="init":
            if "init" in self.fusion_pos:
                return self.init_cross_attention_module(h,ret_feat,batch,mask_ligand)
            else :
                return h
        elif cur_pos=="ending":
            if "ending" in self.fusion_pos:
                return self.ending_cross_attention_module(h,ret_feat,batch,mask_ligand)
            else :
                return h
        else:
            # fusion_pos is layer number
            if "all_layers" in self.fusion_pos:
                return self.cross_attention_modules[cur_pos](h,ret_feat,batch,mask_ligand)
            elif "a_third_layers" in self.fusion_pos:
                if int(cur_pos)%3==0:
                    return self.cross_attention_modules[cur_pos](h,ret_feat,batch,mask_ligand)
                else:
                    return h
            else:
                return h
            

class Concatenation(nn.Module):
    def __init__(self,config,  h_dim, ret_feat_dim, n_heads,fusion_pos=None):
        super(Concatenation, self).__init__()
        if fusion_pos is not None:
            raise NotImplementedError
        self.config=config
        self.h_dim=h_dim
        self.ret_feat_dim=ret_feat_dim
        self.n_heads=n_heads
        self.ret_proj=nn.Linear(ret_feat_dim, h_dim)
        self.linear=nn.Linear(h_dim+h_dim,h_dim)

    def forward(self, h, ret_feat,batch):
        # h is the [all_atom_num, h_dim] tensor, batch is the [all_atom_num] tensor, indicating the batch id of each atom
        # ret_feat is the [batch_size, 10 ,ret_feat_dim] tensor

        # get the batch id of each atom
        batch_size = batch.max().item() + 1
        num_atom_each_batch = torch.bincount(batch).to(h.device)
        # pooling
        ret_feat=ret_feat.mean(dim=1)
        ret_feat=self.ret_proj(ret_feat)
        for i in range(batch_size):
            h[batch==i]=self.linear(torch.cat([h[batch==i],ret_feat[i].repeat(num_atom_each_batch[i],1)],dim=1))
        return h