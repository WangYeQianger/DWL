#!usr/bin/env python3
# -*- coding:utf-8 -*-
# @time : 2022/8/8 14:33
# @author : Xujun Zhang, Tianyue Wang

from copy import deepcopy
import torch
import networkx as nx
import numpy as np
import copy
from torch_geometric.utils import to_networkx
from rdkit import Chem
import warnings
from rdkit.Chem import AllChem
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from collections import OrderedDict

bb_order_2_parent = {
    'N':'C',
    'CA': 'N',
    'C': 'CA',
    'O': 'C',                 
                     }
bb_order_2_parent_reverse = {
    'C':'N',
    'O': 'C',
    'CA': 'C',
    'N': 'CA',                 
                     }

bb_order_2_3parent = {
    'N':['C', 'CA', 'O'],
    'CA': ['N', 'C', 9999],
    'C': ['CA', 'N', 9999],
    'O': ['C', 'CA', 9999],                 
                     }
bb_order_2_3parent_reverse = {
    'C':'N',
    'O': 'C',
    'CA': 'C',
    'N': 'CA',                 
                     }

def get_transformation_mask(pyg_data):
    G = to_networkx(pyg_data, to_undirected=False)
    to_rotate = []
    edges = pyg_data.edge_index.T.numpy()
    for i in range(0, edges.shape[0], 2):
        assert edges[i, 0] == edges[i+1, 1]

        G2 = G.to_undirected()
        G2.remove_edge(*edges[i])
        if not nx.is_connected(G2):
            l = list(sorted(nx.connected_components(G2), key=len)[0])
            if len(l) > 1:
                if edges[i, 0] in l:
                    to_rotate.append([])
                    to_rotate.append(l)
                else:
                    to_rotate.append(l)
                    to_rotate.append([])
                continue
        to_rotate.append([])
        to_rotate.append([])

    mask_edges = np.asarray([0 if len(l) == 0 else 1 for l in to_rotate], dtype=bool)
    mask_rotate = np.zeros((np.sum(mask_edges), len(G.nodes())), dtype=bool)
    idx = 0
    for i in range(len(G.edges())):
        if mask_edges[i]:
            mask_rotate[idx][np.asarray(to_rotate[i], dtype=int)] = True
            idx += 1

    return mask_edges, mask_rotate

def binarize(x):
    return torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x))

def get_higher_order_adj_matrix(adj, order):
    """
    Args:
        adj:        (N, N)
        type_mat:   (N, N)
    Returns:
        Following attributes will be updated:
            - edge_index
            - edge_type
        Following attributes will be added to the data object:
            - bond_edge_index:  Original edge_index.
    """
    adj_mats = [torch.eye(adj.size(0), dtype=torch.long, device=adj.device), \
                binarize(adj + torch.eye(adj.size(0), dtype=torch.long, device=adj.device))]

    for i in range(2, order + 1):
        adj_mats.append(binarize(adj_mats[i - 1] @ adj_mats[1]))
    order_mat = torch.zeros_like(adj)

    for i in range(1, order + 1):
        order_mat += (adj_mats[i] - adj_mats[i - 1]) * i

    return order_mat

def get_ring_adj_matrix(mol, adj):
    new_adj = deepcopy(adj)
    ssr = Chem.GetSymmSSSR(mol)
    for ring in ssr:
        # print(ring)
        for i in ring:
            for j in ring:
                if i==j:
                    continue
                elif new_adj[i][j] != 1:
                    new_adj[i][j]+=1
    return new_adj




# orderd by perodic table
atom_vocab = ["H", "B", "C", "N", "O", "F", "Mg", "Si", "P", "S", "Cl", "Cu", "Zn", "Se", "Br", "Sn", "I"]
atom_vocab = {a: i for i, a in enumerate(atom_vocab)}
degree_vocab = range(7)
num_hs_vocab = range(7)
formal_charge_vocab = range(-5, 6)
chiral_tag_vocab = range(4)
total_valence_vocab = range(8)
num_radical_vocab = range(8)
hybridization_vocab = range(len(Chem.rdchem.HybridizationType.values))

bond_type_vocab = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
                   Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
bond_type_vocab = {b: i for i, b in enumerate(bond_type_vocab)}
bond_dir_vocab = range(len(Chem.rdchem.BondDir.values))
bond_stereo_vocab = range(len(Chem.rdchem.BondStereo.values))

# orderd by molecular mass
residue_vocab = ["GLY", "ALA", "SER", "PRO", "VAL", "THR", "CYS", "ILE", "LEU", "ASN",
                 "ASP", "GLN", "LYS", "GLU", "MET", "HIS", "PHE", "ARG", "TYR", "TRP"]


def onehot(x, vocab, allow_unknown=False):
    if x in vocab:
        if isinstance(vocab, dict):
            index = vocab[x]
        else:
            index = vocab.index(x)
    else:
        index = -1
    if allow_unknown:
        feature = [0] * (len(vocab) + 1)
        if index == -1:
            warnings.warn("Unknown value `%s`" % x)
        feature[index] = 1
    else:
        feature = [0] * len(vocab)
        if index == -1:
            raise ValueError("Unknown value `%s`. Available vocabulary is `%s`" % (x, vocab))
        feature[index] = 1

    return feature


def atom_default(atom):
    """Default atom feature.

    Features:
        GetSymbol(): one-hot embedding for the atomic symbol dim=18
        
        GetChiralTag(): one-hot embedding for atomic chiral tag dim=5
        
        GetTotalDegree(): one-hot embedding for the degree of the atom in the molecule including Hs dim=5
        
        GetFormalCharge(): one-hot embedding for the number of formal charges in the molecule
        
        GetTotalNumHs(): one-hot embedding for the total number of Hs (explicit and implicit) on the atom 
        
        GetNumRadicalElectrons(): one-hot embedding for the number of radical electrons on the atom
        
        GetHybridization(): one-hot embedding for the atom's hybridization
        
        GetIsAromatic(): whether the atom is aromatic
        
        IsInRing(): whether the atom is in a ring
        18 + 5 + 8 + 12 + 8 + 9 + 10 + 9 + 3 + 4 
    """
    return onehot(atom.GetSymbol(), atom_vocab, allow_unknown=True) + \
           onehot(atom.GetChiralTag(), chiral_tag_vocab, allow_unknown=True) + \
           onehot(atom.GetTotalDegree(), degree_vocab, allow_unknown=True) + \
           onehot(atom.GetFormalCharge(), formal_charge_vocab, allow_unknown=True) + \
           onehot(atom.GetTotalNumHs(), num_hs_vocab, allow_unknown=True) + \
           onehot(atom.GetNumRadicalElectrons(), num_radical_vocab, allow_unknown=True) + \
           onehot(atom.GetHybridization(), hybridization_vocab, allow_unknown=True) + \
            onehot(atom.GetTotalValence(), total_valence_vocab, allow_unknown=True) + \
           [atom.GetIsAromatic(), atom.IsInRing(), atom.IsInRing() and (not atom.IsInRingSize(3)) and (not atom.IsInRingSize(4)) \
            and (not atom.IsInRingSize(5)) and (not atom.IsInRingSize(6))]+[atom.IsInRingSize(i) for i in range(3, 7)]


def atom_center_identification(atom):
    """Reaction center identification atom feature.

    Features:
        GetSymbol(): one-hot embedding for the atomic symbol
        
        GetTotalNumHs(): one-hot embedding for the total number of Hs (explicit and implicit) on the atom 
        
        GetTotalDegree(): one-hot embedding for the degree of the atom in the molecule including Hs
        
        GetTotalValence(): one-hot embedding for the total valence (explicit + implicit) of the atom
        
        GetIsAromatic(): whether the atom is aromatic
        
        IsInRing(): whether the atom is in a ring
    """
    return onehot(atom.GetSymbol(), atom_vocab, allow_unknown=True) + \
           onehot(atom.GetTotalNumHs(), num_hs_vocab) + \
           onehot(atom.GetTotalDegree(), degree_vocab, allow_unknown=True) + \
           onehot(atom.GetTotalValence(), total_valence_vocab) + \
           [atom.GetIsAromatic(), atom.IsInRing()]



def atom_synthon_completion(atom):
    """Synthon completion atom feature.

    Features:
        GetSymbol(): one-hot embedding for the atomic symbol

        GetTotalNumHs(): one-hot embedding for the total number of Hs (explicit and implicit) on the atom 
        
        GetTotalDegree(): one-hot embedding for the degree of the atom in the molecule including Hs
        
        IsInRing(): whether the atom is in a ring
        
        IsInRingSize(3, 4, 5, 6): whether the atom is in a ring of a particular size
        
        IsInRing() and not IsInRingSize(3, 4, 5, 6): whether the atom is in a ring and not in a ring of 3, 4, 5, 6
    """
    return onehot(atom.GetSymbol(), atom_vocab, allow_unknown=True) + \
           onehot(atom.GetTotalNumHs(), num_hs_vocab) + \
           onehot(atom.GetTotalDegree(), degree_vocab, allow_unknown=True) + \
           [atom.IsInRing(), atom.IsInRingSize(3), atom.IsInRingSize(4),
            atom.IsInRingSize(5), atom.IsInRingSize(6), 
            atom.IsInRing() and (not atom.IsInRingSize(3)) and (not atom.IsInRingSize(4)) \
            and (not atom.IsInRingSize(5)) and (not atom.IsInRingSize(6))]



def atom_symbol(atom):
    """Symbol atom feature.

    Features:
        GetSymbol(): one-hot embedding for the atomic symbol
    """
    return onehot(atom.GetSymbol(), atom_vocab, allow_unknown=True)



def atom_explicit_property_prediction(atom):
    """Explicit property prediction atom feature.

    Features:
        GetSymbol(): one-hot embedding for the atomic symbol

        GetDegree(): one-hot embedding for the degree of the atom in the molecule

        GetTotalValence(): one-hot embedding for the total valence (explicit + implicit) of the atom
        
        GetFormalCharge(): one-hot embedding for the number of formal charges in the molecule
        
        GetIsAromatic(): whether the atom is aromatic
    """
    return onehot(atom.GetSymbol(), atom_vocab, allow_unknown=True) + \
           onehot(atom.GetDegree(), degree_vocab, allow_unknown=True) + \
           onehot(atom.GetTotalValence(), total_valence_vocab, allow_unknown=True) + \
           onehot(atom.GetFormalCharge(), formal_charge_vocab) + \
           [atom.GetIsAromatic()]



def atom_property_prediction(atom):
    """Property prediction atom feature.

    Features:
        GetSymbol(): one-hot embedding for the atomic symbol
        
        GetDegree(): one-hot embedding for the degree of the atom in the molecule
        
        GetTotalNumHs(): one-hot embedding for the total number of Hs (explicit and implicit) on the atom 
        
        GetTotalValence(): one-hot embedding for the total valence (explicit + implicit) of the atom
        
        GetFormalCharge(): one-hot embedding for the number of formal charges in the molecule
        
        GetIsAromatic(): whether the atom is aromatic
    """
    return onehot(atom.GetSymbol(), atom_vocab, allow_unknown=True) + \
           onehot(atom.GetDegree(), degree_vocab, allow_unknown=True) + \
           onehot(atom.GetTotalNumHs(), num_hs_vocab, allow_unknown=True) + \
           onehot(atom.GetTotalValence(), total_valence_vocab, allow_unknown=True) + \
           onehot(atom.GetFormalCharge(), formal_charge_vocab, allow_unknown=True) + \
           [atom.GetIsAromatic()]



def atom_position(atom):
    """
    Atom position in the molecular conformation.
    Return 3D position if available, otherwise 2D position is returned.

    Note it takes much time to compute the conformation for large molecules.
    """
    mol = atom.GetOwningMol()
    if mol.GetNumConformers() == 0:
        mol.Compute2DCoords()
    conformer = mol.GetConformer()
    pos = conformer.GetAtomPosition(atom.GetIdx())
    return [pos.x, pos.y, pos.z]



def atom_pretrain(atom):
    """Atom feature for pretraining.

    Features:
        GetSymbol(): one-hot embedding for the atomic symbol
        
        GetChiralTag(): one-hot embedding for atomic chiral tag
    """
    return onehot(atom.GetSymbol(), atom_vocab, allow_unknown=True) + \
           onehot(atom.GetChiralTag(), chiral_tag_vocab)



def atom_residue_symbol(atom):
    """Residue symbol as atom feature. Only support atoms in a protein.

    Features:
        GetSymbol(): one-hot embedding for the atomic symbol
        GetResidueName(): one-hot embedding for the residue symbol
    """
    residue = atom.GetPDBResidueInfo()
    return onehot(atom.GetSymbol(), atom_vocab, allow_unknown=True) + \
           onehot(residue.GetResidueName() if residue else -1, residue_vocab, allow_unknown=True)


def bond_default(bond):
    """Default bond feature.

    Features:
        GetBondType(): one-hot embedding for the type of the bond
        
        GetBondDir(): one-hot embedding for the direction of the bond
        
        GetStereo(): one-hot embedding for the stereo configuration of the bond
        
        GetIsConjugated(): whether the bond is considered to be conjugated
    """
    return onehot(bond.GetBondType(), bond_type_vocab, allow_unknown=True) + \
           onehot(bond.GetBondDir(), bond_dir_vocab) + \
           onehot(bond.GetStereo(), bond_stereo_vocab, allow_unknown=True) + \
           [int(bond.GetIsConjugated())]



def bond_length(bond):
    """
    Bond length in the molecular conformation.

    Note it takes much time to compute the conformation for large molecules.
    """
    mol = bond.GetOwningMol()
    if mol.GetNumConformers() == 0:
        mol.Compute2DCoords()
    conformer = mol.GetConformer()
    h = conformer.GetAtomPosition(bond.GetBeginAtomIdx())
    t = conformer.GetAtomPosition(bond.GetEndAtomIdx())
    return [h.Distance(t)]



def bond_property_prediction(bond):
    """Property prediction bond feature.

    Features:
        GetBondType(): one-hot embedding for the type of the bond
        
        GetIsConjugated(): whether the bond is considered to be conjugated
        
        IsInRing(): whether the bond is in a ring
    """
    return onehot(bond.GetBondType(), bond_type_vocab) + \
           [int(bond.GetIsConjugated()), bond.IsInRing()]



def bond_pretrain(bond):
    """Bond feature for pretraining.

    Features:
        GetBondType(): one-hot embedding for the type of the bond
        
        GetBondDir(): one-hot embedding for the direction of the bond
    """
    return onehot(bond.GetBondType(), bond_type_vocab) + \
           onehot(bond.GetBondDir(), bond_dir_vocab)



def residue_symbol(residue):
    """Symbol residue feature.

    Features:
        GetResidueName(): one-hot embedding for the residue symbol
    """
    return onehot(residue.GetResidueName(), residue_vocab, allow_unknown=True)



def residue_default(residue):
    """Default residue feature.

    Features:
        GetResidueName(): one-hot embedding for the residue symbol
    """
    return residue_symbol(residue)



def ExtendedConnectivityFingerprint(mol, radius=2, length=1024):
    """Extended Connectivity Fingerprint molecule feature.

    Features:
        GetMorganFingerprintAsBitVect(): a Morgan fingerprint for a molecule as a bit vector
    """
    ecfp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, length)
    return list(ecfp)




def molecule_default(mol):
    """Default molecule feature."""
    return ExtendedConnectivityFingerprint(mol)


ECFP = ExtendedConnectivityFingerprint

def get_full_connected_edge(frag):
    frag = np.asarray(list(frag))
    return torch.from_numpy(np.repeat(frag, len(frag)-1)), \
        torch.from_numpy(np.concatenate([np.delete(frag, i) for i in range(frag.shape[0])], axis=0))

def remove_repeat_edges(new_edge_index, refer_edge_index, N_atoms):
    new = to_dense_adj(new_edge_index, max_num_nodes=N_atoms)
    ref = to_dense_adj(refer_edge_index, max_num_nodes=N_atoms)
    delta_ = new - ref
    delta_[delta_<1] = 0
    unique, _ = dense_to_sparse(delta_)
    return unique

def get_loop_feature_v1(mol, nonloop_res, loop_res, use_chirality=True):
    '''
    for bio-molecule
    '''
    xyz = mol.GetConformer().GetPositions()
    # covalent
    N_atoms = mol.GetNumAtoms()
    node_feature = []
    edge_index = []
    edge_feature = []
    not_allowed_node = []
    atom2nonloopres = []
    atom2loopres = []
    nonloop_mask = []
    loop_edge_mask = []
    loop_name=[]
    loopbb_graph_idxs = []
    loop_parent_mask_forward = []
    loop_parent_mask_reverse = []
    mol_idx_2_graph_idx = {}
    loop_idx_2_mol_idx = []
    loop_bb_atom_mask = []
    graph_idx = 0
    pose = mol.GetConformer().GetPositions()
    seq_parent = []
    # atom
    old_res_num = 0
    old_res_atom_symbol = ''
    res_atom_idx = 0
    for idx, atom in enumerate(mol.GetAtoms()):
        monomer_info = atom.GetMonomerInfo()
        atom_symbol_ = atom.GetSymbol()
        node_name = f"{monomer_info.GetChainId().replace(' ', 'SYSTEM')}-{monomer_info.GetResidueNumber()}{monomer_info.GetInsertionCode().strip()}"
        # print(f'{monomer_info.GetChainId()}-{monomer_info.GetResidueNumber()}-{monomer_info.GetResidueName()}')
        if old_res_num != monomer_info.GetResidueNumber():
            res_atom_idx = 0
        else:
            res_atom_idx += 1
        if node_name in nonloop_res:
            atom2nonloopres.append(nonloop_res.index(node_name))
            nonloop_mask.append(False)
            loop_bb_atom_mask.append(False)
            if res_atom_idx == 0:
                loop_parent_mask_forward.append(False)     
                if len(loopbb_graph_idxs) !=0 and sum(loop_parent_mask_reverse) == 0:
                    loop_parent_mask_reverse.append(True)
                else:
                    loop_parent_mask_reverse.append(False)
        elif node_name in loop_res:
            atom2loopres.append(loop_res.index(node_name))
            nonloop_mask.append(True)
            loop_idx_2_mol_idx.append(idx)
            if res_atom_idx < 4 and atom_symbol_ in ['C', 'N', 'O']:
                print(f'{atom_symbol_}: {graph_idx}')
                if len(loopbb_graph_idxs) == 0:
                    seq_parent.append(9999)
                    loop_parent_mask_forward[-1] = True
                elif old_res_atom_symbol == 'O':
                    seq_parent.append(loopbb_graph_idxs[-2])
                else:
                    seq_parent.append(loopbb_graph_idxs[-1])
                loop_bb_atom_mask.append(True)
                loopbb_graph_idxs.append(graph_idx)
                old_res_atom_symbol = atom_symbol_
            else:
                loop_bb_atom_mask.append(False)
            # if res_atom_idx == 0: #  and node_name == loop_res[0]:
                
        else:
            not_allowed_node.append(idx)
            continue
        atom_name = f"{monomer_info.GetResidueName()}{monomer_info.GetResidueNumber()}{monomer_info.GetInsertionCode().strip()}-{atom.GetIdx()}{atom_symbol_}"
        loop_name.append(atom_name)
        old_res_num = monomer_info.GetResidueNumber()
        # node
        node_feature.append(atom_default(atom))
        mol_idx_2_graph_idx[idx] = graph_idx
        # update graph idx
        graph_idx += 1
    assert sum(loop_parent_mask_reverse) == 1, f'loop residues are not consecutive'
    assert sum(loop_parent_mask_forward) == 1, f'loop residues are not consecutive'
    # bond
    for bond in mol.GetBonds():
        src_idx, dst_idx = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        try:
            src_idx = mol_idx_2_graph_idx[src_idx]
            dst_idx = mol_idx_2_graph_idx[dst_idx]
        except:
            continue
        if src_idx in loopbb_graph_idxs and dst_idx in loopbb_graph_idxs:
            loop_edge_mask.append(True)
            loop_edge_mask.append(True)
        else:
            loop_edge_mask.append(False)
            loop_edge_mask.append(False)
        # edge
        edge_index.append([src_idx, dst_idx])
        edge_index.append([dst_idx, src_idx])
        edge_feats = bond_default(bond)
        edge_feature.append(edge_feats)
        edge_feature.append(edge_feats)
    # to tensor
    node_feature = torch.from_numpy(np.asanyarray(node_feature)).float()  # nodes_chemical_features
    xyz = pose[list(mol_idx_2_graph_idx.keys()), :]
    xyz = torch.from_numpy(xyz)
    N_atoms = len(mol_idx_2_graph_idx)
    N_loop_atoms = len(loopbb_graph_idxs)
    if use_chirality:
        try:
            chiralcenters = Chem.FindMolChiralCenters(mol,force=True,includeUnassigned=True, useLegacyImplementation=False)
        except:
            chiralcenters = []
        chiral_arr = torch.zeros([N_atoms,3]) 
        for (i, rs) in chiralcenters:
            try:
                i = mol_idx_2_graph_idx[i]
            except:
                continue
            if rs == 'R':
                chiral_arr[i, 0] =1 
            elif rs == 'S':
                chiral_arr[i, 1] =1 
            else:
                chiral_arr[i, 2] =1 
        node_feature = torch.cat([node_feature, chiral_arr.float()],dim=1)
    edge_index = torch.from_numpy(np.asanyarray(edge_index).T).long()
    edge_feature = torch.from_numpy(np.asanyarray(edge_feature)).float()
    # loop
    loopidx2zeroidx = {i:idx for idx, i in enumerate(loopbb_graph_idxs)}
    loop_edge_index = edge_index[:, loop_edge_mask]
    loop_edge_index = loop_edge_index.apply_(lambda x: loopidx2zeroidx.get(x, x))
    seq_parent = torch.from_numpy(np.asanyarray(seq_parent)).long().apply_(lambda x: loopidx2zeroidx.get(x, x))
    loop_xyz = xyz[loop_bb_atom_mask, :]
    # 0:4 bond type 5:11 bond direction 12:18 bond stero 19 bond conjunction
    l_full_edge_s = edge_feature[loop_edge_mask, :5]
    l_full_edge_s = torch.cat([l_full_edge_s, torch.pairwise_distance(loop_xyz[loop_edge_index[0]], loop_xyz[loop_edge_index[1]]).unsqueeze(dim=-1)], dim=-1)
    cov_edge_num = l_full_edge_s.size(0)
    # get fragments based on rotation bonds
    G = nx.Graph()
    [G.add_edge(*edge_idx) for edge_idx in loop_edge_index.numpy().T.tolist()]
    frags = []
    for e in G.edges():
        G2 = copy.deepcopy(G)
        G2.remove_edge(*e)
        if nx.is_connected(G2): continue
        # print(f'{sorted(nx.connected_components(G2), key=len)[0]}|{sorted(nx.connected_components(G2), key=len)[1]}')
        l = (sorted(nx.connected_components(G2), key=len)[0])
        if len(l) < 2: continue
        else:
            frags.append(l)
    if len(frags) != 0:
        frags = sorted(frags, key=len)
        for i in range(len(frags)):
            for j in range(i+1, len(frags)):
                frags[j] -= frags[i]
        frags = [i for i in frags if len(i) > 1]
        frag_edge_index = torch.cat([torch.stack(get_full_connected_edge(i), dim=0) for i in frags], dim=1).long()
        loop_edge_index_new = remove_repeat_edges(new_edge_index=frag_edge_index, refer_edge_index=loop_edge_index, N_atoms=N_loop_atoms)
        loop_edge_index = torch.cat([loop_edge_index, loop_edge_index_new], dim=1)
        l_full_edge_s_new = torch.cat([torch.zeros((loop_edge_index_new.size(1), 5)), torch.pairwise_distance(loop_xyz[loop_edge_index_new[0]], loop_xyz[loop_edge_index_new[1]]).unsqueeze(dim=-1)], dim=-1)
        l_full_edge_s = torch.cat([l_full_edge_s, l_full_edge_s_new], dim=0)
    # interaction
    adj_interaction = torch.ones((N_loop_atoms, N_loop_atoms)) - torch.eye(N_loop_atoms, N_loop_atoms)
    loop_interaction_edge_index, _ = dense_to_sparse(adj_interaction)
    loop_edge_index_new = remove_repeat_edges(new_edge_index=loop_interaction_edge_index, refer_edge_index=loop_edge_index, N_atoms=N_loop_atoms)
    loop_edge_index = torch.cat([loop_edge_index, loop_edge_index_new], dim=1)
    # scale the distance
    l_full_edge_s[:, -1] *= 0.1
    l_full_edge_s_new = torch.cat([torch.zeros((loop_edge_index_new.size(1), 5)), -torch.ones(loop_edge_index_new.size(1), 1)], dim=-1)
    l_full_edge_s = torch.cat([l_full_edge_s, l_full_edge_s_new], dim=0)
    interaction_edge_num = l_full_edge_s_new.size(0)
    loop_cov_edge_mask = torch.zeros(len(l_full_edge_s))
    loop_cov_edge_mask[:cov_edge_num] = 1
    loop_frag_edge_mask = torch.ones(len(l_full_edge_s))
    loop_frag_edge_mask[-interaction_edge_num:] = 0
    assert node_feature.size(0) > 10, 'pocket rdkit mol fail'
    x = (loop_name,loop_xyz, node_feature, edge_index, edge_feature, atom2nonloopres, nonloop_mask, loop_edge_mask, loop_edge_index, l_full_edge_s, loop_cov_edge_mask.bool(), loop_frag_edge_mask.bool(), loop_idx_2_mol_idx, loop_bb_atom_mask, loop_parent_mask_forward, loop_parent_mask_reverse, seq_parent) 
    return x 

def get_loop_feature_strict(mol, nonloop_res, loop_res, use_chirality=True):
    '''
    for bio-molecule
    '''
    # covalent
    N_atoms = mol.GetNumAtoms()
    pose = mol.GetConformer().GetPositions()
    node_feature = []
    edge_index = []
    edge_feature = []
    atom2nonloopres = []
    nonloop_mask = []
    loop_edge_mask = []
    loopbb_graph_idxs = []
    loop_parent_mask_forward = []
    loop_parent_mask_reverse = []
    mol_idx_2_graph_idx = {}
    loop_idx_2_mol_idx = []
    loop_bb_atom_mask = []
    graph_idx = 0
    loop_res_bb_atom_idx_lis = []
    tmp_loop_res_bb_atom_idx_dic = OrderedDict()
    # atom
    old_res_name = ''
    record_C = True
    record_N = True
    for idx, atom in enumerate(mol.GetAtoms()):
        monomer_info = atom.GetMonomerInfo()
        node_name = monomer_info.GetName().strip()
        res_name = f"{monomer_info.GetChainId().replace(' ', 'SYSTEM')}-{monomer_info.GetResidueNumber()}{monomer_info.GetInsertionCode().strip()}"
        # print(f'{monomer_info.GetChainId()}-{monomer_info.GetResidueNumber()}-{monomer_info.GetResidueName()}')
        if old_res_name != res_name:
            res_atom_idx = 0
            if len(tmp_loop_res_bb_atom_idx_dic) != 0:
                loop_res_bb_atom_idx_lis.append(tmp_loop_res_bb_atom_idx_dic)
                tmp_loop_res_bb_atom_idx_dic = OrderedDict()
        else:
            res_atom_idx += 1
        # res
        if res_name in nonloop_res:
            if res_atom_idx == 0:
                loop_parent_mask_forward.append(False)     
                loop_parent_mask_reverse.append(False)
                if old_res_name in loop_res:
                    loop_parent_mask_reverse[-1] = True     
            atom2nonloopres.append(nonloop_res.index(res_name))
            nonloop_mask.append(False)
            loop_bb_atom_mask.append(False)
            if node_name == 'N' and record_N:
                non_loop_N_asparent = atom
                if old_res_name in loop_res:
                    record_N = False                   
            if node_name == 'C' and record_C:
                non_loop_C_asparent = atom
        elif res_name in loop_res:
            # print(f'{node_name}: {graph_idx}')
            if old_res_name in nonloop_res:
                loop_parent_mask_forward[-1] = True
            record_C = False
            nonloop_mask.append(True)
            if node_name in ['N', 'CA', 'C', 'O']:
                loop_idx_2_mol_idx.append(idx)
                tmp_loop_res_bb_atom_idx_dic[node_name] = (graph_idx, atom)
                loop_bb_atom_mask.append(True)
                loopbb_graph_idxs.append(graph_idx)
            else:
                loop_bb_atom_mask.append(False)
        else:
            continue
        # atom_name = f"{monomer_info.GetResidueName()}{monomer_info.GetResidueNumber()}{monomer_info.GetInsertionCode().strip()}-{atom.GetIdx()}{atom_symbol_}"
        # loop_name.append(atom_name)
        old_res_name = res_name
        # node
        node_feature.append(atom_default(atom))
        mol_idx_2_graph_idx[idx] = graph_idx
        # update graph idx
        graph_idx += 1
        # last atom
        if idx == N_atoms-1:
            if len(tmp_loop_res_bb_atom_idx_dic) != 0:
                loop_res_bb_atom_idx_lis.append(tmp_loop_res_bb_atom_idx_dic)
    # calc seq_parent
    reverse_ok = False
    forward_ok = False
    # check if forward is ok
    N_atom = loop_res_bb_atom_idx_lis[0]['N'][1]
    if mol.GetBondBetweenAtoms(non_loop_C_asparent.GetIdx(), N_atom.GetIdx()):
        forward_ok = True
        assert sum(loop_parent_mask_forward) == 1, f'loop residues are not consecutive'
    # check if reverse is ok
    C_atom = loop_res_bb_atom_idx_lis[-1]['C'][1]
    if record_N == False:
        if mol.GetBondBetweenAtoms(non_loop_N_asparent.GetIdx(),C_atom.GetIdx()):
            reverse_ok = True
            assert sum(loop_parent_mask_reverse) == 1, f'loop residues are not consecutive'
    assert forward_ok or reverse_ok, f'both forward and reverse are not ok'
    # forward
    seq_order_forward = []
    seq_parent_forward = []
    if forward_ok:
        for idx, tmp_dic in enumerate(loop_res_bb_atom_idx_lis):
            for order_ele, parent_ele in bb_order_2_parent.items():
                current_atom_gidx, _ = tmp_dic[order_ele]
                seq_order_forward.append(current_atom_gidx)
                if order_ele == 'N':
                    if idx == 0:
                        seq_parent_forward.append(mol_idx_2_graph_idx[non_loop_C_asparent.GetIdx()])                    
                    else:
                        parent_atom_gidx, _ = loop_res_bb_atom_idx_lis[idx-1][parent_ele]
                        seq_parent_forward.append(parent_atom_gidx)
                else:
                    parent_atom_gidx, _ = tmp_dic[parent_ele]
                    seq_parent_forward.append(parent_atom_gidx)
    # reverse 
    seq_order_reverse = []
    seq_parent_reverse = []
    if reverse_ok:
        for idx, tmp_dic in enumerate(reversed(loop_res_bb_atom_idx_lis)):
            for order_ele, parent_ele in bb_order_2_parent_reverse.items():
                current_atom_gidx, _ = tmp_dic[order_ele]
                seq_order_reverse.append(current_atom_gidx)
                if order_ele == 'C':
                    if idx == 0:
                        seq_parent_reverse.append(mol_idx_2_graph_idx[non_loop_N_asparent.GetIdx()])                     
                    else:
                        parent_atom_gidx, _ = loop_res_bb_atom_idx_lis[-idx][parent_ele]
                        seq_parent_reverse.append(parent_atom_gidx)
                else:
                    parent_atom_gidx, _ = tmp_dic[parent_ele]
                    seq_parent_reverse.append(parent_atom_gidx)
    # to tensor
    xyz = pose[list(mol_idx_2_graph_idx.keys()), :]
    xyz = torch.from_numpy(xyz)
    node_feature = torch.from_numpy(np.asanyarray(node_feature)).float()  # nodes_chemical_features
    N_atoms = len(mol_idx_2_graph_idx)
    N_loop_atoms = len(loopbb_graph_idxs)
    # check the distance between seq_parent and seq_order
    if forward_ok:
        dist = torch.pairwise_distance(xyz[torch.from_numpy(np.asarray(seq_parent_forward)).long()], xyz[torch.from_numpy(np.asarray(seq_order_forward)).long()])
        assert (dist < 2).all(), f'loop atoms are not consecutive (bond length > 2)'
    if reverse_ok:
        dist = torch.pairwise_distance(xyz[torch.from_numpy(np.asarray(seq_parent_reverse)).long()], xyz[torch.from_numpy(np.asarray(seq_order_reverse)).long()])
        assert (dist < 2).all(), f'loop atoms are not consecutive (bond length > 2)'
    # bond
    for bond in mol.GetBonds():
        src_idx, dst_idx = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        try:
            src_idx = mol_idx_2_graph_idx[src_idx]
            dst_idx = mol_idx_2_graph_idx[dst_idx]
        except:
            continue
        if src_idx in loopbb_graph_idxs and dst_idx in loopbb_graph_idxs:
            loop_edge_mask.append(True)
            loop_edge_mask.append(True)
        else:
            loop_edge_mask.append(False)
            loop_edge_mask.append(False)
        # edge
        edge_index.append([src_idx, dst_idx])
        edge_index.append([dst_idx, src_idx])
        edge_feats = bond_default(bond)
        edge_feature.append(edge_feats)
        edge_feature.append(edge_feats)
    # to tensor
    if use_chirality:
        try:
            chiralcenters = Chem.FindMolChiralCenters(mol,force=True,includeUnassigned=True, useLegacyImplementation=False)
        except:
            chiralcenters = []
        chiral_arr = torch.zeros([N_atoms,3]) 
        for (i, rs) in chiralcenters:
            try:
                i = mol_idx_2_graph_idx[i]
            except:
                continue
            if rs == 'R':
                chiral_arr[i, 0] =1 
            elif rs == 'S':
                chiral_arr[i, 1] =1 
            else:
                chiral_arr[i, 2] =1 
        node_feature = torch.cat([node_feature, chiral_arr.float()],dim=1)
    edge_index = torch.from_numpy(np.asanyarray(edge_index).T).long()
    edge_feature = torch.from_numpy(np.asanyarray(edge_feature)).float()
    # loop
    loopidx2zeroidx = {i:idx for idx, i in enumerate(loopbb_graph_idxs)}
    loop_edge_index = edge_index[:, loop_edge_mask]
    loop_edge_index = loop_edge_index.apply_(lambda x: loopidx2zeroidx.get(x, x))
    seq_parent_forward = torch.from_numpy(np.asanyarray(seq_parent_forward)).long().apply_(lambda x: loopidx2zeroidx.get(x, x))
    seq_order_forward = torch.from_numpy(np.asanyarray(seq_order_forward)).long().apply_(lambda x: loopidx2zeroidx.get(x, x))
    seq_parent_reverse = torch.from_numpy(np.asanyarray(seq_parent_reverse)).long().apply_(lambda x: loopidx2zeroidx.get(x, x))
    seq_order_reverse = torch.from_numpy(np.asanyarray(seq_order_reverse)).long().apply_(lambda x: loopidx2zeroidx.get(x, x))
    loop_xyz = xyz[loop_bb_atom_mask, :]
    # 0:4 bond type 5:11 bond direction 12:18 bond stero 19 bond conjunction
    l_full_edge_s = edge_feature[loop_edge_mask, :5]
    l_full_edge_s = torch.cat([l_full_edge_s, torch.pairwise_distance(loop_xyz[loop_edge_index[0]], loop_xyz[loop_edge_index[1]]).unsqueeze(dim=-1)], dim=-1)
    cov_edge_num = l_full_edge_s.size(0)
    # get fragments based on rotation bonds
    G = nx.Graph()
    [G.add_edge(*edge_idx) for edge_idx in loop_edge_index.numpy().T.tolist()]
    frags = []
    for e in G.edges():
        G2 = copy.deepcopy(G)
        G2.remove_edge(*e)
        if nx.is_connected(G2): continue
        # print(f'{sorted(nx.connected_components(G2), key=len)[0]}|{sorted(nx.connected_components(G2), key=len)[1]}')
        l = (sorted(nx.connected_components(G2), key=len)[0])
        if len(l) < 2: continue
        else:
            frags.append(l)
    if len(frags) != 0:
        frags = sorted(frags, key=len)
        for i in range(len(frags)):
            for j in range(i+1, len(frags)):
                frags[j] -= frags[i]
        frags = [i for i in frags if len(i) > 1]
        frag_edge_index = torch.cat([torch.stack(get_full_connected_edge(i), dim=0) for i in frags], dim=1).long()
        loop_edge_index_new = remove_repeat_edges(new_edge_index=frag_edge_index, refer_edge_index=loop_edge_index, N_atoms=N_loop_atoms)
        loop_edge_index = torch.cat([loop_edge_index, loop_edge_index_new], dim=1)
        l_full_edge_s_new = torch.cat([torch.zeros((loop_edge_index_new.size(1), 5)), torch.pairwise_distance(loop_xyz[loop_edge_index_new[0]], loop_xyz[loop_edge_index_new[1]]).unsqueeze(dim=-1)], dim=-1)
        l_full_edge_s = torch.cat([l_full_edge_s, l_full_edge_s_new], dim=0)
    # interaction
    adj_interaction = torch.ones((N_loop_atoms, N_loop_atoms)) - torch.eye(N_loop_atoms, N_loop_atoms)
    loop_interaction_edge_index, _ = dense_to_sparse(adj_interaction)
    loop_edge_index_new = remove_repeat_edges(new_edge_index=loop_interaction_edge_index, refer_edge_index=loop_edge_index, N_atoms=N_loop_atoms)
    loop_edge_index = torch.cat([loop_edge_index, loop_edge_index_new], dim=1)
    # scale the distance
    l_full_edge_s[:, -1] *= 0.1
    l_full_edge_s_new = torch.cat([torch.zeros((loop_edge_index_new.size(1), 5)), -torch.ones(loop_edge_index_new.size(1), 1)], dim=-1)
    l_full_edge_s = torch.cat([l_full_edge_s, l_full_edge_s_new], dim=0)
    loop_cov_edge_mask = torch.zeros(len(l_full_edge_s))
    loop_cov_edge_mask[:cov_edge_num] = 1
    assert node_feature.size(0) > 10, 'pocket rdkit mol fail'
    x = (loop_xyz, node_feature, edge_index, edge_feature, atom2nonloopres, nonloop_mask, loop_edge_index, l_full_edge_s, loop_cov_edge_mask.bool(), loop_idx_2_mol_idx, loop_bb_atom_mask, loop_parent_mask_forward, loop_parent_mask_reverse, seq_order_forward, seq_parent_forward, seq_order_reverse, seq_parent_reverse, forward_ok, reverse_ok) 
    return x 

def get_loop_feature_strict_rot(mol, nonloop_res, loop_res, use_chirality=True):
    '''
    for bio-molecule
    '''
    # covalent
    N_atoms = mol.GetNumAtoms()
    pose = mol.GetConformer().GetPositions()
    node_feature = []
    edge_index = []
    edge_feature = []
    atom2nonloopres = []
    nonloop_mask = []
    loop_edge_mask = []
    loopbb_graph_idxs = []
    loop_parent_mask_forward = []
    loop_parent_mask_reverse = []
    mol_idx_2_graph_idx = {}
    loop_idx_2_mol_idx = []
    loop_bb_atom_mask = []
    graph_idx = 0
    loop_res_bb_atom_idx_lis = []
    tmp_loop_res_bb_atom_idx_dic = OrderedDict()
    # atom
    old_res_name = ''
    record_C = True
    record_N = True
    for idx, atom in enumerate(mol.GetAtoms()):
        monomer_info = atom.GetMonomerInfo()
        node_name = monomer_info.GetName().strip()
        res_name = f"{monomer_info.GetChainId().replace(' ', 'SYSTEM')}-{monomer_info.GetResidueNumber()}{monomer_info.GetInsertionCode().strip()}"
        # print(f'{monomer_info.GetChainId()}-{monomer_info.GetResidueNumber()}-{monomer_info.GetResidueName()}')
        if old_res_name != res_name:
            res_atom_idx = 0
            if len(tmp_loop_res_bb_atom_idx_dic) != 0:
                loop_res_bb_atom_idx_lis.append(tmp_loop_res_bb_atom_idx_dic)
                tmp_loop_res_bb_atom_idx_dic = OrderedDict()
        else:
            res_atom_idx += 1
        # res
        if res_name in nonloop_res:
            if res_atom_idx == 0:
                loop_parent_mask_forward.append(False)     
                loop_parent_mask_reverse.append(False)
                if old_res_name in loop_res:
                    loop_parent_mask_reverse[-1] = True     
            atom2nonloopres.append(nonloop_res.index(res_name))
            nonloop_mask.append(False)
            loop_bb_atom_mask.append(False)
            if node_name == 'N' and record_N:
                non_loop_N_asparent = atom
                if old_res_name in loop_res:
                    record_N = False                   
            if node_name == 'C' and record_C:
                non_loop_C_asparent = atom
        elif res_name in loop_res:
            # print(f'{node_name}: {graph_idx}')
            if old_res_name in nonloop_res:
                loop_parent_mask_forward[-1] = True
            record_C = False
            nonloop_mask.append(True)
            if node_name in ['N', 'CA', 'C', 'O']:
                loop_idx_2_mol_idx.append(idx)
                tmp_loop_res_bb_atom_idx_dic[node_name] = (graph_idx, atom)
                loop_bb_atom_mask.append(True)
                loopbb_graph_idxs.append(graph_idx)
            else:
                loop_bb_atom_mask.append(False)
        else:
            continue
        # atom_name = f"{monomer_info.GetResidueName()}{monomer_info.GetResidueNumber()}{monomer_info.GetInsertionCode().strip()}-{atom.GetIdx()}{atom_symbol_}"
        # loop_name.append(atom_name)
        old_res_name = res_name
        # node
        node_feature.append(atom_default(atom))
        mol_idx_2_graph_idx[idx] = graph_idx
        # update graph idx
        graph_idx += 1
        # last atom
        if idx == N_atoms-1:
            if len(tmp_loop_res_bb_atom_idx_dic) != 0:
                loop_res_bb_atom_idx_lis.append(tmp_loop_res_bb_atom_idx_dic)
    # calc seq_parent
    reverse_ok = False
    forward_ok = False
    # check if forward is ok
    N_atom = loop_res_bb_atom_idx_lis[0]['N'][1]
    if mol.GetBondBetweenAtoms(non_loop_C_asparent.GetIdx(), N_atom.GetIdx()):
        forward_ok = True
        assert sum(loop_parent_mask_forward) == 1, f'loop residues are not consecutive'
    # check if reverse is ok
    C_atom = loop_res_bb_atom_idx_lis[-1]['C'][1]
    if record_N == False:
        if mol.GetBondBetweenAtoms(non_loop_N_asparent.GetIdx(),C_atom.GetIdx()):
            reverse_ok = True
            assert sum(loop_parent_mask_reverse) == 1, f'loop residues are not consecutive'
    assert forward_ok or reverse_ok, f'both forward and reverse are not ok'
    # forward
    seq_order_forward = []
    seq_parent_forward = []
    if forward_ok:
        for idx, tmp_dic in enumerate(loop_res_bb_atom_idx_lis):
            for order_ele, parent_ele_lis in bb_order_2_3parent.items():
                current_atom_gidx, _ = tmp_dic[order_ele]
                seq_order_forward.append(current_atom_gidx)
                if order_ele == 'N':
                    if idx == 0:
                        seq_parent_forward.append([mol_idx_2_graph_idx[non_loop_C_asparent.GetIdx()], 9999, 9999])                    
                    else:
                        tmp_lis = []
                        last_dic = loop_res_bb_atom_idx_lis[idx-1]
                        for parent_ele in parent_ele_lis:
                            tmp_lis.append(last_dic.get(parent_ele, [parent_ele])[0])
                        seq_parent_forward.append(tmp_lis)
                elif order_ele == 'CA':
                    last_dic = loop_res_bb_atom_idx_lis[idx-1]
                    cur_dic = loop_res_bb_atom_idx_lis[idx]
                    tmp_lis = [cur_dic.get(parent_ele_lis[0])[0], last_dic.get(parent_ele_lis[1])[0], last_dic.get(parent_ele_lis[-1], [parent_ele_lis[-1]])[0]]
                    seq_parent_forward.append(tmp_lis)
                else:
                    tmp_lis = []
                    cur_dic = loop_res_bb_atom_idx_lis[idx]
                    for parent_ele in parent_ele_lis:
                        tmp_lis.append(cur_dic.get(parent_ele, [parent_ele])[0])
                    seq_parent_forward.append(tmp_lis)
    # reverse 
    seq_order_reverse = []
    seq_parent_reverse = []
    if reverse_ok:
        for idx, tmp_dic in enumerate(reversed(loop_res_bb_atom_idx_lis)):
            for order_ele, parent_ele in bb_order_2_parent_reverse.items():
                current_atom_gidx, _ = tmp_dic[order_ele]
                seq_order_reverse.append(current_atom_gidx)
                if order_ele == 'C':
                    if idx == 0:
                        seq_parent_reverse.append(mol_idx_2_graph_idx[non_loop_N_asparent.GetIdx()])                     
                    else:
                        parent_atom_gidx, _ = loop_res_bb_atom_idx_lis[-idx][parent_ele]
                        seq_parent_reverse.append(parent_atom_gidx)
                else:
                    parent_atom_gidx, _ = tmp_dic[parent_ele]
                    seq_parent_reverse.append(parent_atom_gidx)
    # to tensor
    xyz = pose[list(mol_idx_2_graph_idx.keys()), :]
    xyz = torch.from_numpy(xyz)
    node_feature = torch.from_numpy(np.asanyarray(node_feature)).float()  # nodes_chemical_features
    N_atoms = len(mol_idx_2_graph_idx)
    N_loop_atoms = len(loopbb_graph_idxs)
    # check the distance between seq_parent and seq_order
    if forward_ok:
        dist = torch.pairwise_distance(xyz[torch.from_numpy(np.asarray(seq_parent_forward)).long()], xyz[torch.from_numpy(np.asarray(seq_order_forward)).long()])
        assert (dist < 2).all(), f'loop atoms are not consecutive (bond length > 2)'
    if reverse_ok:
        dist = torch.pairwise_distance(xyz[torch.from_numpy(np.asarray(seq_parent_reverse)).long()], xyz[torch.from_numpy(np.asarray(seq_order_reverse)).long()])
        assert (dist < 2).all(), f'loop atoms are not consecutive (bond length > 2)'
    # bond
    for bond in mol.GetBonds():
        src_idx, dst_idx = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        try:
            src_idx = mol_idx_2_graph_idx[src_idx]
            dst_idx = mol_idx_2_graph_idx[dst_idx]
        except:
            continue
        if src_idx in loopbb_graph_idxs and dst_idx in loopbb_graph_idxs:
            loop_edge_mask.append(True)
            loop_edge_mask.append(True)
        else:
            loop_edge_mask.append(False)
            loop_edge_mask.append(False)
        # edge
        edge_index.append([src_idx, dst_idx])
        edge_index.append([dst_idx, src_idx])
        edge_feats = bond_default(bond)
        edge_feature.append(edge_feats)
        edge_feature.append(edge_feats)
    # to tensor
    if use_chirality:
        try:
            chiralcenters = Chem.FindMolChiralCenters(mol,force=True,includeUnassigned=True, useLegacyImplementation=False)
        except:
            chiralcenters = []
        chiral_arr = torch.zeros([N_atoms,3]) 
        for (i, rs) in chiralcenters:
            try:
                i = mol_idx_2_graph_idx[i]
            except:
                continue
            if rs == 'R':
                chiral_arr[i, 0] =1 
            elif rs == 'S':
                chiral_arr[i, 1] =1 
            else:
                chiral_arr[i, 2] =1 
        node_feature = torch.cat([node_feature, chiral_arr.float()],dim=1)
    edge_index = torch.from_numpy(np.asanyarray(edge_index).T).long()
    edge_feature = torch.from_numpy(np.asanyarray(edge_feature)).float()
    # loop
    loopidx2zeroidx = {i:idx for idx, i in enumerate(loopbb_graph_idxs)}
    loop_edge_index = edge_index[:, loop_edge_mask]
    loop_edge_index = loop_edge_index.apply_(lambda x: loopidx2zeroidx.get(x, x))
    seq_parent_forward = torch.from_numpy(np.asanyarray(seq_parent_forward)).long().apply_(lambda x: loopidx2zeroidx.get(x, x))
    seq_order_forward = torch.from_numpy(np.asanyarray(seq_order_forward)).long().apply_(lambda x: loopidx2zeroidx.get(x, x))
    seq_parent_reverse = torch.from_numpy(np.asanyarray(seq_parent_reverse)).long().apply_(lambda x: loopidx2zeroidx.get(x, x))
    seq_order_reverse = torch.from_numpy(np.asanyarray(seq_order_reverse)).long().apply_(lambda x: loopidx2zeroidx.get(x, x))
    loop_xyz = xyz[loop_bb_atom_mask, :]
    # 0:4 bond type 5:11 bond direction 12:18 bond stero 19 bond conjunction
    l_full_edge_s = edge_feature[loop_edge_mask, :5]
    l_full_edge_s = torch.cat([l_full_edge_s, torch.pairwise_distance(loop_xyz[loop_edge_index[0]], loop_xyz[loop_edge_index[1]]).unsqueeze(dim=-1)], dim=-1)
    cov_edge_num = l_full_edge_s.size(0)
    # get fragments based on rotation bonds
    G = nx.Graph()
    [G.add_edge(*edge_idx) for edge_idx in loop_edge_index.numpy().T.tolist()]
    frags = []
    for e in G.edges():
        G2 = copy.deepcopy(G)
        G2.remove_edge(*e)
        if nx.is_connected(G2): continue
        # print(f'{sorted(nx.connected_components(G2), key=len)[0]}|{sorted(nx.connected_components(G2), key=len)[1]}')
        l = (sorted(nx.connected_components(G2), key=len)[0])
        if len(l) < 2: continue
        else:
            frags.append(l)
    if len(frags) != 0:
        frags = sorted(frags, key=len)
        for i in range(len(frags)):
            for j in range(i+1, len(frags)):
                frags[j] -= frags[i]
        frags = [i for i in frags if len(i) > 1]
        frag_edge_index = torch.cat([torch.stack(get_full_connected_edge(i), dim=0) for i in frags], dim=1).long()
        loop_edge_index_new = remove_repeat_edges(new_edge_index=frag_edge_index, refer_edge_index=loop_edge_index, N_atoms=N_loop_atoms)
        loop_edge_index = torch.cat([loop_edge_index, loop_edge_index_new], dim=1)
        l_full_edge_s_new = torch.cat([torch.zeros((loop_edge_index_new.size(1), 5)), torch.pairwise_distance(loop_xyz[loop_edge_index_new[0]], loop_xyz[loop_edge_index_new[1]]).unsqueeze(dim=-1)], dim=-1)
        l_full_edge_s = torch.cat([l_full_edge_s, l_full_edge_s_new], dim=0)
    # interaction
    adj_interaction = torch.ones((N_loop_atoms, N_loop_atoms)) - torch.eye(N_loop_atoms, N_loop_atoms)
    loop_interaction_edge_index, _ = dense_to_sparse(adj_interaction)
    loop_edge_index_new = remove_repeat_edges(new_edge_index=loop_interaction_edge_index, refer_edge_index=loop_edge_index, N_atoms=N_loop_atoms)
    loop_edge_index = torch.cat([loop_edge_index, loop_edge_index_new], dim=1)
    # scale the distance
    l_full_edge_s[:, -1] *= 0.1
    l_full_edge_s_new = torch.cat([torch.zeros((loop_edge_index_new.size(1), 5)), -torch.ones(loop_edge_index_new.size(1), 1)], dim=-1)
    l_full_edge_s = torch.cat([l_full_edge_s, l_full_edge_s_new], dim=0)
    loop_cov_edge_mask = torch.zeros(len(l_full_edge_s))
    loop_cov_edge_mask[:cov_edge_num] = 1
    assert node_feature.size(0) > 10, 'pocket rdkit mol fail'
    x = (loop_xyz, node_feature, edge_index, edge_feature, atom2nonloopres, nonloop_mask, loop_edge_index, l_full_edge_s, loop_cov_edge_mask.bool(), loop_idx_2_mol_idx, loop_bb_atom_mask, loop_parent_mask_forward, loop_parent_mask_reverse, seq_order_forward, seq_parent_forward, seq_order_reverse, seq_parent_reverse, forward_ok, reverse_ok) 
    return x 

def show_mol_idx(mol):
    from rdkit import Chem
    from rdkit.Chem import Draw

    # 为分子的每个原子编号
    for atom in mol.GetAtoms():
        atom.SetProp("atomLabel", str(atom.GetIdx()))

    # 保存图片到文件
    Draw.MolToFile(mol, "/root/molecule.png", size=(3000,3000), bgcolor="white", kekulize=True, wedgeBonds=True)


# N CA C N CA C N CA C 
#      O      O      O
### angle  mean    std
# N-CA-C   111.10  2.55
# CA-C-O   120.04  0.94
# O-C-N    122.78  2.81
# CA-C-N   116.67  2.22
# C-N-CA   121.62  2.61
###       
# forward start from N to C
#  要预测N原子的坐标, get [C, CA, O], dist = C-N, 根据预测的C-N和C-O确定平面，投影C-N到平面上，获得C-N’，限制CA-C-N’的角度为116.67+-2.22, 以C-N’和C-CA的法线为旋转轴，旋转C-CA，旋转角为限制后的CA-C-N’的角度，获得旋转后的C-CA向量，将其归一化，得到vec, N的坐标 = C的坐标 + vec * dist
#  要预测CA原子的坐标, get [N, C], dist = N-CA, 根据预测的N-CA和N-C确定平面，限制C-CA-N的角度为111.10+-2.55, 以N-CA和N-C的法线为旋转轴，旋转N-C，旋转角为限制后的C-CA-N的角度，获得旋转后的N-C向量，将其归一化，得到vec, CA的坐标 = N的坐标 + vec * dist
#  要预测C原子的坐标，get [CA, N], dist = C-CA, 根据预测的CA-C和CA-N确定平面，限制N-CA-C的角度为121.62+-2.61, 以CA-C和CA-N的法线为旋转轴，旋转CA-N，旋转角为限制后的N-CA-C的角度，获得旋转后的CA-N向量，将其归一化，得到vec, C的坐标 = CA的坐标 + vec * dist
#  要预测O原子的坐标，get [C, CA], dist = O-C, 根据预测的C-O和C-N确定平面，限制N-CA-C的角度为122.78+-2.81, 以C-O和C-N的法线为旋转轴，旋转C-N，旋转角为限制后的N-CA-C的角度，获得旋转后的C-N向量，将其归一化，得到vec, O的坐标 = C的坐标 + vec * dist

# reverse start from C to N
#  要预测C原子的坐标, get [N, CA], dist = C-N, 根据预测的C-N和N-CA确定平面，限制CA-N-C的角度为121.62+-2.61, 以C-N和N-CA的法线为旋转轴，旋转N-CA，旋转角为限制后的CA-N-C的角度，获得旋转后的N-CA向量，将其归一化，得到vec, C的坐标 = N的坐标 + vec * dist
#  要预测O原子的坐标，get [C, N], dist = O-C, 根据预测的C-O和C-N确定平面，限制N-C-O的角度为122.78+-2.81, 以C-O和C-N的法线为旋转轴，旋转C-N，旋转角为限制后的N-C-O的角度，获得旋转后的C-N向量，将其归一化，得到vec, O的坐标 = C的坐标 + vec * dist
#  要预测CA原子的坐标, get [C, O, N], dist = CA-C, 根据预测的C-O和C-N确定平面，投影C-CA到平面上，获得C-CA‘，限制N-C-CA的角度为116.67+-2.22, 以C-CA’和C-N的法线为旋转轴，旋转C-N，旋转角为限制后的N-C-CA的角度，获得旋转后的C-N向量，将其归一化，得到vec, CA的坐标 = C的坐标 + vec * dist
#  要预测N原子的坐标, get [CA, C], dist = N-CA, 根据预测的CA-N和CA-C确定平面，限制C-CA-N的角度为111.10+-2.55, 以CA-N和CA-C的法线为旋转轴，旋转CA-C，旋转角为限制后的C-CA-N的角度，获得旋转后的CA-C向量，将其归一化，得到vec, N的坐标 = CA的坐标 + vec * dist
