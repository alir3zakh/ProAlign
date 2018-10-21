"""
this module contains implementations for different alignment algorithms
"""

import numpy as np
import networkx as nx
import scipy.sparse as sparse
import scipy.optimize as optimize
import sklearn.cluster as cluster
import scipy.sparse.linalg as slnlg
import heapq, random, os, pickle, math
from sklearn.preprocessing import normalize
from numba import jit, njit

import utils
import organism
import interface
import string_db
import visualize
import constants as cs


class Alignment():
    """docstring for Alignment"""

    def __init__(self, paired_nodes, paired_edges, method):
        # store filepaths if needed later
        self.method = method
        self.paired_nodes = paired_nodes
        self.paired_edges = paired_edges
        self.pairs, self.pair_edges = self.load_alignment()

    def load_alignment(self, file_path=cs.JSON_PATH):
        # load nodes
        file_path = os.path.join(file_path, self.method)
        pairs = utils.load_json(
            utils.join_path(file_path, self.paired_nodes))

        # load edges
        pair_edges = utils.load_json(
            utils.join_path(file_path, self.paired_edges))

        return (pairs, pair_edges)


class Aligner():
    """docstring for Aligner"""

    def __init__(self, method):
        self.method = method
        self.alignment = None

    def calculate_measures(self, pairs, pair_edges, bio_net):
        self.measures = {}
        self.ce = len(pair_edges)
        self.measures['CE'] = self.ce

        self.nbs = sum([bio_net.blast_sim_n_rel[
            bio_net.v_ind(x[0], x[1])] for x in pairs])

        self.measures['NBS'] = self.nbs

        p = [(x[0], x[1]) for x in pairs]

        dimAdj = (len(pairs), len(pairs))
        adj = sparse.dok_matrix(dimAdj)
        for e in pair_edges:
            s = p.index(tuple(e[0]))
            t = p.index(tuple(e[1]))
            adj[s, t] = 1
            # adj[t, s] = 1
        adj = adj.tocsr()

        # find connected components
        cc = sparse.csgraph.connected_components(adj)

        # find component size for all labels
        sizes = {}
        for label in cc[1]:
            sizes[label] = sizes.get(label, 0) + 1

        # giant component size
        gcc = max(sizes.values())

        self.lccs = gcc
        self.measures['LCCS'] = self.lccs

        ec1 = sum(bio_net.org1.degree)
        ec2 = sum(bio_net.org2.degree)

        mec = min(ec1, ec2)
        medg = (2 * len(pair_edges))
        EC = medg / mec

        # edge correctnes (EC)
        self.ec = EC
        self.measures['EC'] = self.ec

        if (ec1 < ec2):
            selection = np.array([y for x, y, z in pairs])
            indAdj = bio_net.org2.adjacency[selection[:, None], selection]
        else:
            selection = np.array([x for x, y, z in pairs])
            indAdj = bio_net.org1.adjacency[selection[:, None], selection]

        mind = sum(sum(indAdj))
        ICS = medg / mind
        S3 = medg / (mec + mind - medg)

        # induced conserved structure (ICS)
        self.ics = ICS
        self.measures['ICS'] = self.ics

        # S^3
        self.s3 = S3
        self.measures['S3'] = self.s3

        try:
            # calculation GOC score
            GO_scores = string_db.extract_all_organism_GO('4932+7227+9606.out')
            GOC = 0
            for pair in pairs:
                prot_id1 = bio_net.org1.id_to_node[pair[0]]
                prot_id2 = bio_net.org2.id_to_node[pair[1]]
                terms1 = set(GO_scores.get(prot_id1, []))
                terms2 = set(GO_scores.get(prot_id2, []))
                union_terms = terms1.union(terms2)
                intersection_terms = terms1.intersection(terms2)
                if len(union_terms) > 0:
                    GOC += (len(intersection_terms) / len(union_terms))
            size = min(bio_net.org1.node_count, bio_net.org2.node_count)
            self.GOC = GOC / size
            self.measures['GOC'] = self.GOC
        except Exception as e:
            message = 'skipping GOC calculation, GO info not available!'
            utils.print_log(message)

        try:
            # pathway score calculation:
            PW_score = 0
            PW_count = 0
            matching = {}
            for pair in pairs:
                matching[bio_net.org1.id_to_node[pair[0]]
                         ] = bio_net.org2.id_to_node[pair[1]]

            pw1 = interface.get_pathways(bio_net.org1.org_id)
            pathways1 = set(pw1.keys())
            pw2 = interface.get_pathways(bio_net.org2.org_id)
            pathways2 = set(pw2.keys())

            pathways = list(pathways1.intersection(pathways2))

            for pathway in pathways:
                # print('pathway id: {}'.format(pathway))
                prots1 = set(pw1[pathway])
                # print('prots1: {}'.format(prots1))
                prots2 = set(pw2[pathway])
                # print('prots2: {}'.format(prots2))
                prots1to2 = set([matching[x]
                                 for x in prots1 if x in matching.keys()])
                # print('prots1to2: {}'.format(prots1to2))
                kept = prots1to2.intersection(prots2)
                # PW_score += len(kept) / min(len(prots1), len(prots2))
                PW_score += len(kept)
                # if PW_count>0 and (len(kept) / min(len(prots1), len(prots2)) > 0.7):
                #     print('pathway id: {}, \tscore: {},\tsize: {},{},{}'.format(pathway, len(kept) / min(len(prots1), len(prots2)),len(prots1), len(prots2), len(kept)))
                # PW_count += 1
                PW_count += min(len(prots1), len(prots2))
            self.PWS1 = PW_score / PW_count
            self.measures['PWS1'] = self.PWS1
            # self.PWS = PW_score / len(pathways)

            # pathway score 2 calculation:
            PW_score = 0
            PW_count = 0
            matching = {}
            for pair in pairs:
                matching[bio_net.org1.id_to_node[pair[0]]
                         ] = bio_net.org2.id_to_node[pair[1]]

            pw1 = interface.get_pathways(bio_net.org1.org_id)
            pathways1 = set(pw1.keys())
            pw2 = interface.get_pathways(bio_net.org2.org_id)
            pathways2 = set(pw2.keys())

            pathways = list(pathways1.intersection(pathways2))

            for pathway in pathways:
                prots1 = set(pw1[pathway])
                prots2 = set(pw2[pathway])
                prots1to2 = set([matching[x]
                                 for x in prots1 if x in matching.keys()])
                kept = prots1to2.intersection(prots2)
                if min(len(prots1), len(prots2)) >= cs.PWS2_LIMIT:
                    PW_score += len(kept)
                    PW_count += min(len(prots1), len(prots2))
            self.PWS2 = PW_score / PW_count
            self.measures['PWS2'] = self.PWS2
            # self.PWS = PW_score / len(pathways)
        except Exception as e:
            message = 'skipping PWS calculation, PW info not available!'
            utils.print_log(message)

        return self.measures

    @staticmethod
    def find_paired_edges(pairs, bio_net, file_path=cs.JSON_PATH):
        # find remained edges in the alignment
        pair_edges = []
        to_sec = {}

        for pair in pairs:
            n1, n2, s = pair
            to_sec[n1] = n2

        for e1 in bio_net.org1.edges:
            try:
                x1, y1 = e1
                x2 = to_sec[x1]
                y2 = to_sec[y1]
                e2 = (min(x2, y2), max(x2, y2))
                if e2 in bio_net.org2.edges:
                    pair_edges.append(((x1, x2), (y1, y2)))
            # in case the node was not aligned
            except KeyError:
                continue

        return pair_edges

    def cluster_networks(self, bio_net):
        labels1, label_cnt1 = bio_net.org1.cluster_network(self.method)
        labels2, label_cnt2 = bio_net.org2.cluster_network(self.method)
        return (labels1, label_cnt1, labels2, label_cnt2)

    def cluster_similarity(self, labels1, label_cnt1, labels2, label_cnt2,
                           bio_net):
        if self.method in ['clstr', 'rclst', 'cclst', 'l2clstr',
                           'l2extend', 'l2selextend', 'l2mincpl',
                           'l2mincplextend', 'l2mincplselextend',
                           'l2maxcut', 'l2maxcutextend', 'l2maxcutselextend',
                           'l2brutecut', 'l2brutecutextend',
                           'l2brutecutselextend']:
            # basic overal sum similarity
            dimLbl = (label_cnt1, label_cnt2)
            Lbl = np.zeros(dimLbl)
            for n1 in range(len(labels1)):
                for n2 in range(len(labels2)):
                    Lbl[labels1[n1]
                        ][labels2[n2]
                          ] += bio_net.similarity[bio_net.v_ind(n1, n2)]
            return Lbl

    @utils.time_it
    def greedy_align(self, bio_net, file_path=cs.JSON_PATH):
        # greedy algorithm
        scores = []
        for n1 in range(bio_net.org1.node_count):
            for n2 in range(bio_net.org2.node_count):
                scores.append(
                    (n1, n2, bio_net.similarity[bio_net.v_ind(n1, n2)]))
        scores.sort(key=lambda x: x[2])
        pairs = []
        nodes1 = set()
        nodes2 = set()
        for score in scores:
            n1, n2, s = score
            if (n1 not in nodes1) and (n2 not in nodes2):
                nodes1.add(n1)
                nodes2.add(n2)
                pairs.append(score)
            if ((len(nodes1) == bio_net.org1.node_count) or
                    (len(nodes2) == bio_net.org2.node_count)):
                break

        return pairs

    @utils.time_it
    def semi_greedy_align(self):
        # semi-greedy algorithm
        scores = []
        for n1 in range(bio_net.org1.node_count):
            for n2 in range(bio_net.org2.node_count):
                scores.append(
                    (n1, n2, bio_net.similarity[bio_net.v_ind(n1, n2)]))
        scores.sort(key=lambda x: x[2])

        pairs = []
        pair_dic1 = {}
        pair_dic2 = {}
        remains = []
        remain_dic = {}
        nodes1 = []
        nodes2 = []
        for score in scores:
            n1, n2, s = score
            if (n1 not in nodes1) and (n2 not in nodes2):
                nodes1.append(n1)
                nodes2.append(n2)
                pairs.append(score)
                pair_dic1[n1] = (n2, s)
                pair_dic2[n2] = (n1, s)
            else:
                remains.append(score)
                remain_dic[(n1, n2)] = s
            if ((len(nodes1) == bio_net.org1.node_count) or
                    (len(nodes2) == bio_net.org2.node_count)):
                break

        # round1 finished

        for point in remains:
            n11, n22, s_r1 = point
            n21, s_t1 = pair_dic1.get(n11, (-1, 0))
            n12, s_t2 = pair_dic2.get(n22, (-1, 0))
            if (n12 == -1) or (n21 == -1):
                continue
            s_r2 = remain_dic.get((n12, n21), 0)
            if (s_r1 + s_r2) > (s_t1 + s_t2):
                pairs.remove((n11, n21, s_t1))
                del pair_dic1[n11]
                del pair_dic2[n21]
                pairs.remove((n12, n22, s_t2))
                del pair_dic1[n12]
                del pair_dic2[n22]
                pairs.append((n11, n22, s_r1))
                pair_dic1[n11] = (n22, s_r1)
                pair_dic2[n22] = (n11, s_r1)
                pairs.append((n12, n21, s_r2))
                pair_dic1[n12] = (n21, s_r2)
                pair_dic2[n21] = (n12, s_r2)

        return pairs

    def select_pairs(self, bio_net, pairs):
        if self.method not in ['l2selextend', 'l2mincplselextend',
                               'l2maxcutselextend', 'l2brutecutselextend']:
            return pairs

        new_pairs = [x for x in pairs if x[2] > 0]
        return new_pairs

    def extend_pairs(self, bio_net, pairs):
        if self.method not in ['l2extend', 'l2selextend',
                               'l2mincplextend', 'l2mincplselextend',
                               'l2maxcutextend', 'l2maxcutselextend',
                               'l2brutecutextend', 'l2brutecutselextend']:
            return pairs

        paired1 = set()
        paired2 = set()
        for pair in pairs:
            paired1.add(pair[0])
            paired2.add(pair[1])
        remains1 = [x for x in range(
            bio_net.org1.node_count) if x not in paired1]
        reverse1 = {x: i for i, x in enumerate(remains1)}
        remainset1 = set(remains1)
        remains2 = [x for x in range(
            bio_net.org2.node_count) if x not in paired2]
        reverse2 = {x: i for i, x in enumerate(remains2)}
        remainset2 = set(remains2)

        remain_sim = np.zeros((len(remains1), len(remains2)))

        for pair in pairs:
            nodes1 = [x for x in bio_net.org1.neighbors(
                pair[0]) if x in remainset1]
            nodes2 = [x for x in bio_net.org2.neighbors(
                pair[1]) if x in remainset2]
            for n1 in nodes1:
                for n2 in nodes2:
                    remain_sim[reverse1[n1]][reverse2[n2]] += 1

        # now connect remains
        rp1, rp2 = optimize.linear_sum_assignment(-remain_sim)

        new_pairs = pairs
        for i in range(len(rp1)):
            n1 = remains1[int(rp1[i])]
            n2 = remains2[int(rp2[i])]
            new_pairs.append(
                (n1, n2, bio_net.similarity[bio_net.v_ind(n1, n2)]))

        return new_pairs

    @utils.time_it
    def cluster_align(self, bio_net):
        # maximum weight matching after chosen clustering
        org_cluster = organism.OrgCluster(*self.cluster_networks(bio_net))
        bio_net.org_cluster = org_cluster
        svg_name = '{}-{}-{}_clusters.svg'.format(
            bio_net.org1.org_id, bio_net.org2.org_id, self.method)
        visualize.cluster_vertices(bio_net, file_name=svg_name)

        message = 'clustering networks for "{}" algorithm finihed'.format(
            self.method)
        utils.print_log(message)

        # TODO: save clusters
        # print (org_cluster.label_cnt1, org_cluster.label_cnt2)

        # construct a label simillarity metric for clusters
        cluster_sim = self.cluster_similarity(
            org_cluster.labels1,
            org_cluster.label_cnt1,
            org_cluster.labels2,
            org_cluster.label_cnt2,
            bio_net)

        # now connect clusters
        pl1, pl2 = optimize.linear_sum_assignment(-cluster_sim)

        message = 'clusters aligned for "{}" algorithm'.format(
            self.method)
        utils.print_log(message)

        cl_pairs = set()
        for i in range(len(pl1)):
            n1 = int(pl1[i])
            n2 = int(pl2[i])
            cl_pairs.add((n1, n2))

        # construct empty simillarity matrix for each cluster pair
        cl_sim = {}
        for cl_pair in cl_pairs:
            l1, l2 = cl_pair
            dim_cl = (org_cluster.cl_size1[l1], org_cluster.cl_size2[l2])
            cl_sim[cl_pair] = np.zeros(dim_cl)

        # calculate inner costs in each cluster pair
        for n1 in range(bio_net.org1.node_count):
            for n2 in range(bio_net.org2.node_count):
                l1, lind1 = org_cluster.node_dic1[n1]
                l2, lind2 = org_cluster.node_dic2[n2]
                if (l1, l2) in cl_pairs:
                    cl_sim[(l1, l2)][lind1][lind2] = 1 - \
                        bio_net.similarity[bio_net.v_ind(n1, n2)]

        message = ('starting to pair nodes in each cluster'
                   ' for "{}" algorithm').format(self.method)
        utils.print_log(message)

        # now allign nodes inside alligned clusters
        pairs = []
        for cl_pair in cl_pairs:
            l1, l2 = cl_pair
            p1, p2 = optimize.linear_sum_assignment(cl_sim[cl_pair])
            for i in range(len(p1)):
                n1 = org_cluster.cl_dic1[(l1, int(p1[i]))]
                n2 = org_cluster.cl_dic2[(l2, int(p2[i]))]
                pairs.append(
                    (n1, n2, bio_net.similarity[bio_net.v_ind(n1, n2)]))

        pairs = self.select_pairs(bio_net, pairs)
        pairs = self.extend_pairs(bio_net, pairs)
        return pairs

    @utils.time_it
    def max_weight_align(self, bio_net):
        # maximum weight matching algorithm
        # use the scipy implementation
        p1, p2 = optimize.linear_sum_assignment(-bio_net.similarity)

        pairs = []
        for i in range(len(p1)):
            n1 = int(p1[i])
            n2 = int(p2[i])
            pairs.append((n1, n2, bio_net.similarity[bio_net.v_ind(n1, n2)]))

        return pairs

    @utils.time_it
    def isorankN_align(self, bio_net):
        # use interface for isorankN to align
        pairs = interface.isorankN_align(bio_net)
        return pairs

    @utils.time_it
    def NETAL_align(self, bio_net):
        # use interface for NETAL to align
        pairs = interface.NETAL_align(bio_net)
        return pairs

    @utils.time_it
    def pinalog_align(self, bio_net):
        # use interface for pinalog to align
        pairs = interface.pinalog_align(bio_net)
        return pairs

    @utils.time_it
    def CGRAAL_align(self, bio_net):
        # use interface for CGRAAL to align
        pairs = interface.CGRAAL_align(bio_net)
        return pairs

    @utils.time_it
    def GRAAL_align(self, bio_net):
        # use interface for GRAAL to align
        pairs = interface.GRAAL_align(bio_net)
        return pairs

    @utils.time_it
    def MIGRAAL_align(self, bio_net):
        # use interface for GRAAL to align
        pairs = interface.MIGRAAL_align(bio_net)
        return pairs

    @utils.time_it
    def HubAlign_align(self, bio_net):
        # use interface for GRAAL to align
        pairs = interface.HubAlign_align(bio_net)
        return pairs

    @utils.time_it
    def MAGNA_align(self, bio_net):
        # use interface for GRAAL to align
        pairs = interface.MAGNA_align(bio_net)
        return pairs

    @utils.time_it
    def PROPER_align(self, bio_net):
        # use interface for GRAAL to align
        pairs = interface.PROPER_align(bio_net)
        return pairs

    @utils.time_it
    def SPINAL_align(self, bio_net):
        # use interface for GRAAL to align
        pairs = interface.SPINAL_align(bio_net, self.spinal_alg)
        return pairs

    @utils.time_it
    def seed_extend_SA_align(self, bio_net):
        #initialization
        assignment_file_path = os.path.join(cs.JSON_PATH,
                                            '{}_assignment'.format(self.seed_alg))
        node_idx1 = bio_net.org1.degree.argsort()
        node_idx2 = bio_net.org2.degree.argsort()

        pairs ,algn_infos = [], []
        round_select1, round_select2 = [], []
        algn_info = {}

        # seed similarity
        sim_dim = (bio_net.org1.node_count, bio_net.org2.node_count)
        net_sim = bio_net.similarity.reshape(sim_dim)
        net_pure_sim = bio_net.blast_sim.reshape(sim_dim)

        # organism graphs
        G1 = nx.Graph()
        G1.add_nodes_from(range(bio_net.org1.node_count))
        G1.add_edges_from(bio_net.org1.edges)

        G2 = nx.Graph()
        G2.add_nodes_from(range(bio_net.org2.node_count))
        G2.add_edges_from(bio_net.org2.edges)

        if self.seed_alg == 'blast+cut_coeff+betweenness_centrality':
            bc_file_name1 = ('mss={}-{}.bc'.format(cs.MAX_SEED_SIZE, bio_net.org1.org_id))
            bc_file_name2 = ('mss={}-{}.bc'.format(cs.MAX_SEED_SIZE, bio_net.org2.org_id))

            if not utils.files_exist([bc_file_name1, bc_file_name2], assignment_file_path):
                BC1 = nx.betweenness_centrality(G1)
                BC2 = nx.betweenness_centrality(G2)

                with open(utils.join_path(assignment_file_path, bc_file_name1), 'wb') as fp:
                    pickle.dump(BC1, fp)

                with open(utils.join_path(assignment_file_path, bc_file_name2), 'wb') as fp:
                    pickle.dump(BC2, fp)

            else:
                print('using previous betweenness centrality')
                with open(utils.join_path(assignment_file_path, bc_file_name1), 'rb') as fp:
                    BC1 = pickle.load(fp)

                with open(utils.join_path(assignment_file_path, bc_file_name2), 'rb') as fp:
                    BC2 = pickle.load(fp)

            BC1_list = [(node, value) for node, value in BC1.items()]
            BC2_list = [(node, value) for node, value in BC2.items()]
            for node, value in sorted(BC1_list, key=lambda tup: tup[1], reverse=True):
                round_select1.append(node)
                if len(round_select1) == cs.MAX_SEED_SIZE:
                    break

            for node, value in sorted(BC2_list, key=lambda tup: tup[1], reverse=True):
                round_select2.append(node)
                if len(round_select2) == cs.MAX_SEED_SIZE:
                    break

        algn_info['s1'] = round_select1
        algn_info['s2'] = round_select2

        # pair the bucket
        seed_sim = net_sim[np.array(round_select1)[:, None], np.array(round_select2)]

        # now connect clusters
        assignment_file_name = ('seed-mss={}-hungarian-{}-{}-sim={}.json'.format(
                                    cs.MAX_SEED_SIZE, bio_net.org1.org_id,
                                    bio_net.org2.org_id, bio_net.similarity_mode))

        pl1, pl2 = utils.linear_sum_assignment(-seed_sim, file_name=assignment_file_name,
                                                        path_name=assignment_file_path)

        message = 'seeds aligned for "{}" algorithm'.format(self.method)
        utils.print_log(message)


        # save seed pairs
        seed_pairs = []
        for i in range(len(pl1)):
            n1 = round_select1[int(pl1[i])]
            n2 = round_select2[int(pl2[i])]
            seed_pairs.append((n1, n2, bio_net.similarity[bio_net.v_ind(n1, n2)]))

        # remove some pairs
        sorted_pairs = sorted(seed_pairs, key=lambda x: x[2], reverse=True)
        cut_point = max(int(cs.SEED_KEEP_RATIO * len(seed_pairs)), 0)
        seed_pairs = sorted_pairs[:cut_point]


        # algn_info['pairs'] = seed_pairs
        # algn_infos.append(algn_info)
        # algn_info = {}

        # seed_pairs = [(x, x, bio_net.similarity[bio_net.v_ind(x, x)]) for x in range(200)]

        final_pairs = self.extend(seed_pairs, bio_net)
        current_CE = len(self.find_paired_edges(final_pairs, bio_net))
        utils.print_log('Initial CE: ' + str(current_CE))

        #Anealing fase
        itr = 0
        T = cs.T_HIGH
        while T >= cs.T_LOW:
            itr += 1
            T *= 0.99
            S1new, S2new = self.neighbor(seed_pairs, final_pairs, BC1, BC2, bio_net)
            # print(S1new, S2new)

            seed_sim = net_sim[np.array(S1new)[:, None], np.array(S2new)]
            pl1, pl2 = utils.linear_sum_assignment(-seed_sim, check=False)
            # utils.print_log('iteration: {}, Matching Ended.'.format(itr))

            # save seed pairs
            new_seed_pairs = []
            for i in range(len(pl1)):
                n1 = round_select1[int(pl1[i])]
                n2 = round_select2[int(pl2[i])]
                new_seed_pairs.append((n1, n2, bio_net.similarity[bio_net.v_ind(n1, n2)]))

            new_extended_pairs = self.extend(new_seed_pairs, bio_net)
            new_CE = len(self.find_paired_edges(new_extended_pairs, bio_net))
            msg = 'iter: {}, CE: {}, T: {}'.format(itr, new_CE, T)
            utils.print_log(msg)

            if new_CE >= current_CE:
                current_CE = new_CE
                final_pairs = new_extended_pairs
                seed_pairs = new_seed_pairs

            else:
                prob = math.exp((new_CE - current_CE) / T)
                choice = np.random.choice(['pick', 'ignore'], p=[prob, 1 - prob])

                if(choice == 'pick'):
                    current_CE = new_CE
                    final_pairs = new_extended_pairs
                    seed_pairs = new_seed_pairs

        return(final_pairs)

    def neighbor(self, pairs, extended_pairs, BC1, BC2, bio_net):
        #Swaping nodes in first seed
        S1, S2 = [], []
        for pair in pairs:
            S1.append(pair[0])
            S2.append(pair[1])

        org1_mapping, org2_mapping = {}, {}
        for pair in extended_pairs:
            org1_mapping[pair[0]] = pair[1]
            org2_mapping[pair[1]] = pair[0]

        x_space = [x for x in range(len(BC1)) if x not in S1]
        y_space = S1

        utils.print_log('Seed changes in first orgasim')

        E1, E2 = [], []
        for _ in range(cs.SWAP_SIZE):
            x_space_sum = sum([BC1[x] for x in x_space])
            y_space_sum = sum([BC1[y] for y in y_space])

            x_space_weight, y_space_weight = {}, {}
            for node in x_space:
                x_space_weight[node] = BC1[node] / x_space_sum

            for node in y_space:
                y_space_weight[node] = BC1[node] / y_space_sum

            x = np.random.choice(x_space, p=[x_space_weight[x] for x in x_space])
            y = np.random.choice(y_space, p=[y_space_weight[y] for y in y_space])

            x_data, y_data = [], []
            CE_x, CE_y = 0, 0

            if x in org1_mapping:
                xx = org1_mapping[x]
                for n1 in bio_net.org1.neighbors(x):
                    if n1 in org1_mapping:
                        n2 = org1_mapping[n1]
                        if n2 in bio_net.org2.neighbors(xx):
                            CE_x += 1

            if y in org1_mapping:
                yy = org1_mapping[y]
                for n1 in bio_net.org1.neighbors(y):
                    if n1 in org1_mapping:
                        n2 = org1_mapping[n1]
                        if n2 in bio_net.org2.neighbors(yy):
                            CE_y += 1

            x_data.append(CE_x)
            x_data.append(BC1[x])
            x_data.append(bio_net.org1.degree[x])
            utils.print_log('node to add: {}. info: {}'.format(x, x_data))

            y_data.append(CE_y)
            y_data.append(BC1[y])
            y_data.append(bio_net.org1.degree[y])

            utils.print_log('node to remove: {}. info: {}'.format(y, y_data))
            [x_space.remove(x), y_space.remove(y), E1.append(x)]

        y_space.extend(E1)
        S1new = y_space.copy()


        #Swaping nodes in second seed
        x_space = [x for x in range(len(BC2)) if x not in S2]
        y_space = S2

        utils.print_log('Seed changes in second orgasim')

        for _ in range(cs.SWAP_SIZE):
            x_space_sum = sum([BC2[x] for x in x_space])
            y_space_sum = sum([BC2[y] for y in y_space])

            x_space_weight, y_space_weight = {}, {}
            for node in x_space:
                x_space_weight[node] = BC2[node] / x_space_sum

            for node in y_space:
                y_space_weight[node] = BC2[node] / y_space_sum

            x = np.random.choice(x_space, p=[x_space_weight[x] for x in x_space])
            y = np.random.choice(y_space, p=[y_space_weight[y] for y in y_space])

            x_data, y_data = [], []
            CE_x, CE_y = 0, 0

            if x in org2_mapping:
                xx = org2_mapping[x]
                for n1 in bio_net.org2.neighbors(x):
                    if n1 in org2_mapping:
                        n2 = org2_mapping[n1]
                        if n2 in bio_net.org1.neighbors(xx):
                            CE_x += 1

            if y in org2_mapping:
                yy = org2_mapping[y]
                for n1 in bio_net.org2.neighbors(y):
                    if n1 in org2_mapping:
                        n2 = org2_mapping[n1]
                        if n2 in bio_net.org1.neighbors(yy):
                            CE_y += 1

            x_data.append(CE_x)
            x_data.append(BC1[x])
            x_data.append(bio_net.org1.degree[x])
            utils.print_log('node to add: {}. info: {}'.format(x, x_data))

            y_data.append(CE_y)
            y_data.append(BC1[y])
            y_data.append(bio_net.org1.degree[y])

            utils.print_log('node to remove: {}. info: {}'.format(y, y_data))
            [x_space.remove(x), y_space.remove(y), E2.append(x)]

        y_space.extend(E2)
        S2new = y_space.copy()

        return((S1new, S2new))

    def extend(self, seed_pairs, node_paired1, node_paired2, bio_net):
        node_idx1 = bio_net.org1.degree.argsort()
        node_idx2 = bio_net.org2.degree.argsort()

        pairs = seed_pairs.copy()
        reached_neighs1 = set.union(
            *[set(bio_net.org1.neighbors(x[0])) for x in pairs])
        remains1 = [x for x in node_idx1 if not node_paired1[x]]

        reached_neighs2 = set.union(
            *[set(bio_net.org2.neighbors(x[1])) for x in pairs])
        remains2 = [x for x in node_idx2 if not node_paired2[x]]

        paired1 = set([x[0] for x in pairs])
        paired2 = set([x[1] for x in pairs])

        scores_dict = dict()
        scores_record_dict = dict()
        scores_heap = []
        scores_set = set()

        # generate seed scores
        for seed_pair in pairs:
            for i1 in bio_net.org1.neighbors(seed_pair[0]):
                for i2 in bio_net.org2.neighbors(seed_pair[1]):
                    # if not already aligned
                    if not(node_paired1[i1] or node_paired2[i2]):
                        if self.extend_sim_method == 'common_neighbor':
                            # common neighbor score
                            scores_dict[(i1, i2)] = scores_dict.get((i1, i2), 0) + 1
                        elif self.extend_sim_method == 'jaccard':
                            # Jaccard's coef score
                            d1 = bio_net.org1.degree[i1]
                            d2 = bio_net.org2.degree[i2]
                            scores_dict[(i1, i2)] = scores_dict.get((i1, i2), 0) + (1 / (d1 + d2))
                        elif self.extend_sim_method == 'adamic':
                            # Adamic's coef score
                            ne1 = seed_pair[0]
                            ne2 = seed_pair[1]
                            nd1 = bio_net.org1.degree[ne1]
                            nd2 = bio_net.org2.degree[ne2]
                            scores_dict[(i1, i2)] = scores_dict.get((i1, i2), 0) + (1 / np.log(nd1 + nd2))

        # # build data structures needed
        # for pair in scores_dict:
        #     score = scores_dict[pair]
        #     scores_record_dict[score] = scores_record_dict.get(score, []) + [pair]
        # for score in scores_record_dict:
        #     # max heap
        #     heapq.heappush(scores_heap, -score)
        # heapq.heappush(scores_heap, 0)
        #
        # # sorted blast
        # sim_scores = []
        # for n1 in range(bio_net.org1.node_count):
        #     for n2 in range(bio_net.org2.node_count):
        #         sim_scores.append((n1, n2, bio_net.similarity[bio_net.v_ind(n1, n2)]))
        #
        # sim_scores.sort(key=lambda x: -x[2])
        # sim_pointer = 0
        #
        # # start extend procedure
        # message = 'starting single extend procedure'
        # utils.print_log(message)
        #
        # loops_counter = [0, 0, 0]
        # while not (all([node_paired1[x] for x in node_paired1]) or
        #            all([node_paired2[x] for x in node_paired2])):
        #     loops_counter[0] += 1
        #     message = 'paired {} so far; {} loops ({},{})'.format(
        #         len([x for x in node_paired1 if node_paired1[x]]),
        #         loops_counter[0], loops_counter[1], loops_counter[2],)
        #     utils.print_log(message, mode='progress')
        #
        #     score = abs(heapq.heappop(scores_heap))
        #     if score != 0:
        #         loops_counter[1] += 1
        #         nexts = scores_record_dict[score]
        #
        #         # proper like max func
        #         next_pair = min(nexts, key=lambda x: (
        #             abs(bio_net.org1.degree[x[0]] - bio_net.org2.degree[x[1]]),
        #             bio_net.org1.degree[x[0]] + bio_net.org2.degree[x[1]], random.random()))
        #         # note: add other max functions!
        #
        #         # current_score = scores_dict[next_pair]
        #         del scores_dict[next_pair]
        #         if len(nexts) > 1:
        #             heapq.heappush(scores_heap, -score)
        #             scores_record_dict[score].remove(next_pair)
        #         else:
        #             del scores_record_dict[score]
        #
        #     else:
        #         # restore elemnt in heap
        #         heapq.heappush(scores_heap, 0)
        #
        #         # choose from blast
        #         while (node_paired1[sim_scores[sim_pointer][0]] or
        #                node_paired2[sim_scores[sim_pointer][1]]):
        #             loops_counter[2] += 1
        #             sim_pointer += 1
        #
        #         chosen = sim_scores[sim_pointer]
        #         next_pair = (chosen[0], chosen[1])
        #
        #     if not (node_paired1[next_pair[0]] or node_paired2[next_pair[1]]):
        #         # add new pair
        #         pairs.append((next_pair[0], next_pair[1],
        #             bio_net.similarity[bio_net.v_ind(next_pair[0], next_pair[1])]))
        #         node_paired1[next_pair[0]] = True
        #         node_paired2[next_pair[1]] = True
        #
        #         # update data structures
        #         for i1 in bio_net.org1.neighbors(next_pair[0]):
        #             for i2 in bio_net.org2.neighbors(next_pair[1]):
        #                 old_score = scores_dict.get((i1, i2), 0)
        #                 # update old score
        #                 old_record = scores_record_dict.get(old_score, [])
        #                 if old_score > 0:
        #                     del scores_dict[(i1, i2)]
        #                     if len(old_record) > 1:
        #                         scores_record_dict[old_score].remove(
        #                             (i1, i2))
        #                     elif len(old_record) == 1:
        #                         del scores_record_dict[old_score]
        #                         # delete from heap
        #                         # this location is still O(n)
        #                         heap_index = scores_heap.index(-old_score)
        #                         # O(log(n)) deletion
        #                         # https://stackoverflow.com/questions/10162679/python-delete-element-from-heap
        #                         scores_heap[heap_index] = scores_heap[-1]
        #                         scores_heap.pop()
        #                         if heap_index < len(scores_heap):
        #                             heapq._siftup(scores_heap, heap_index)
        #                             heapq._siftdown(scores_heap, 0,
        #                                             heap_index)
        #
        #                 # if not already aligned
        #                 if not(node_paired1[i1] or node_paired2[i2]):
        #                     if self.extend_sim_method == 'common_neighbor':
        #                         # common neighbor score
        #                         new_score = old_score + 1
        #                     if self.extend_sim_method == 'jaccard':
        #                         # Jaccard's coef score
        #                         d1 = bio_net.org1.degree[i1]
        #                         d2 = bio_net.org2.degree[i2]
        #                         new_score = old_score + (1 / (d1 + d2))
        #                     if self.extend_sim_method == 'adamic':
        #                         # Jaccard's coef score
        #                         ne1 = next_pair[0]
        #                         ne2 = next_pair[1]
        #                         nd1 = bio_net.org1.degree[ne1]
        #                         nd2 = bio_net.org2.degree[ne2]
        #                         new_score = old_score + \
        #                             (1 / np.log(nd1 + nd2))
        #
        #                     scores_dict[(i1, i2)] = new_score
        #
        #                     # update new score
        #                     new_record = scores_record_dict.get(new_score, [])
        #                     if len(new_record) == 0:
        #                         heapq.heappush(scores_heap, -new_score)
        #                         scores_record_dict[new_score] = [(i1, i2)]
        #                     else:
        #                         scores_record_dict[new_score].append((i1, i2))
        #
        # message = 'single extend procedure finished'
        # utils.print_log(message, mode='end_progress')
        #
        # return pairs

    def optnet_align(self, bio_net):
        # use interface for GRAAL to align
        pairs = interface.optnet_align(bio_net)
        return pairs

    @utils.time_it
    def moduleAlign_align(self, bio_net):
        # use interface for GRAAL to align
        pairs = interface.moduleAlign_align(bio_net)
        return pairs

    @utils.time_it
    def seed_extend_align(self, bio_net):
        # initialization
        assignment_file_path = os.path.join(cs.JSON_PATH,
                                            '{}_assignment'.format(self.seed_alg))

        if not os.path.exists(assignment_file_path):
            os.makedirs(assignment_file_path)

        if self.cut_coef == 'degree':
            node_idx1 = bio_net.org1.degree.argsort()
            node_idx2 = bio_net.org2.degree.argsort()
        elif self.cut_coef == 'neighbor-degree':
            node_score1 = [0] * bio_net.org1.node_count
            for i in range(len(node_score1)):
                node_score1[i] = (cs.NEIGHBOR_STRENGTH *
                                  sum([bio_net.org1.degree[x]
                                       for x in bio_net.org1.neighbors(i)]))
            node_score1 = [sum(x)
                           for x in zip(node_score1, bio_net.org1.degree)]
            node_idx1 = np.array(node_score1).argsort()
            node_score2 = [0] * bio_net.org2.node_count
            for i in range(len(node_score2)):
                node_score2[i] = (cs.NEIGHBOR_STRENGTH *
                                  sum([bio_net.org2.degree[x]
                                       for x in bio_net.org2.neighbors(i)]))
            node_score2 = [sum(x)
                           for x in zip(node_score2, bio_net.org2.degree)]
            node_idx2 = np.array(node_score2).argsort()

        node_selected1 = {i: False for i in node_idx1}
        node_paired1 = {i: False for i in node_idx1}
        node_selected2 = {i: False for i in node_idx2}
        node_paired2 = {i: False for i in node_idx2}
        pairs = []
        algn_infos = []
        algn_info = {}
        # main loop

        # make sure base seed has all components
        base_select1 = []
        component_labels1 = bio_net.org1.components()[1]
        comp_in_seed1 = {i: False for i in set(component_labels1)}
        for i in reversed(node_idx1):
            if not comp_in_seed1[component_labels1[i]]:
                base_select1.append(i)
                # node_selected1[i] = True
                comp_in_seed1[component_labels1[i]] = True
            if False not in comp_in_seed1.values():
                break

        base_select2 = []
        component_labels2 = bio_net.org2.components()[1]
        comp_in_seed2 = {i: False for i in set(component_labels2)}
        for i in reversed(node_idx2):
            if not comp_in_seed2[component_labels2[i]]:
                base_select2.append(i)
                # node_selected2[i] = True
                comp_in_seed2[component_labels2[i]] = True
            if False not in comp_in_seed2.values():
                break

        # seed similarity
        sim_dim = (bio_net.org1.node_count, bio_net.org2.node_count)
        net_sim = bio_net.similarity.reshape(sim_dim)
        net_pure_sim = bio_net.blast_sim.reshape(sim_dim)

        G1 = nx.Graph()
        G1.add_nodes_from(range(bio_net.org1.node_count))
        G1.add_edges_from(bio_net.org1.edges)

        G2 = nx.Graph()
        G2.add_nodes_from(range(bio_net.org2.node_count))
        G2.add_edges_from(bio_net.org2.edges)

        if self.seed_alg == 'blast':
            # greedy algorithm
            scores = []
            for n1 in range(bio_net.org1.node_count):
                for n2 in range(bio_net.org2.node_count):
                    scores.append(
                        (n1, n2, bio_net.blast_sim[bio_net.v_ind(n1, n2)]))
            scores.sort(key=lambda x: -x[2])

            round_select1 = []
            round_select2 = []
            new_pairs = []
            for score in scores:
                if score[2] < cs.BLAST_CUT:
                    break
                n1 = score[0]
                n2 = score[1]
                round_select1.append(n1)
                round_select2.append(n2)
                if ((not node_paired1[n1]) and (not node_paired2[n2])):
                    new_pairs.append(
                        (n1, n2, bio_net.similarity[bio_net.v_ind(n1, n2)]))
                    node_paired1[n1] = True
                    node_paired2[n2] = True
            algn_info['s1'] = round_select1
            algn_info['s2'] = round_select2
            algn_info['pairs'] = new_pairs
            algn_infos.append(algn_info)
            algn_info = {}

        # blast with regards to degree
        elif self.seed_alg in ['blast+cut_coeff',
                               'blast+cut_coeff+pagerank_degree',
                               'blast+cut_coeff+pagerank_blast',
                               'blast+cut_coeff+fiedler_vector',
                               'blast+cut_coeff+betweenness_centrality',
                               'blast+cut_coeff+closeness_centrality',
                               'blast+cut_coeff+current_flow_betweenness_centrality',
                               'blast+cut_coeff+current_flow_closeness_centrality']:
            round_select1, round_select2 = [], []

            # first find the base seed
            # round_select1 = base_select1
            # Creating seeds

            if self.seed_alg == 'blast+cut_coeff':
                for i in reversed(node_idx1):
                    if not node_selected1[i]:
                        round_select1.append(i)
                        node_selected1[i] = True
                    if len(round_select1) >= cs.MAX_SEED_SIZE:
                        break
                if len(round_select1) < cs.MAX_SEED_SIZE:
                    for i in base_select1:
                        if not node_selected1[i]:
                            round_select1.append(i)
                            node_selected1[i] = True
                        if len(round_select1) >= cs.MAX_SEED_SIZE:
                            break

                # round_select2 = base_select2
                for i in reversed(node_idx2):
                    if not node_selected2[i]:
                        round_select2.append(i)
                        node_selected2[i] = True
                    if len(round_select2) >= cs.MAX_SEED_SIZE:
                        break
                if len(round_select2) < cs.MAX_SEED_SIZE:
                    for i in base_select2:
                        if not node_selected2[i]:
                            round_select2.append(i)
                            node_selected2[i] = True
                        if len(round_select2) >= cs.MAX_SEED_SIZE:
                            break

            elif self.seed_alg == 'blast+cut_coeff+closeness_centrality':
                cc_file_name1 = ('mss={}-{}.cc'.format(cs.MAX_SEED_SIZE,
                                                       bio_net.org1.org_id))

                cc_file_name2 = ('mss={}-{}.cc'.format(cs.MAX_SEED_SIZE,
                                                       bio_net.org2.org_id))

                if not utils.files_exist([cc_file_name1, cc_file_name2], assignment_file_path):
                    CC1 = nx.closeness_centrality(G1)
                    CC1 = [(node, value) for node, value in CC1.items()]

                    CC2 = nx.closeness_centrality(G2)
                    CC2 = [(node, value) for node, value in CC2.items()]

                    with open(utils.join_path(assignment_file_path, cc_file_name1), 'wb') as fp:
                        pickle.dump(CC1, fp)

                    with open(utils.join_path(assignment_file_path, cc_file_name2), 'wb') as fp:
                        pickle.dump(CC2, fp)

                else:
                    print('using previous closeness centrality')
                    with open(utils.join_path(assignment_file_path, cc_file_name1), 'rb') as fp:
                        CC1 = pickle.load(fp)

                    with open(utils.join_path(assignment_file_path, cc_file_name2), 'rb') as fp:
                        CC2 = pickle.load(fp)

                for node, value in sorted(CC1, key=lambda tup: tup[1], reverse=True):
                    round_select1.append(node)
                    if len(round_select1) == cs.MAX_SEED_SIZE:
                        break

                for node, value in sorted(CC2, key=lambda tup: tup[1], reverse=True):
                    round_select2.append(node)
                    if len(round_select2) == cs.MAX_SEED_SIZE:
                        break

            elif self.seed_alg == 'blast+cut_coeff+current_flow_betweenness_centrality':
                cfbc_file1 = ('mss={}-{}.cfbc'.format(cs.MAX_SEED_SIZE,
                                                       bio_net.org1.org_id))

                cfbc_file2 = ('mss={}-{}.cfbc'.format(cs.MAX_SEED_SIZE,
                                                       bio_net.org2.org_id))

                connected_graphs1 = list(nx.connected_component_subgraphs(G1))
                connected_graphs2 = list(nx.connected_component_subgraphs(G2))

                comp1 = sorted(connected_graphs1, key=len, reverse=True)[0]
                comp2 = sorted(connected_graphs2, key=len, reverse=True)[0]

                if not utils.files_exist([cfbc_file1, cfbc_file2], assignment_file_path):
                    CFBC1 = nx.current_flow_betweenness_centrality(comp1)
                    CFBC1 = [(node, value) for node, value in CFBC1.items()]

                    CFBC2 = nx.current_flow_betweenness_centrality(comp2)
                    CFBC2 = [(node, value) for node, value in CFBC2.items()]

                    with open(utils.join_path(assignment_file_path, cfbc_file1), 'wb') as fp:
                        pickle.dump(CFBC1, fp)

                    with open(utils.join_path(assignment_file_path, cfbc_file2), 'wb') as fp:
                        pickle.dump(CFBC2, fp)

                else:
                    print('using previous CFBC')
                    with open(utils.join_path(assignment_file_path, cfbc_file1), 'rb') as fp:
                        CFBC1 = pickle.load(fp)

                    with open(utils.join_path(assignment_file_path, cfbc_file2), 'rb') as fp:
                        CFBC2 = pickle.load(fp)

                for node, value in sorted(CFBC1, key=lambda tup: tup[1], reverse=True):
                    round_select1.append(node)
                    if len(round_select1) == cs.MAX_SEED_SIZE:
                        break

                for node, value in sorted(CFBC2, key=lambda tup: tup[1], reverse=True):
                    round_select2.append(node)
                    if len(round_select2) == cs.MAX_SEED_SIZE:
                        break

            elif self.seed_alg == 'blast+cut_coeff+current_flow_closeness_centrality':
                cfcc_file1 = ('mss={}-{}.cfcc'.format(cs.MAX_SEED_SIZE,
                                                       bio_net.org1.org_id))

                cfcc_file2 = ('mss={}-{}.cfcc'.format(cs.MAX_SEED_SIZE,
                                                       bio_net.org2.org_id))

                connected_graphs1 = list(nx.connected_component_subgraphs(G1))
                connected_graphs2 = list(nx.connected_component_subgraphs(G2))

                comp1 = sorted(connected_graphs1, key=len, reverse=True)[0]
                comp2 = sorted(connected_graphs2, key=len, reverse=True)[0]

                if not utils.files_exist([cfcc_file1, cfcc_file2], assignment_file_path):
                    CFCC1 = nx.current_flow_closeness_centrality(comp1)
                    CFCC1 = [(node, value) for node, value in CFCC1.items()]

                    CFCC2 = nx.current_flow_closeness_centrality(comp2)
                    CFCC2 = [(node, value) for node, value in CFCC2.items()]

                    with open(utils.join_path(assignment_file_path, cfcc_file1), 'wb') as fp:
                        pickle.dump(CFCC1, fp)

                    with open(utils.join_path(assignment_file_path, cfcc_file2), 'wb') as fp:
                        pickle.dump(CFCC2, fp)

                else:
                    print('using previous CFCC')
                    with open(utils.join_path(assignment_file_path, cfcc_file1), 'rb') as fp:
                        CFCC1 = pickle.load(fp)

                    with open(utils.join_path(assignment_file_path, cfcc_file2), 'rb') as fp:
                        CFCC2 = pickle.load(fp)

                for node, value in sorted(CFCC1, key=lambda tup: tup[1], reverse=True):
                    round_select1.append(node)
                    if len(round_select1) == cs.MAX_SEED_SIZE:
                        break

                for node, value in sorted(CFCC2, key=lambda tup: tup[1], reverse=True):
                    round_select2.append(node)
                    if len(round_select2) == cs.MAX_SEED_SIZE:
                        break

            elif self.seed_alg == 'blast+cut_coeff+fiedler_vector':
                fv_file_name1 = ('mss={}-{}.fv'.format(cs.MAX_SEED_SIZE,
                                                       bio_net.org1.org_id))
                fv_file_name2 = ('mss={}-{}.fv'.format(cs.MAX_SEED_SIZE,
                                                       bio_net.org2.org_id))

                connected_graphs1 = list(nx.connected_component_subgraphs(G1))
                connected_graphs2 = list(nx.connected_component_subgraphs(G2))

                comp1 = sorted(connected_graphs1, key=len, reverse=True)[0]
                comp2 = sorted(connected_graphs2, key=len, reverse=True)[0]

                if not utils.files_exist([fv_file_name1, fv_file_name2], assignment_file_path):
                    fv1 = nx.linalg.algebraicconnectivity.fiedler_vector(comp1)
                    fv2 = nx.linalg.algebraicconnectivity.fiedler_vector(comp2)

                    with open(utils.join_path(assignment_file_path, fv_file_name1), 'wb') as fp:
                        pickle.dump(fv1, fp)

                    with open(utils.join_path(assignment_file_path, fv_file_name2), 'wb') as fp:
                        pickle.dump(fv2, fp)

                else:
                    print('using previous fieldler vectors')
                    with open(utils.join_path(assignment_file_path, fv_file_name1), 'rb') as fp:
                        fv1 = pickle.load(fp)

                    with open(utils.join_path(assignment_file_path, fv_file_name2), 'rb') as fp:
                        fv2 = pickle.load(fp)

                fv1 = zip(list(comp1.nodes), fv1)
                fv2 = zip(list(comp2.nodes), fv2)

                for node, value in sorted(fv1, key=lambda tup: tup[1]):
                    round_select1.append(node)
                    if len(round_select1) == cs.MAX_SEED_SIZE:
                        break

                for node, value in sorted(fv2, key=lambda tup: tup[1]):
                    round_select2.append(node)
                    if len(round_select2) == cs.MAX_SEED_SIZE:
                        break

            elif self.seed_alg == 'blast+cut_coeff+betweenness_centrality':
                bc_file_name1 = ('mss={}-{}.bc'.format(cs.MAX_SEED_SIZE, bio_net.org1.org_id))
                bc_file_name2 = ('mss={}-{}.bc'.format(cs.MAX_SEED_SIZE, bio_net.org2.org_id))

                if not utils.files_exist([bc_file_name1, bc_file_name2], assignment_file_path):
                    BC1 = nx.betweenness_centrality(G1)
                    BC2 = nx.betweenness_centrality(G2)

                    with open(utils.join_path(assignment_file_path, bc_file_name1), 'wb') as fp:
                        pickle.dump(BC1, fp)

                    with open(utils.join_path(assignment_file_path, bc_file_name2), 'wb') as fp:
                        pickle.dump(BC2, fp)

                else:
                    print('using previous betweenness centrality')
                    with open(utils.join_path(assignment_file_path, bc_file_name1), 'rb') as fp:
                        BC1 = pickle.load(fp)

                    with open(utils.join_path(assignment_file_path, bc_file_name2), 'rb') as fp:
                        BC2 = pickle.load(fp)

                BC1_list = [(node, value) for node, value in BC1.items()]
                BC2_list = [(node, value) for node, value in BC2.items()]
                for node, value in sorted(BC1_list, key=lambda tup: tup[1], reverse=True):
                    round_select1.append(node)
                    if len(round_select1) == cs.MAX_SEED_SIZE:
                        break

                for node, value in sorted(BC2_list, key=lambda tup: tup[1], reverse=True):
                    round_select2.append(node)
                    if len(round_select2) == cs.MAX_SEED_SIZE:
                        break

            elif self.seed_alg in ['blast+cut_coeff+pagerank_degree',
                                   'blast+cut_coeff+pagerank_blast']:

                max_ind = np.unravel_index(np.argmax(net_sim), net_sim.shape)
                max_ind = np.unravel_index(
                    np.argmax(net_pure_sim), net_pure_sim.shape)

                personalization1 = {
                    i: 0 for i in range(bio_net.org1.node_count)}
                personalization2 = {
                    i: 0 for i in range(bio_net.org2.node_count)}

                if self.seed_alg == 'blast+cut_coeff+pagerank_degree':
                    personalization1[node_idx1[-1]] = 1
                    personalization2[node_idx2[-1]] = 1

                elif self.seed_alg == 'blast+cut_coeff+pagerank_blast':
                    personalization1[max_ind[0]] = 1
                    personalization2[max_ind[1]] = 1

                PPR1 = nx.pagerank(G1, cs.SEED_PR_ALPHA, personalization1)
                PPR1 = [(node, value) for node, value in PPR1.items()]

                PPR2 = nx.pagerank(G2, cs.SEED_PR_ALPHA, personalization2)
                PPR2 = [(node, value) for node, value in PPR2.items()]

                for node, value in sorted(PPR1, key=lambda tup: tup[1]):
                    round_select1.append(node)
                    if len(round_select1) == cs.MAX_SEED_SIZE:
                        break

                for node, value in sorted(PPR2, key=lambda tup: tup[1]):
                    round_select2.append(node)
                    if len(round_select2) == cs.MAX_SEED_SIZE:
                        break

            algn_info['s1'] = round_select1
            algn_info['s2'] = round_select2


            if self.seed_alg == 'blast+cut_coeff+betweenness_centrality':
                # pair the base seed
                seed_blast_sim = net_sim[np.array(round_select1)[:, None],
                                   np.array(round_select2)]

                tmp = [x for x in seed_blast_sim.reshape(-1) if x < cs.BLAST_TH]
                print(max(tmp), len(tmp))
                blast_avg = sum(tmp) / len(tmp)


                seed_degree_diff = np.zeros(seed_blast_sim.shape)
                seed_factor_diff = np.zeros(seed_blast_sim.shape)
                seed_sim = np.zeros(seed_blast_sim.shape)

                for i in range(len(round_select1)):
                    for j in range(len(round_select1)):
                        n1, n2 = round_select1[i], round_select2[j]
                        seed_factor_diff[i][j] = abs(BC1[n1] - BC2[n2])
                        seed_degree_diff[i][j] = abs(bio_net.org1.degree[n1] -
                                                    bio_net.org2.degree[n2])



                seed_blast_sim = normalize(seed_blast_sim, norm='l1')
                seed_factor_diff = normalize(seed_factor_diff, norm='l1')
                seed_degree_diff = normalize(seed_degree_diff, norm='l1')

                for i in range(len(round_select1)):
                    for j in range(len(round_select1)):
                        # if(seed_blast_sim[i][j] > cs.BLAST_TH):
                        #     seed_sim[i][j] = seed_blast_sim[i][j]
                        #
                        # else:
                        #     seed_sim[i][j] = cs.BLAST_COEF * blast_avg + cs.SEED_FACTOR_COEF * seed_factor_diff[i][j]
                        seed_sim[i][j] = cs.BLAST_COEF * seed_blast_sim[i][j] + cs.SEED_FACTOR_COEF * seed_factor_diff[i][j] + cs.DEGREE_COEF * seed_degree_diff[i][j]


                # now connect clusters
                assignment_file_name = ('seed-mss={}-blastCoef={}-'
                                        'degreeCoef={}-seed_factor_coef={}-'
                                        'cut_coef={}-normalized-{}-{}-'
                                        'sim={}.json'.format(cs.MAX_SEED_SIZE,
                                            cs.BLAST_COEF, cs.DEGREE_COEF,
                                            cs.SEED_FACTOR_COEF, self.cut_coef,
                                            bio_net.org1.org_id, bio_net.org2.org_id,
                                            bio_net.similarity_mode))

                # # now connect clusters
                # assignment_file_name = ('seed-mss={}-blastCoef={}-'
                #                         'blast_threshold={}-seed_factor_coef={}-'
                #                         'cut_coef={}-normalized-{}-{}-'
                #                         'sim={}.json'.format(cs.MAX_SEED_SIZE,
                #                             cs.BLAST_COEF, cs.BLAST_TH,
                #                             cs.SEED_FACTOR_COEF, self.cut_coef,
                #                             bio_net.org1.org_id, bio_net.org2.org_id,
                #                             bio_net.similarity_mode))

            else:
                # pair the base seed
                seed_sim = net_sim[np.array(round_select1)[:, None],
                                   np.array(round_select2)]

                # now connect clusters
                assignment_file_name = ('seed-mss={}-sim={}-cut_coef={}-hungarian-{}-{}'
                                    '.json'.format(cs.MAX_SEED_SIZE, bio_net.similarity_mode,
                                    self.cut_coef, bio_net.org1.org_id, bio_net.org2.org_id))


            # Add alpha parameter if seed is choosed with pagerank
            if self.seed_alg in ['blast+cut_coeff+pagerank_degree',
                                 'blast+cut_coeff+pagerank_blast']:
                assignment_file_name = 'alpha={}-{}'.format(cs.SEED_PR_ALPHA,
                                                            assignment_file_name)

            pl1, pl2 = utils.linear_sum_assignment(-seed_sim, file_name=assignment_file_name, path_name=assignment_file_path)

            message = 'seeds aligned for "{}" algorithm'.format(self.method)
            utils.print_log(message)

            # save seed pairs
            new_pairs = []
            for i in range(len(pl1)):
                n1 = round_select1[int(pl1[i])]
                n2 = round_select2[int(pl2[i])]
                new_pairs.append(
                    (n1, n2, bio_net.similarity[bio_net.v_ind(n1, n2)]))
                node_paired1[n1] = True
                node_paired2[n2] = True

            # remove some pairs
            sorted_pairs = sorted(new_pairs, key=lambda x: x[2])
            cut_point = max((int((1 - cs.SEED_KEEP_RATIO) * len(new_pairs)) - 1), 0)
            for pair in sorted_pairs[:cut_point]:
                n1 = pair[0]
                n2 = pair[1]
                node_paired1[n1] = False
                node_paired2[n2] = False
                new_pairs.remove(pair)
            algn_info['pairs'] = new_pairs
            algn_infos.append(algn_info)
            algn_info = {}

        pairs += new_pairs
        reached_neighs1 = set.union(
            *[set(bio_net.org1.neighbors(x[0])) for x in pairs])
        remains1 = [x for x in node_idx1 if not node_paired1[x]]
        # paired_nodes1 = set([x[0] for x in pairs])
        reached_neighs2 = set.union(
            *[set(bio_net.org2.neighbors(x[1])) for x in pairs])
        remains2 = [x for x in node_idx2 if not node_paired2[x]]
        # paired_nodes2 = set([x[1] for x in pairs])

        # pair_lookup1 = {x[0]:x[1] for x in pairs}
        # pair_lookup2 = {x[1]:x[0] for x in pairs}
        paired1 = set([x[0] for x in pairs])
        paired2 = set([x[1] for x in pairs])

        if self.extend_alg == 'multiple+cut':
            # finished = False
            finished = ((len(remains1) == 0) or (len(remains2) == 0))
            while not finished:
                # extend to new pairs
                round_select1 = []
                round_reverse1 = {}
                for i in reversed(remains1):
                    if i in reached_neighs1:
                        round_reverse1[i] = len(round_select1)
                        round_select1.append(i)
                        node_selected1[i] = True
                    if len(round_select1) >= cs.MAX_EXTEND_SIZE:
                        break
                if len(round_select1) < cs.MAX_EXTEND_SIZE:
                    for i in base_select1:
                        if not node_paired1[i]:
                            round_select1.append(i)
                            node_selected1[i] = True
                        if len(round_select1) >= cs.MAX_EXTEND_SIZE:
                            break
                algn_info['s1'] = round_select1

                round_select2 = []
                round_reverse2 = {}
                for i in reversed(remains2):
                    if i in reached_neighs2:
                        round_reverse2[i] = len(round_select2)
                        round_select2.append(i)
                        node_selected2[i] = True
                    if len(round_select2) >= cs.MAX_EXTEND_SIZE:
                        break
                if len(round_select2) < cs.MAX_EXTEND_SIZE:
                    for i in base_select2:
                        if not node_paired2[i]:
                            round_select2.append(i)
                            node_selected2[i] = True
                        if len(round_select2) >= cs.MAX_EXTEND_SIZE:
                            break
                algn_info['s2'] = round_select2

                # primary similarity for the selection
                base_sim = net_sim[np.array(round_select1)[:, None],
                                   np.array(round_select2)]
                base_sim = utils.normalize(base_sim)

                # new similarity based on pairs
                topo_sim = np.zeros((len(round_select1), len(round_select2)))
                for pair in pairs:
                    nodes1 = [x for x in bio_net.org1.neighbors(
                        pair[0]) if x in round_select1]
                    nodes2 = [x for x in bio_net.org2.neighbors(
                        pair[1]) if x in round_select2]
                    for n1 in nodes1:
                        for n2 in nodes2:
                            topo_sim[round_reverse1[n1]
                                     ][round_reverse2[n2]] += 1
                            if self.method == "seedexcost":
                                topo_sim[round_reverse1[n1]
                                         ][round_reverse2[n2]
                                           ] += 2 * cs.BAD_EDGE_COST

                if self.method == "seedexcost":
                    cost_vec1 = np.array(
                        [len([x for x in bio_net.org1.neighbors(y)
                              if x in paired1])
                         for y in round_select1])
                    cost_sim1 = cs.BAD_EDGE_COST * cost_vec1.reshape(
                        len(round_select1), 1).repeat(
                        len(round_select2), axis=1)
                    topo_sim -= cost_sim1

                    cost_vec2 = np.array(
                        [len([x for x in bio_net.org2.neighbors(y)
                              if x in paired2])
                         for y in round_select2])
                    cost_sim2 = cs.BAD_EDGE_COST * cost_vec2.reshape(
                        1, len(round_select2)).repeat(
                        len(round_select1), axis=0)
                    topo_sim -= cost_sim2

                topo_sim = utils.normalize(topo_sim)

                # now generate final similarity
                select_sim = ((cs.TOPO_STRENGTH * topo_sim) +
                              ((1 - cs.TOPO_STRENGTH) * base_sim))

                # now connect clusters
                if self.matching_alg == 'hungarian':
                    pl1, pl2 = utils.linear_sum_assignment(-select_sim)
                elif self.matching_alg == 'greedy':
                    pl1, pl2 = utils.greedy_assignment(select_sim)

                # save new pairs
                new_pairs = []
                for i in range(len(pl1)):
                    n1 = round_select1[int(pl1[i])]
                    reached_neighs1.update(bio_net.org1.neighbors(n1))
                    n2 = round_select2[int(pl2[i])]
                    reached_neighs2.update(bio_net.org2.neighbors(n2))
                    new_pairs.append(
                        (n1, n2, bio_net.similarity[bio_net.v_ind(n1, n2)]))
                    node_paired1[n1] = True
                    node_paired2[n2] = True

                # remove some pairs
                sorted_pairs = sorted(new_pairs, key=lambda x: x[2])
                cut_point = max((int((1 - cs.EXTEND_KEEP_RATIO) *
                                     len(new_pairs)) - 1), 0)
                for pair in sorted_pairs[:cut_point]:
                    n1 = pair[0]
                    n2 = pair[1]
                    node_paired1[n1] = False
                    node_paired2[n2] = False
                    new_pairs.remove(pair)
                algn_info['pairs'] = new_pairs
                algn_infos.append(algn_info)
                algn_info = {}

                remains1 = [x for x in remains1 if not node_paired1[x]]
                remains2 = [x for x in remains2 if not node_paired2[x]]

                finished = ((False not in node_paired1.values()) or
                            (False not in node_paired2.values()))

                paired1 = set([x[0] for x in pairs])
                paired2 = set([x[1] for x in pairs])

                pairs += new_pairs

        elif self.extend_alg == 'single_extend':
            scores_dict = {}
            scores_record_dict = {}
            scores_heap = []
            # scores_set = set()

            # generate seed scores
            for seed_pair in pairs:
                for i1 in bio_net.org1.neighbors(seed_pair[0]):
                    for i2 in bio_net.org2.neighbors(seed_pair[1]):
                        # if not already aligned
                        if not(node_paired1[i1] or node_paired2[i2]):
                            if self.extend_sim_method == 'common_neighbor':
                                # common neighbor score
                                scores_dict[(i1, i2)] = scores_dict.get(
                                    (i1, i2), 0) + 1
                            if self.extend_sim_method == 'jaccard':
                                # Jaccard's coef score
                                d1 = bio_net.org1.degree[i1]
                                d2 = bio_net.org2.degree[i2]
                                scores_dict[(i1, i2)] = scores_dict.get(
                                    (i1, i2), 0) + (1 / (d1 + d2))
                            if self.extend_sim_method == 'jaccard-neg':
                                d1 = bio_net.org1.degree[i1]
                                d2 = bio_net.org2.degree[i2]
                                scores_dict[(i1, i2)] = scores_dict.get(
                                    (i1, i2), 0) + (1 / (1 + abs(d1 - d2)))
                            if self.extend_sim_method == 'jaccard-nmul':
                                # Jaccard's coef score
                                d1 = bio_net.org1.degree[i1]
                                d2 = bio_net.org2.degree[i2]
                                scores_dict[(i1, i2)] = scores_dict.get(
                                    (i1, i2), 0) + (1 / ((d1 + d2) * (1 + abs(d1 - d2))))
                            if self.extend_sim_method == 'adamic':
                                # Adamic's coef score
                                ne1 = seed_pair[0]
                                ne2 = seed_pair[1]
                                nd1 = bio_net.org1.degree[ne1]
                                nd2 = bio_net.org2.degree[ne2]
                                scores_dict[(i1, i2)] = scores_dict.get(
                                    (i1, i2), 0) + (1 / np.log(nd1 + nd2))
                            if self.extend_sim_method == 'adamic-neg':
                                ne1 = seed_pair[0]
                                ne2 = seed_pair[1]
                                nd1 = bio_net.org1.degree[ne1]
                                nd2 = bio_net.org2.degree[ne2]
                                scores_dict[(i1, i2)] = scores_dict.get(
                                    (i1, i2), 0) + (1 / (1 + np.log(1 + abs(nd1 - nd2))))
                            if self.extend_sim_method == 'adamic-nmul':
                                # Adamic's coef score
                                ne1 = seed_pair[0]
                                ne2 = seed_pair[1]
                                nd1 = bio_net.org1.degree[ne1]
                                nd2 = bio_net.org2.degree[ne2]
                                scores_dict[(i1, i2)] = scores_dict.get(
                                    (i1, i2), 0) + (1 / (np.log(nd1 + nd2) * (1 + np.log(1 + abs(nd1 - nd2)))))

            # build data structures needed
            for pair in scores_dict:
                score = scores_dict[pair]
                scores_record_dict[score] = scores_record_dict.get(
                    score, []) + [pair]
            for score in scores_record_dict:
                # max heap
                heapq.heappush(scores_heap, -score)
            heapq.heappush(scores_heap, 0)

            # sorted blast
            sim_scores = []
            for n1 in range(bio_net.org1.node_count):
                for n2 in range(bio_net.org2.node_count):
                    sim_scores.append(
                        (n1, n2, bio_net.similarity[bio_net.v_ind(n1, n2)]))
            sim_scores.sort(key=lambda x: -x[2])
            sim_pointer = 0

            # start extend procedure
            message = 'starting single extend procedure'
            utils.print_log(message)

            loops_counter = [0, 0, 0]
            while not (all([node_paired1[x] for x in node_paired1]) or
                       all([node_paired2[x] for x in node_paired2])):
                loops_counter[0] += 1
                message = 'paired {} so far; {} loops ({},{})'.format(
                    len([x for x in node_paired1 if node_paired1[x]]),
                    loops_counter[0], loops_counter[1], loops_counter[2],)
                utils.print_log(message, mode='progress')

                score = abs(heapq.heappop(scores_heap))
                if score != 0:
                    loops_counter[1] += 1
                    nexts = scores_record_dict[score]

                    # proper like max func
                    next_pair = min(nexts, key=lambda x: (
                        abs(bio_net.org1.degree[x[0]] -
                            bio_net.org2.degree[x[1]]),
                        bio_net.org1.degree[x[0]] + bio_net.org2.degree[x[1]],
                        random.random()))
                    # note: add other max functions!

                    # current_score = scores_dict[next_pair]
                    del scores_dict[next_pair]
                    if len(nexts) > 1:
                        heapq.heappush(scores_heap, -score)
                        scores_record_dict[score].remove(next_pair)
                    else:
                        del scores_record_dict[score]

                else:
                    # restore elemnt in heap
                    heapq.heappush(scores_heap, 0)

                    # choose from blast
                    while (node_paired1[sim_scores[sim_pointer][0]] or
                           node_paired2[sim_scores[sim_pointer][1]]):
                        loops_counter[2] += 1
                        sim_pointer += 1

                    chosen = sim_scores[sim_pointer]
                    next_pair = (chosen[0], chosen[1])

                if not (node_paired1[next_pair[0]] or
                        node_paired2[next_pair[1]]):
                    # add new pair
                    # print(next_pair)
                    pairs.append((
                        next_pair[0],
                        next_pair[1],
                        bio_net.similarity[bio_net.v_ind(next_pair[0],
                                                         next_pair[1])]))
                    node_paired1[next_pair[0]] = True
                    node_paired2[next_pair[1]] = True

                    # update data structures
                    for i1 in bio_net.org1.neighbors(next_pair[0]):
                        for i2 in bio_net.org2.neighbors(next_pair[1]):
                            old_score = scores_dict.get((i1, i2), 0)
                            # update old score
                            old_record = scores_record_dict.get(old_score, [])
                            if old_score > 0:
                                del scores_dict[(i1, i2)]
                                if len(old_record) > 1:
                                    scores_record_dict[old_score].remove(
                                        (i1, i2))
                                elif len(old_record) == 1:
                                    del scores_record_dict[old_score]
                                    # delete from heap
                                    # this location is still O(n)
                                    heap_index = scores_heap.index(-old_score)
                                    # O(log(n)) deletion
                                    # https://stackoverflow.com/questions/10162679/python-delete-element-from-heap
                                    scores_heap[heap_index] = scores_heap[-1]
                                    scores_heap.pop()
                                    if heap_index < len(scores_heap):
                                        heapq._siftup(scores_heap, heap_index)
                                        heapq._siftdown(scores_heap, 0,
                                                        heap_index)

                            # if not already aligned
                            if not(node_paired1[i1] or node_paired2[i2]):
                                if self.extend_sim_method == 'common_neighbor':
                                    # common neighbor score
                                    new_score = old_score + 1
                                if self.extend_sim_method == 'jaccard':
                                    # Jaccard's coef score
                                    d1 = bio_net.org1.degree[i1]
                                    d2 = bio_net.org2.degree[i2]
                                    new_score = old_score + (1 / (d1 + d2))
                                if self.extend_sim_method == 'jaccard-neg':
                                    # Jaccard's coef score
                                    d1 = bio_net.org1.degree[i1]
                                    d2 = bio_net.org2.degree[i2]
                                    new_score = old_score + \
                                        (1 / (1 + abs(d1 - d2)))
                                if self.extend_sim_method == 'jaccard-nmul':
                                    # Jaccard's coef score
                                    d1 = bio_net.org1.degree[i1]
                                    d2 = bio_net.org2.degree[i2]
                                    new_score = old_score + \
                                        (1 / ((d1 + d2) * (1 + abs(d1 - d2))))
                                if self.extend_sim_method == 'adamic':
                                    # Jaccard's coef score
                                    ne1 = next_pair[0]
                                    ne2 = next_pair[1]
                                    nd1 = bio_net.org1.degree[ne1]
                                    nd2 = bio_net.org2.degree[ne2]
                                    new_score = old_score + \
                                        (1 / np.log(nd1 + nd2))
                                if self.extend_sim_method == 'adamic-neg':
                                    # Jaccard's coef score
                                    ne1 = next_pair[0]
                                    ne2 = next_pair[1]
                                    nd1 = bio_net.org1.degree[ne1]
                                    nd2 = bio_net.org2.degree[ne2]
                                    new_score = old_score + \
                                        (1 / (1 + np.log(1 + abs(nd1 - nd2))))
                                if self.extend_sim_method == 'adamic-nmul':
                                    # Jaccard's coef score
                                    ne1 = next_pair[0]
                                    ne2 = next_pair[1]
                                    nd1 = bio_net.org1.degree[ne1]
                                    nd2 = bio_net.org2.degree[ne2]
                                    new_score = old_score + \
                                        (1 / (np.log(nd1 + nd2) *
                                              (1 + np.log(1 + abs(nd1 - nd2)))))

                                scores_dict[(i1, i2)] = new_score

                                # update new score
                                new_record = scores_record_dict.get(
                                    new_score, [])
                                if len(new_record) == 0:
                                    heapq.heappush(scores_heap, -new_score)
                                    scores_record_dict[new_score] = [(i1, i2)]
                                else:
                                    scores_record_dict[new_score].append(
                                        (i1, i2))

            message = 'single extend procedure finished'
            utils.print_log(message, mode='end_progress')

        return (pairs, algn_infos)

    def seed_extend_align_manager(self, bio_net):
        pairs, algn_infos = self.seed_extend_align(bio_net)
        csv_infos = [[
            'round',
            ('selected nodes from' + bio_net.org1.org_id),
            ('selected nodes from' + bio_net.org2.org_id),
            'aligned nodes in round',
            ('edges inside round component from' + bio_net.org1.org_id),
            ('edges inside round component from' + bio_net.org2.org_id),
            'edges inside combined round component',
            ('edges between round components from' + bio_net.org1.org_id),
            ('edges between round components from' + bio_net.org2.org_id),
            'edges between combined round components',
        ]]
        prev1 = []
        prev2 = []
        for index, algn_info in enumerate(algn_infos):
            csv_info = []
            csv_info.append(index)

            csv_info.append(len(algn_info['s1']))
            csv_info.append(len(algn_info['s2']))

            roundpairs = algn_info['pairs']
            csv_info.append(len(roundpairs))

            sel1 = [x[0] for x in roundpairs]
            adj1 = bio_net.org1.adjacency[
                np.array(sel1)[:, None],
                np.array(sel1)]
            csv_info.append(sum(sum(adj1)) / 2)

            sel2 = [x[1] for x in roundpairs]
            adj2 = bio_net.org2.adjacency[
                np.array(sel2)[:, None],
                np.array(sel2)]
            csv_info.append(sum(sum(adj2)) / 2)

            adjcomb = np.multiply(adj1, adj2)
            csv_info.append(sum(sum(adjcomb)) / 2)

            if (index > 0):
                round_adj1 = bio_net.org1.adjacency[
                    np.array(prev1)[:, None],
                    np.array(sel1)]
                csv_info.append(sum(sum(round_adj1)))
            else:
                csv_info.append(0)
            prev1 += sel1

            if (index > 0):
                round_adj2 = bio_net.org2.adjacency[
                    np.array(prev2)[:, None],
                    np.array(sel2)]
                csv_info.append(sum(sum(round_adj2)))
            else:
                csv_info.append(0)
            prev2 += sel2

            if (index > 0):
                round_adjcomb = np.multiply(round_adj1, round_adj2)
                csv_info.append(sum(sum(round_adjcomb)))
            else:
                csv_info.append(0)

            csv_infos.append(csv_info)

        csv_file = utils.join_path(cs.SVG_PATH,
                                   '<{},{}>extend_log{}_{}.csv'.format(
                                       bio_net.org1.org_id,
                                       bio_net.org2.org_id,
                                       bio_net.status,
                                       self.method))
        utils.write_csv(csv_infos, csv_file)

        return pairs

    def verify_alignment(self, pairs):
        v_pairs = []
        aligned1 = set()
        aligned2 = set()
        for pair in pairs:
            n1 = pair[0]
            n2 = pair[1]
            if (n1 not in aligned1) and (n2 not in aligned2):
                aligned1.add(n1)
                aligned2.add(n2)
                v_pairs.append(pair)
        return v_pairs

    def calculate_frobenius_norm(self, pairs, bio_net):
        m = bio_net.org1.node_count
        n = bio_net.org2.node_count

        alignment_adjacency = np.zeros((m, n))
        for pair in pairs:
            n1, n2, s = pair
            alignment_adjacency[n1][n2] = 1

        mat1 = np.dot(alignment_adjacency, bio_net.org2.adjacency)
        mat2 = np.dot(mat1, alignment_adjacency.transpose())
        mat3 = np.subtract(bio_net.org1.adjacency, mat2)

        self.frobenius = np.linalg.norm(mat3, 'fro')
        return self.frobenius

    def align(self, bio_net, check=True, file_path=cs.JSON_PATH):
        # choose aligner's algorithm
        if self.method == 'greedy':
            # greedy algorithm
            self.aligner = self.greedy_align
        elif self.method == 'sgreedy':
            # semi-greedy algorithm
            self.aligner = self.semi_greedy_align
        elif self.method in ['clstr', 'rclst', 'cclst', 'l2clstr',
                             'l2extend', 'l2selextend', 'l2mincpl',
                             'l2mincplextend', 'l2mincplselextend',
                             'l2maxcut', 'l2maxcutextend',
                             'l2maxcutselextend', 'l2brutecut',
                             'l2brutecutextend', 'l2brutecutselextend']:
            # maximum weight matching after noisy spectral clustering
            # maximum weight matching after repetetive devide clustering
            # maximum weight matching after component spectral clustering
            # maximum weight matching after component spectral clustering
            self.aligner = self.cluster_align
        elif self.method == 'max':
            # maximum weight matching on raw data
            self.aligner = self.max_weight_align
        # TODO: This one not implemented in the new codes yet.
        # elif self.method == 'compclstr':
        #     # maximum weight matching after component complement clustering
        #     self.aligner = self.cluster_align
        elif self.method == 'isoN':
            # isorankN code interface
            self.aligner = self.isorankN_align
            bio_net.status = ''
            bio_net.similarity_mode = 'raw_blast'
        elif self.method == 'NETAL':
            # NETAL code interface
            self.aligner = self.NETAL_align
            bio_net.status = ''
            bio_net.similarity_mode = 'raw_blast'
        elif self.method == 'pinalog':
            # pinalog code interface
            self.aligner = self.pinalog_align
            bio_net.status = ''
            bio_net.similarity_mode = 'raw_blast'
        elif self.method in ['seedex']:
            # basic seed & extend alignment
            bio_net.status += '+<ts={},skr={},ekr={},mss={},mes={}>'.format(
                cs.TOPO_STRENGTH, cs.SEED_KEEP_RATIO, cs.EXTEND_KEEP_RATIO,
                cs.MAX_SEED_SIZE, cs.MAX_EXTEND_SIZE)
            self.aligner = self.seed_extend_align_manager
            self.cut_coef = 'degree'
            self.seed_alg = 'blast+cut_coeff'
            self.extend_alg = 'multiple+cut'
            self.matching_alg = 'hungarian'
        elif self.method in ['seedex-greedy']:
            # basic seed & extend alignment
            bio_net.status += '+<ts={},skr={},ekr={},mss={},mes={}>'.format(
                cs.TOPO_STRENGTH, cs.SEED_KEEP_RATIO, cs.EXTEND_KEEP_RATIO,
                cs.MAX_SEED_SIZE, cs.MAX_EXTEND_SIZE)
            self.aligner = self.seed_extend_align_manager
            self.cut_coef = 'degree'
            self.seed_alg = 'blast+cut_coeff'
            self.extend_alg = 'multiple+cut'
            self.matching_alg = 'greedy'

        elif self.method in ['seedexneigh']:
            bio_net.status += '+<ts={},skr={},ekr={},mss={},mes={}>'.format(
                cs.TOPO_STRENGTH, cs.SEED_KEEP_RATIO, cs.EXTEND_KEEP_RATIO,
                cs.MAX_SEED_SIZE, cs.MAX_EXTEND_SIZE)
            self.aligner = self.seed_extend_align_manager
            self.cut_coef = 'neighbor-degree'
            self.seed_alg = 'blast+cut_coeff'
            self.extend_alg = 'multiple+cut'
            self.matching_alg = 'hungarian'
        elif self.method in ['seedexcost']:
            bio_net.status += ('+<ts={},skr={},ekr={},'
                               'mss={},mes={},bec={}>').format(
                cs.TOPO_STRENGTH, cs.SEED_KEEP_RATIO, cs.EXTEND_KEEP_RATIO,
                cs.MAX_SEED_SIZE, cs.MAX_EXTEND_SIZE, cs.BAD_EDGE_COST)
            self.aligner = self.seed_extend_align_manager
            self.cut_coef = 'degree'
            self.seed_alg = 'blast+cut_coeff'
            self.extend_alg = 'multiple+cut'
            self.matching_alg = 'hungarian'
        elif self.method in ['seedexblast']:
            bio_net.status += ('+<ts={},ekr={},'
                               'mes={},bc={}>').format(
                cs.TOPO_STRENGTH, cs.EXTEND_KEEP_RATIO,
                cs.MAX_EXTEND_SIZE, cs.BLAST_CUT)
            self.aligner = self.seed_extend_align_manager
            self.cut_coef = 'degree'
            self.seed_alg = 'blast'
            self.extend_alg = 'multiple+cut'
            self.matching_alg = 'hungarian'
        elif self.method in ['seedexsingle']:
            bio_net.status += '+<skr={},mss={}>'.format(
                cs.SEED_KEEP_RATIO, cs.MAX_SEED_SIZE)
            self.aligner = self.seed_extend_align_manager
            self.cut_coef = 'degree'
            self.seed_alg = 'blast+cut_coeff'
            self.extend_alg = 'single_extend'
            self.extend_sim_method = 'common_neighbor'
        elif self.method in ['seedex-SA']:
            bio_net.status += '+<skr={},mss={}>'.format(
                cs.SEED_KEEP_RATIO, cs.MAX_SEED_SIZE)
            self.aligner = self.seed_extend_SA_align
            self.seed_alg = 'blast+cut_coeff+betweenness_centrality'
            self.extend_sim_method = 'common_neighbor'
        elif self.method in ['seedexCC']:
            bio_net.status += '+<skr={},mss={}>'.format(
                cs.SEED_KEEP_RATIO, cs.MAX_SEED_SIZE)
            self.aligner = self.seed_extend_align_manager
            self.cut_coef = 'degree'
            self.seed_alg = 'blast+cut_coeff+closeness_centrality'
            self.extend_alg = 'single_extend'
            self.extend_sim_method = 'common_neighbor'
        elif self.method in ['seedexBC']:
            bio_net.status += '+<skr={},mss={},blastCoef={},' 'degreeCoef={},seed_factor_coef={}>'.format(cs.SEED_KEEP_RATIO,  cs.MAX_SEED_SIZE, cs.BLAST_COEF, cs.DEGREE_COEF, cs.SEED_FACTOR_COEF)
            self.aligner = self.seed_extend_align_manager
            self.cut_coef = 'degree'
            self.seed_alg = 'blast+cut_coeff+betweenness_centrality'
            self.extend_alg = 'single_extend'
            self.extend_sim_method = 'common_neighbor'
        elif self.method in ['seedexCFBC']:
            bio_net.status += '+<skr={},mss={}>'.format(
                cs.SEED_KEEP_RATIO, cs.MAX_SEED_SIZE)
            self.aligner = self.seed_extend_align_manager
            self.cut_coef = 'degree'
            self.seed_alg = 'blast+cut_coeff+current_flow_betweenness_centrality'
            self.extend_alg = 'single_extend'
            self.extend_sim_method = 'common_neighbor'
        elif self.method in ['seedexCFCC']:
            bio_net.status += '+<skr={},mss={}>'.format(
                cs.SEED_KEEP_RATIO, cs.MAX_SEED_SIZE)
            self.aligner = self.seed_extend_align_manager
            self.cut_coef = 'degree'
            self.seed_alg = 'blast+cut_coeff+current_flow_closeness_centrality'
            self.extend_alg = 'single_extend'
            self.extend_sim_method = 'common_neighbor'
        elif self.method in ['seedexSLE']:
            bio_net.status += '+<skr={},mss={}>'.format(
                cs.SEED_KEEP_RATIO, cs.MAX_SEED_SIZE)
            self.aligner = self.seed_extend_align_manager
            self.cut_coef = 'degree'
            self.seed_alg = 'blast+cut_coeff+fiedler_vector'
            self.extend_alg = 'single_extend'
            self.extend_sim_method = 'common_neighbor'
        elif self.method in ['seedexsingle-prd']:
            bio_net.status += '+<skr={},mss={},spra={}>'.format(
                cs.SEED_KEEP_RATIO, cs.MAX_SEED_SIZE, cs.SEED_PR_ALPHA)
            self.aligner = self.seed_extend_align_manager
            self.cut_coef = 'degree'
            self.seed_alg = 'blast+cut_coeff+pagerank_degree'
            self.extend_alg = 'single_extend'
            self.extend_sim_method = 'common_neighbor'
        elif self.method in ['seedexsingle-prb']:
            bio_net.status += '+<skr={},mss={},spra={}>'.format(
                cs.SEED_KEEP_RATIO, cs.MAX_SEED_SIZE, cs.SEED_PR_ALPHA)
            self.aligner = self.seed_extend_align_manager
            self.cut_coef = 'degree'
            self.seed_alg = 'blast+cut_coeff+pagerank_blast'
            self.extend_alg = 'single_extend'
            self.extend_sim_method = 'common_neighbor'

        elif self.method in ['seedexproper']:
            bio_net.status += ('+<ts={},ekr={},'
                               'mes={},bc={}>').format(
                cs.TOPO_STRENGTH, cs.EXTEND_KEEP_RATIO,
                cs.MAX_EXTEND_SIZE, cs.BLAST_CUT)
            self.aligner = self.seed_extend_align_manager
            self.cut_coef = 'degree'
            self.seed_alg = 'blast'
            self.extend_alg = 'single_extend'
            self.extend_sim_method = 'common_neighbor'
        elif self.method in ['seedexsingle-jac']:
            bio_net.status += '+<skr={},mss={}>'.format(
                cs.SEED_KEEP_RATIO, cs.MAX_SEED_SIZE)
            self.aligner = self.seed_extend_align_manager
            self.cut_coef = 'degree'
            self.seed_alg = 'blast+cut_coeff'
            self.extend_alg = 'single_extend'
            self.extend_sim_method = 'jaccard'

        elif self.method in ['seedexCC-jac']:
            bio_net.status += '+<skr={},mss={}>'.format(
                cs.SEED_KEEP_RATIO, cs.MAX_SEED_SIZE)
            self.aligner = self.seed_extend_align_manager
            self.cut_coef = 'degree'
            self.seed_alg = 'blast+cut_coeff+closeness_centrality'
            self.extend_alg = 'single_extend'
            self.extend_sim_method = 'jaccard'

        elif self.method in ['seedexSLE-jac']:
            bio_net.status += '+<skr={},mss={}>'.format(
                cs.SEED_KEEP_RATIO, cs.MAX_SEED_SIZE)
            self.aligner = self.seed_extend_align_manager
            self.cut_coef = 'degree'
            self.seed_alg = 'blast+cut_coeff+fiedler_vector'
            self.extend_alg = 'single_extend'
            self.extend_sim_method = 'jaccard'
        elif self.method in ['seedexBC-jac']:
            bio_net.status += '+<skr={},mss={},blastCoef={},' 'degreeCoef={},seed_factor_coef={}>'.format(cs.SEED_KEEP_RATIO,  cs.MAX_SEED_SIZE, cs.BLAST_COEF, cs.DEGREE_COEF, cs.SEED_FACTOR_COEF)
            self.aligner = self.seed_extend_align_manager
            self.cut_coef = 'degree'
            self.seed_alg = 'blast+cut_coeff+betweenness_centrality'
            self.extend_alg = 'single_extend'
            self.extend_sim_method = 'jaccard'

        elif self.method in ['seedexCFBC-jac']:
            bio_net.status += '+<skr={},mss={}>'.format(
                cs.SEED_KEEP_RATIO, cs.MAX_SEED_SIZE)
            self.aligner = self.seed_extend_align_manager
            self.cut_coef = 'degree'
            self.seed_alg = 'blast+cut_coeff+current_flow_betweenness_centrality'
            self.extend_alg = 'single_extend'
            self.extend_sim_method = 'jaccard'

        elif self.method in ['seedexCFCC-jac']:
            bio_net.status += '+<skr={},mss={}>'.format(
                cs.SEED_KEEP_RATIO, cs.MAX_SEED_SIZE)
            self.aligner = self.seed_extend_align_manager
            self.cut_coef = 'degree'
            self.seed_alg = 'blast+cut_coeff+current_flow_closeness_centrality'
            self.extend_alg = 'single_extend'
            self.extend_sim_method = 'jaccard'

        elif self.method in ['seedexsingle-prd-jac']:
            bio_net.status += '+<skr={},mss={},spra={}>'.format(
                cs.SEED_KEEP_RATIO, cs.MAX_SEED_SIZE, cs.SEED_PR_ALPHA)
            self.aligner = self.seed_extend_align_manager
            self.cut_coef = 'degree'
            self.seed_alg = 'blast+cut_coeff+pagerank_degree'
            self.extend_alg = 'single_extend'
            self.extend_sim_method = 'jaccard'
        elif self.method in ['seedexsingle-prb-jac']:
            bio_net.status += '+<skr={},mss={},spra={}>'.format(
                cs.SEED_KEEP_RATIO, cs.MAX_SEED_SIZE, cs.SEED_PR_ALPHA)
            self.aligner = self.seed_extend_align_manager
            self.cut_coef = 'degree'
            self.seed_alg = 'blast+cut_coeff+pagerank_blast'
            self.extend_alg = 'single_extend'
            self.extend_sim_method = 'jaccard'

        elif self.method in ['seedexsingle-jacn']:
            bio_net.status += '+<skr={},mss={}>'.format(
                cs.SEED_KEEP_RATIO, cs.MAX_SEED_SIZE)
            self.aligner = self.seed_extend_align_manager
            self.cut_coef = 'degree'
            self.seed_alg = 'blast+cut_coeff'
            self.extend_alg = 'single_extend'
            self.extend_sim_method = 'jaccard-neg'
        elif self.method in ['seedexsingle-jacnn']:
            bio_net.status += '+<skr={},mss={}>'.format(
                cs.SEED_KEEP_RATIO, cs.MAX_SEED_SIZE)
            self.aligner = self.seed_extend_align_manager
            self.cut_coef = 'degree'
            self.seed_alg = 'blast+cut_coeff'
            self.extend_alg = 'single_extend'
            self.extend_sim_method = 'jaccard-nmul'
        elif self.method in ['seedexsingle-ada']:
            bio_net.status += '+<skr={},mss={}>'.format(
                cs.SEED_KEEP_RATIO, cs.MAX_SEED_SIZE)
            self.aligner = self.seed_extend_align_manager
            self.cut_coef = 'degree'
            self.seed_alg = 'blast+cut_coeff'
            self.extend_alg = 'single_extend'
            self.extend_sim_method = 'adamic'

        elif self.method in ['seedexCC-ada']:
            bio_net.status += '+<skr={},mss={}>'.format(
                cs.SEED_KEEP_RATIO, cs.MAX_SEED_SIZE)
            self.aligner = self.seed_extend_align_manager
            self.cut_coef = 'degree'
            self.seed_alg = 'blast+cut_coeff+closeness_centrality'
            self.extend_alg = 'single_extend'
            self.extend_sim_method = 'adamic'

        elif self.method in ['seedexSLE-ada']:
            bio_net.status += '+<skr={},mss={}>'.format(
                cs.SEED_KEEP_RATIO, cs.MAX_SEED_SIZE)
            self.aligner = self.seed_extend_align_manager
            self.cut_coef = 'degree'
            self.seed_alg = 'blast+cut_coeff+fiedler_vector'
            self.extend_alg = 'single_extend'
            self.extend_sim_method = 'adamic'
        elif self.method in ['seedexBC-ada']:
            bio_net.status += '+<skr={},mss={},blastCoef={},' 'degreeCoef={},seed_factor_coef={}>'.format(cs.SEED_KEEP_RATIO,  cs.MAX_SEED_SIZE, cs.BLAST_COEF, cs.DEGREE_COEF, cs.SEED_FACTOR_COEF)
            self.aligner = self.seed_extend_align_manager
            self.cut_coef = 'degree'
            self.seed_alg = 'blast+cut_coeff+betweenness_centrality'
            self.extend_alg = 'single_extend'
            self.extend_sim_method = 'adamic'

        elif self.method in ['seedexCFBC-ada']:
            bio_net.status += '+<skr={},mss={}>'.format(
                cs.SEED_KEEP_RATIO, cs.MAX_SEED_SIZE)
            self.aligner = self.seed_extend_align_manager
            self.cut_coef = 'degree'
            self.seed_alg = 'blast+cut_coeff+current_flow_betweenness_centrality'
            self.extend_alg = 'single_extend'
            self.extend_sim_method = 'adamic'

        elif self.method in ['seedexCFCC-ada']:
            bio_net.status += '+<skr={},mss={}>'.format(
                cs.SEED_KEEP_RATIO, cs.MAX_SEED_SIZE)
            self.aligner = self.seed_extend_align_manager
            self.cut_coef = 'degree'
            self.seed_alg = 'blast+cut_coeff+current_flow_closeness_centrality'
            self.extend_alg = 'single_extend'
            self.extend_sim_method = 'adamic'

        elif self.method in ['seedexsingle-prd-ada']:
            bio_net.status += '+<skr={},mss={},spra={}>'.format(
                cs.SEED_KEEP_RATIO, cs.MAX_SEED_SIZE, cs.SEED_PR_ALPHA)
            self.aligner = self.seed_extend_align_manager
            self.cut_coef = 'degree'
            self.seed_alg = 'blast+cut_coeff+pagerank_degree'
            self.extend_alg = 'single_extend'
            self.extend_sim_method = 'adamic'
        elif self.method in ['seedexsingle-prb-ada']:
            bio_net.status += '+<skr={},mss={},spra={}>'.format(
                cs.SEED_KEEP_RATIO, cs.MAX_SEED_SIZE, cs.SEED_PR_ALPHA)
            self.aligner = self.seed_extend_align_manager
            self.cut_coef = 'degree'
            self.seed_alg = 'blast+cut_coeff+pagerank_blast'
            self.extend_alg = 'single_extend'
            self.extend_sim_method = 'adamic'

        elif self.method in ['seedexsingle-adann']:
            bio_net.status += '+<skr={},mss={}>'.format(
                cs.SEED_KEEP_RATIO, cs.MAX_SEED_SIZE)
            self.aligner = self.seed_extend_align_manager
            self.cut_coef = 'degree'
            self.seed_alg = 'blast+cut_coeff'
            self.extend_alg = 'single_extend'
            self.extend_sim_method = 'adamic-nmul'
        elif self.method == 'CGRAAL':
            # CGRAAL code interface
            self.aligner = self.CGRAAL_align
            bio_net.status = ''
            bio_net.similarity_mode = 'raw_blast'
        elif self.method == 'GRAAL':
            # GRAAL code interface
            self.aligner = self.GRAAL_align
            bio_net.status = ''
            bio_net.similarity_mode = 'raw_blast'
        elif self.method == 'MIGRAAL':
            # GRAAL code interface
            self.aligner = self.MIGRAAL_align
            bio_net.status = ''
            bio_net.similarity_mode = 'raw_blast'
        elif self.method == 'HubAlign':
            # GRAAL code interface
            self.aligner = self.HubAlign_align
            bio_net.status = ''
            bio_net.similarity_mode = 'raw_blast'
        elif self.method == 'MAGNA':
            # GRAAL code interface
            self.aligner = self.MAGNA_align
            bio_net.status = ''
            bio_net.similarity_mode = 'raw_blast'
        elif self.method == 'PROPER':
            # GRAAL code interface
            self.aligner = self.PROPER_align
            bio_net.status = ''
            bio_net.similarity_mode = 'raw_blast'
        elif self.method == 'SPINAL-I':
            # GRAAL code interface
            self.aligner = self.SPINAL_align
            self.spinal_alg = 'I'
            bio_net.status = ''
            bio_net.similarity_mode = 'raw_blast'
        elif self.method == 'SPINAL-II':
            # GRAAL code interface
            self.aligner = self.SPINAL_align
            self.spinal_alg = 'II'
            bio_net.status = ''
            bio_net.similarity_mode = 'raw_blast'
        elif self.method == 'optnet':
            # GRAAL code interface
            self.aligner = self.optnet_align
            bio_net.status += ('+<alg={},cxr={},cxs={},mur={},mus={}'
                               ',oor={},pos={},gen={},hlc={},tlim={}>').format(
                cs.optnet_alg, cs.optnet_cxrate, cs.optnet_cxswappb,
                cs.optnet_mutrate, cs.optnet_mutswappb, cs.optnet_oneobjrate,
                cs.optnet_popsize, cs.optnet_generations,
                cs.optnet_hillclimbiters, cs.optnet_timelimit)
            bio_net.similarity_mode = 'raw_blast'
        elif self.method == 'moduleAlign':
            # GRAAL code interface
            self.aligner = self.moduleAlign_align
            bio_net.status += ('+<alp={}>').format(cs.moduleAlign_alpha)
            bio_net.similarity_mode = 'raw_blast'

        # file name
        paired_nodes = '{}-{}-{}{}_paired_nodes_{}.json'.format(
            bio_net.org1.org_id, bio_net.org2.org_id, bio_net.similarity_mode,
            bio_net.status, self.method)
        paired_edges = '{}-{}-{}{}_paired_edges_{}.json'.format(
            bio_net.org1.org_id, bio_net.org2.org_id, bio_net.similarity_mode,
            bio_net.status, self.method)

        file_names = [paired_nodes, paired_edges]
        file_path = os.path.join(file_path, self.method)

        if not os.path.exists(file_path):
            os.makedirs(file_path)

        if (check and utils.files_exist(file_names, file_path)):
            message = ('using saved alignment for "{}" algorithm '
                       'in "{}"').format(self.method, file_names)
            utils.print_log(message)

            # create alignment object
            self.alignment = Alignment(paired_nodes, paired_edges, self.method)

            # load nodes and edges
            pairs, pair_edges = self.alignment.load_alignment()

        else:
            message = 'starting alignment with "{}" algorithm'.format(
                self.method)
            utils.print_log(message)

            pairs = self.aligner(bio_net)
            pairs = self.verify_alignment(pairs)

            # save nodes
            utils.write_json(pairs, utils.join_path(file_path, paired_nodes))

            message = ('nodes aligned using "{}" algorithm, '
                       'now finding remaining links').format(self.method)
            utils.print_log(message)

            pair_edges = self.find_paired_edges(pairs, bio_net, file_path)

            # save edges
            utils.write_json(pair_edges, utils.join_path(
                file_path, paired_edges))
            # create alignment object
            self.alignment = Alignment(paired_nodes, paired_edges, self.method)

            message = 'alignment with "{}" algorithm finished'.format(
                self.method)
            utils.print_log(message)

        # calculate frobenius norm
        self.calculate_frobenius_norm(pairs, bio_net)

        message = 'now calculating alignment measures'
        utils.print_log(message)

        message = ('<{}> edges remained after alignment with "{}" '
                   'algorithm').format(len(pair_edges), self.method)
        utils.print_log(message)

        self.calculate_measures(pairs, pair_edges, bio_net)

        try:
            message = '"{}" algorithm, LCCS: {}'.format(self.method, self.lccs)
            utils.print_log(message)
            message = '"{}" algorithm, CE: {}'.format(self.method, self.ce)
            utils.print_log(message)
            message = ('"{}" algorithm, edge correctnes'
                       ' (EC): {}').format(self.method, self.ec)
            utils.print_log(message)
            message = ('"{}" algorithm, induced conserved structure'
                       ' (ICS): {}').format(self.method, self.ics)
            utils.print_log(message)
            message = '"{}" algorithm, S^3: {}'.format(self.method, self.s3)
            utils.print_log(message)
            message = '"{}" algorithm, GOC: {}'.format(self.method, self.GOC)
            utils.print_log(message)
            message = '"{}" algorithm, PWS1: {}'.format(self.method, self.PWS1)
            utils.print_log(message)
            message = '"{}" algorithm, PWS2: {}'.format(self.method, self.PWS2)
            utils.print_log(message)
            message = '"{}" algorithm, NBS: {}'.format(self.method, self.nbs)
            utils.print_log(message)
        except Exception as e:
            pass

        return self.alignment
