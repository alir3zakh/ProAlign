"""
this module contains Organism class that stores all the information related
to and organism that we use, such as it's protein sequences and ppi network.
it also includes the BioNet class that stores the relations between two
Organisms such as their Blast score
"""

import numpy as np
import scipy.sparse as sparse
import sklearn.cluster as cluster
import scipy.sparse.linalg as slnlg

import utils
import interface
import visualize
import constants as cs


class OrgCluster():
    """docstring for OrgCluster"""

    def __init__(self, labels1, label_cnt1, labels2, label_cnt2):
        self.labels1 = labels1
        self.label_cnt1 = label_cnt1
        self.labels2 = labels2
        self.label_cnt2 = label_cnt2

        self.cl_size1 = [0] * self.label_cnt1
        self.node_dic1 = {}
        self.cl_dic1 = {}
        for index in range(len(self.labels1)):
            # mark node[index] is in which cluster and which index
            self.node_dic1[index] = (
                self.labels1[index], self.cl_size1[labels1[index]])
            self.cl_dic1[(labels1[index],
                          self.cl_size1[labels1[index]])] = index
            self.cl_size1[labels1[index]] += 1

        self.cl_size2 = [0] * self.label_cnt2
        self.node_dic2 = {}
        self.cl_dic2 = {}
        for index in range(len(self.labels2)):
            # mark node[index] is in which cluster and which index
            self.node_dic2[index] = (
                self.labels2[index], self.cl_size2[labels2[index]])
            self.cl_dic2[(labels2[index],
                          self.cl_size2[labels2[index]])] = index
            self.cl_size2[labels2[index]] += 1


class Organism():
    """docstring for Organism"""

    def __init__(self, nodes_file, edges_file, org_id):
        self.nodes_file = nodes_file
        self.edges_file = edges_file
        self.org_id = org_id
        self.file_name = 'organism-{}.bak'.format(org_id)

        node_data = utils.load_json(nodes_file)
        self.id_to_node = {ind: node for ind, node in enumerate(node_data)}
        self.node_to_id = {node: ind for ind, node in enumerate(node_data)}

        edge_data = utils.load_json(edges_file)

        # dimensions of Incidence Matrix
        self.node_count = len(node_data)
        # self.edge_count = len(edge_data)
        self.dimensions = (len(node_data), len(edge_data))

        # # incidence matrix would be too big -> ignored
        # self.incidence = np.zeros(dimensions)

        self.edges = set()
        self.adjacency = np.zeros((self.node_count, self.node_count))
        for index, edge in enumerate(edge_data):
            n1 = self.node_to_id[edge[0]]
            n2 = self.node_to_id[edge[1]]
            self.edges.add((min(n1, n2), max(n1, n2)))
            # ignore edge weights
            self.adjacency[n1][n2] = 1
            self.adjacency[n2][n1] = 1

        self.degree = sum(self.adjacency)

        # # P = D^-1 * A
        # self.transition = self.adjacency / self.degree

        message = ('{} - Organism imported successfully').format(org_id)
        utils.print_log(message)

        message = ('{} - number of nodes and edges = {}').format(
            org_id, self.dimensions)
        utils.print_log(message)

        utils.save_object(self, self.file_name)

        # visualize.visualise_org_degree(self)

    def neighbors(self, node_id):
        return [i for i, x in enumerate(self.adjacency[node_id]) if x == 1]

    def components(self):
        # return components of an organism
        adj = sparse.csr_matrix(self.adjacency)
        initial_labels = sparse.csgraph.connected_components(adj)
        return initial_labels  # n_components, labels

    def repetetive_devide(self):
        # divide using spectral clustering
        divide = cluster.SpectralClustering(
            n_clusters=2, eigen_solver='arpack',
            affinity="precomputed")

        # cluster each part to the limit with components as initial clusters
        adj = sparse.csr_matrix(self.adjacency)
        initial_labels = sparse.csgraph.connected_components(adj)
        labels = initial_labels[1]
        newlabel = max(labels) + 1

        # find component size for all labels
        sizes = {}
        for label in labels:
            sizes[label] = sizes.get(label, 0) + 1

        keys = list(sizes)
        while sizes[max(sizes, key=sizes.get)] > cs.MAX_CLUSTER_SIZE:
            newkeys = []
            rmkeys = []
            for label in keys:
                if sizes[label] > cs.MAX_CLUSTER_SIZE:
                    selector = np.array(
                        [i for i, x in enumerate(labels) if x == label])
                    ladj = self.adjacency[selector[:, None], selector]
                    division = divide.fit_predict(ladj)
                    changes = [selector[i]
                               for i, x in enumerate(division) if x == 1]
                    sizes[label] = sizes.get(label, 0) - len(changes)
                    sizes[newlabel] = sizes.get(newlabel, 0) + len(changes)
                    for change in changes:
                        labels[change] = newlabel
                    newkeys.append(newlabel)
                    newlabel += 1
                else:
                    rmkeys.append(label)
            keys = [x for x in keys if x not in rmkeys] + newkeys

        return labels, len(sizes)

    def component_clustering(self):
        # component spectral clustering
        if self.method == 'cclst':
            clustering = cluster.SpectralClustering

        adj = sparse.csr_matrix(self.adjacency)
        initial_labels = sparse.csgraph.connected_components(adj)
        labels = initial_labels[1]
        newlabel = max(labels)

        # find component size for all labels
        sizes = {}
        for label in labels:
            sizes[label] = sizes.get(label, 0) + 1

        keys = list(sizes)
        while sizes[max(sizes, key=sizes.get)] > cs.MAX_CLUSTER_SIZE:
            newkeys = set()
            rmkeys = set()
            for label in keys:
                if sizes[label] > cs.MAX_CLUSTER_SIZE:
                    # predict number of clusters
                    nccluster = (sizes[label] // cs.MAX_CLUSTER_SIZE) + 1
                    # spectral clustering
                    ccluster = clustering(
                        n_clusters=nccluster, eigen_solver='arpack',
                        affinity="precomputed")
                    selector = np.array(
                        [i for i, x in enumerate(labels) if x == label])
                    ladj = self.adjacency[selector[:, None], selector]
                    cclusters = ccluster.fit_predict(ladj)
                    changes = [(selector[i], x)
                               for i, x in enumerate(cclusters) if x != 0]
                    sizes[label] = sizes.get(label, 0) - len(changes)
                    for (change, x) in changes:
                        sizes[newlabel + x] = sizes.get(newlabel + x, 0) + 1
                        labels[change] = newlabel + x
                        newkeys.add(newlabel + x)
                    newlabel += (nccluster - 1)
                else:
                    rmkeys.add(label)
            keys = [x for x in keys if x not in rmkeys] + list(newkeys)

        return labels, len(sizes)

    def rep_l2_clustering(self):
        # repetetive l2gap clustering
        adj = sparse.csr_matrix(self.adjacency)
        initial_labels = sparse.csgraph.connected_components(adj)
        labels = initial_labels[1]
        newlabel = max(labels)

        # visualization graph
        graph = {'size': self.node_count, 'clusters': {}}
        cluster_lookup = {}

        # find component size for all labels
        sizes = {}
        for label in labels:
            sizes[label] = sizes.get(label, 0) + 1

        for label in set(labels):
            graph['clusters'][label] = {'size': sizes[label], 'clusters': {}}
            cluster_lookup[label] = graph['clusters'][label]

        keys = list(sizes)
        while sizes[max(sizes, key=sizes.get)] > cs.MAX_CLUSTER_SIZE:
            newkeys = set()
            rmkeys = set()
            for label in keys:
                if sizes[label] > cs.MAX_CLUSTER_SIZE:
                    # predict number of clusters
                    nccluster = (sizes[label] // cs.MAX_CLUSTER_SIZE) + 1
                    selector = np.array(
                        [i for i, x in enumerate(labels) if x == label])
                    # cluster using l2
                    ladj = self.adjacency[selector[:, None], selector]
                    lapl = np.diag(sum(ladj)) - ladj
                    lapl = sparse.csr_matrix(lapl)
                    # eValue, eVector = np.linalg.eig(lapl)
                    eValue, eVector = slnlg.eigs(lapl, k=3, which='SM')
                    idx = eValue.argsort()
                    eValue = eValue[idx]
                    eVector = eVector[:, idx]
                    lambda2 = eVector[:, 1]
                    # now cluster using gap
                    l2idx = list(lambda2.argsort())
                    l2rev = [l2idx.index(x) for x in range(len(l2idx))]
                    l2srt = lambda2[l2idx]
                    l2gap = [(l2srt[x + 1] - l2srt[x])
                             for x in range(len(l2srt) - 1)]
                    gapidx = np.argsort(l2gap)
                    bounds = set(gapidx[-(nccluster - 1):])
                    clbl = 0
                    clstrs = []
                    for i in range(len(l2srt)):
                        clstrs.append(clbl)
                        if i in bounds:
                            clbl += 1
                    cclusters = np.array(clstrs)[l2rev]
                    changes = [(selector[i], x)
                               for i, x in enumerate(cclusters) if x != 0]
                    current_cluster = cluster_lookup[label]
                    sizes[label] = sizes.get(label, 0) - len(changes)
                    current_cluster['clusters'][label] = {
                        'size': sizes[label], 'clusters': {}}
                    cluster_lookup[label] = current_cluster['clusters'][label]
                    for (change, x) in changes:
                        sizes[newlabel + x] = sizes.get(newlabel + x, 0) + 1
                        labels[change] = newlabel + x
                        newkeys.add(newlabel + x)
                    for label in range(newlabel + 1, newlabel + nccluster):
                        current_cluster['clusters'][label] = {
                            'size': sizes[label], 'clusters': {}}
                        cluster_lookup[label] = current_cluster['clusters'
                                                                ][label]
                    newlabel += (nccluster - 1)
                else:
                    rmkeys.add(label)
            keys = [x for x in keys if x not in rmkeys] + list(newkeys)

        visualize.cluster_sunburst(self, graph, 'rep_l2_clustering')
        return labels, len(sizes)

    def min_couple_l2(self):
        # repetetive l2gap clustering
        adj = sparse.csr_matrix(self.adjacency)
        initial_labels = sparse.csgraph.connected_components(adj)
        labels = initial_labels[1]
        newlabel = max(labels)

        # visualization graph
        graph = {'size': self.node_count, 'clusters': {}}
        cluster_lookup = {}

        # find component size for all labels
        sizes = {}
        for label in labels:
            sizes[label] = sizes.get(label, 0) + 1

        for label in set(labels):
            graph['clusters'][label] = {'size': sizes[label], 'clusters': {}}
            cluster_lookup[label] = graph['clusters'][label]

        keys = list(sizes)
        while sizes[max(sizes, key=sizes.get)] > cs.MAX_CLUSTER_SIZE:
            newkeys = set()
            rmkeys = set()
            for label in keys:
                if sizes[label] > cs.MAX_CLUSTER_SIZE:
                    selector = np.array(
                        [i for i, x in enumerate(labels) if x == label])
                    # cluster using l2
                    ladj = self.adjacency[selector[:, None], selector]
                    lapl = np.diag(sum(ladj)) - ladj
                    lapl = sparse.csr_matrix(lapl)
                    # eValue, eVector = np.linalg.eig(lapl)
                    eValue, eVector = slnlg.eigs(lapl, k=3, which='SM')
                    idx = eValue.argsort()
                    eValue = eValue[idx]
                    eVector = eVector[:, idx]
                    lambda2 = eVector[:, 1]
                    # now cluster using gap
                    l2idx = list(lambda2.argsort())
                    l2rev = [l2idx.index(x) for x in range(len(l2idx))]
                    l2srt = lambda2[l2idx]
                    l2gap = [(l2srt[x + 1] - l2srt[x])
                             for x in range(len(l2srt) - 1)]
                    gapidx = np.argsort(l2gap)
                    # store start and ends
                    bound_selected = {i: True for i in gapidx}
                    # now start coupling from min gaps
                    for gap in gapidx:
                        start = gap
                        end = gap
                        cl_lngth = 2
                        while cl_lngth < cs.MAX_CLUSTER_SIZE:
                            if bound_selected.get(start - 1, True) is False:
                                start -= 1
                                cl_lngth += 1
                            elif bound_selected.get(end + 1, True) is False:
                                end += 1
                                cl_lngth += 1
                            else:
                                break
                        if cl_lngth < cs.MAX_CLUSTER_SIZE:
                            bound_selected[gap] = False

                    bounds = set(
                        [x for x in gapidx if bound_selected[x] is True])
                    clbl = 0
                    clstrs = []
                    for i in range(len(l2srt)):
                        clstrs.append(clbl)
                        if i in bounds:
                            clbl += 1
                    nccluster = clbl + 1
                    cclusters = np.array(clstrs)[l2rev]
                    changes = [(selector[i], x)
                               for i, x in enumerate(cclusters) if x != 0]
                    current_cluster = cluster_lookup[label]
                    sizes[label] = sizes.get(label, 0) - len(changes)
                    current_cluster['clusters'][label] = {
                        'size': sizes[label], 'clusters': {}}
                    cluster_lookup[label] = current_cluster['clusters'][label]
                    for (change, x) in changes:
                        sizes[newlabel + x] = sizes.get(newlabel + x, 0) + 1
                        labels[change] = newlabel + x
                        newkeys.add(newlabel + x)
                    for label in range(newlabel + 1, newlabel + nccluster):
                        current_cluster['clusters'][label] = {
                            'size': sizes[label], 'clusters': {}}
                        cluster_lookup[label] = current_cluster['clusters'
                                                                ][label]
                    newlabel += (nccluster - 1)
                else:
                    rmkeys.add(label)
            keys = [x for x in keys if x not in rmkeys] + list(newkeys)

        visualize.cluster_sunburst(self, graph, 'min_couple_l2_clustering')
        return labels, len(sizes)

    def max_cut_l2(self):
        # repetetive l2gap clustering
        adj = sparse.csr_matrix(self.adjacency)
        initial_labels = sparse.csgraph.connected_components(adj)
        labels = initial_labels[1]
        newlabel = max(labels)

        # visualization graph
        graph = {'size': self.node_count, 'clusters': {}}
        cluster_lookup = {}

        # find component size for all labels
        sizes = {}
        for label in labels:
            sizes[label] = sizes.get(label, 0) + 1

        for label in set(labels):
            graph['clusters'][label] = {'size': sizes[label], 'clusters': {}}
            cluster_lookup[label] = graph['clusters'][label]

        keys = list(sizes)
        while sizes[max(sizes, key=sizes.get)] > cs.MAX_CLUSTER_SIZE:
            newkeys = set()
            rmkeys = set()
            for label in keys:
                if sizes[label] > cs.MAX_CLUSTER_SIZE:
                    selector = np.array(
                        [i for i, x in enumerate(labels) if x == label])
                    # cluster using l2
                    ladj = self.adjacency[selector[:, None], selector]
                    lapl = np.diag(sum(ladj)) - ladj
                    lapl = sparse.csr_matrix(lapl)
                    # eValue, eVector = np.linalg.eig(lapl)
                    eValue, eVector = slnlg.eigs(lapl, k=3, which='SM')
                    idx = eValue.argsort()
                    eValue = eValue[idx]
                    eVector = eVector[:, idx]
                    lambda2 = eVector[:, 1]
                    # now cluster using gap
                    l2idx = list(lambda2.argsort())
                    l2rev = [l2idx.index(x) for x in range(len(l2idx))]
                    l2srt = lambda2[l2idx]
                    l2gap = [(l2srt[x + 1] - l2srt[x])
                             for x in range(len(l2srt) - 1)]
                    gapidx = np.argsort(l2gap)
                    # store start and ends
                    bound_selected = {i: False for i in gapidx}
                    # now start coupling from min gaps
                    for gap in reversed(gapidx):
                        start = gap
                        end = gap
                        cl_lngth = 2
                        while cl_lngth < cs.MAX_CLUSTER_SIZE:
                            if bound_selected.get(start - 1, True) is False:
                                start -= 1
                                cl_lngth += 1
                            elif bound_selected.get(end + 1, True) is False:
                                end += 1
                                cl_lngth += 1
                            else:
                                break
                        if cl_lngth >= cs.MAX_CLUSTER_SIZE:
                            bound_selected[gap] = True

                    bounds = set(
                        [x for x in gapidx if bound_selected[x] is True])
                    clbl = 0
                    clstrs = []
                    for i in range(len(l2srt)):
                        clstrs.append(clbl)
                        if i in bounds:
                            clbl += 1
                    nccluster = clbl + 1
                    cclusters = np.array(clstrs)[l2rev]
                    changes = [(selector[i], x)
                               for i, x in enumerate(cclusters) if x != 0]
                    current_cluster = cluster_lookup[label]
                    sizes[label] = sizes.get(label, 0) - len(changes)
                    current_cluster['clusters'][label] = {
                        'size': sizes[label], 'clusters': {}}
                    cluster_lookup[label] = current_cluster['clusters'][label]
                    for (change, x) in changes:
                        sizes[newlabel + x] = sizes.get(newlabel + x, 0) + 1
                        labels[change] = newlabel + x
                        newkeys.add(newlabel + x)
                    for label in range(newlabel + 1, newlabel + nccluster):
                        current_cluster['clusters'][label] = {
                            'size': sizes[label], 'clusters': {}}
                        cluster_lookup[label] = current_cluster['clusters'
                                                                ][label]
                    newlabel += (nccluster - 1)
                else:
                    rmkeys.add(label)
            keys = [x for x in keys if x not in rmkeys] + list(newkeys)

        visualize.cluster_sunburst(self, graph, 'max_cut_l2_clustering')
        return labels, len(sizes)

    def max_brutecut_l2(self):
        # repetetive l2gap clustering
        adj = sparse.csr_matrix(self.adjacency)
        initial_labels = sparse.csgraph.connected_components(adj)
        labels = initial_labels[1]
        newlabel = max(labels)

        # visualization graph
        graph = {'size': self.node_count, 'clusters': {}}
        cluster_lookup = {}

        # find component size for all labels
        sizes = {}
        for label in labels:
            sizes[label] = sizes.get(label, 0) + 1

        for label in set(labels):
            graph['clusters'][label] = {'size': sizes[label], 'clusters': {}}
            cluster_lookup[label] = graph['clusters'][label]

        keys = list(sizes)
        while sizes[max(sizes, key=sizes.get)] > cs.MAX_CLUSTER_SIZE:
            newkeys = set()
            rmkeys = set()
            for label in keys:
                if sizes[label] > cs.MAX_CLUSTER_SIZE:
                    selector = np.array(
                        [i for i, x in enumerate(labels) if x == label])
                    # cluster using l2
                    ladj = self.adjacency[selector[:, None], selector]
                    lapl = np.diag(sum(ladj)) - ladj
                    lapl = sparse.csr_matrix(lapl)
                    # eValue, eVector = np.linalg.eig(lapl)
                    eValue, eVector = slnlg.eigs(lapl, k=3, which='SM')
                    idx = eValue.argsort()
                    eValue = eValue[idx]
                    eVector = eVector[:, idx]
                    lambda2 = eVector[:, 1]
                    # now cluster using gap
                    l2idx = list(lambda2.argsort())
                    l2rev = [l2idx.index(x) for x in range(len(l2idx))]
                    l2srt = lambda2[l2idx]
                    l2gap = [(l2srt[x + 1] - l2srt[x])
                             for x in range(len(l2srt) - 1)]
                    gapidx = np.argsort(l2gap)
                    # store start and ends
                    bound_selected = {i: False for i in gapidx}
                    # now start coupling from min gaps
                    for gap in reversed(gapidx):
                        max_cl_lngth = 0
                        chosen_bounds = [0]
                        chosen_bounds += [x for x in bound_selected if
                                          bound_selected[x]]
                        chosen_bounds += [len(gapidx)]
                        for x in range(len(chosen_bounds) - 1):
                            max_cl_lngth = max(
                                max_cl_lngth,
                                (chosen_bounds[x + 1] - chosen_bounds[x]))
                        if max_cl_lngth >= cs.MAX_CLUSTER_SIZE:
                            bound_selected[gap] = True

                    bounds = set(
                        [x for x in gapidx if bound_selected[x] is True])
                    clbl = 0
                    clstrs = []
                    for i in range(len(l2srt)):
                        clstrs.append(clbl)
                        if i in bounds:
                            clbl += 1
                    nccluster = clbl + 1
                    cclusters = np.array(clstrs)[l2rev]
                    changes = [(selector[i], x)
                               for i, x in enumerate(cclusters) if x != 0]
                    current_cluster = cluster_lookup[label]
                    sizes[label] = sizes.get(label, 0) - len(changes)
                    current_cluster['clusters'][label] = {
                        'size': sizes[label], 'clusters': {}}
                    cluster_lookup[label] = current_cluster['clusters'][label]
                    for (change, x) in changes:
                        sizes[newlabel + x] = sizes.get(newlabel + x, 0) + 1
                        labels[change] = newlabel + x
                        newkeys.add(newlabel + x)
                    for label in range(newlabel + 1, newlabel + nccluster):
                        current_cluster['clusters'][label] = {
                            'size': sizes[label], 'clusters': {}}
                        cluster_lookup[label] = current_cluster['clusters'
                                                                ][label]
                    newlabel += (nccluster - 1)
                else:
                    rmkeys.add(label)
            keys = [x for x in keys if x not in rmkeys] + list(newkeys)

        visualize.cluster_sunburst(self, graph, 'max_brutecut_l2_clustering')
        return labels, len(sizes)

    def cluster_network(self, method):
        self.method = method
        # noisy spectral clustering
        if method == 'clstr':
            clustering = cluster.SpectralClustering(
                n_clusters=cs.CLUSTERS_COUNT,
                eigen_solver='arpack',
                affinity="precomputed")
            labels = clustering.fit_predict(
                self.adjacency +
                (cs.NOISE_STRENGTH *
                    np.random.rand(self.node_count,
                                   self.node_count)))

            return (labels, cs.CLUSTERS_COUNT)

        # repetetive devide clustering
        if method == 'rclst':
            labels, label_cnt = self.component_clustering()

            return (labels, label_cnt)

        # component spectral clustering
        if method == 'cclst':
            labels, label_cnt = self.repetetive_devide()

            return (labels, label_cnt)

        # repetetive l2gap clustering
        if method in ['l2clstr', 'l2extend', 'l2selextend']:
            labels, label_cnt = self.rep_l2_clustering()

            return (labels, label_cnt)

        # min couple l2gap clustering
        if method in ['l2mincpl', 'l2mincplextend', 'l2mincplselextend']:
            labels, label_cnt = self.min_couple_l2()

        # max cut l2gap clustering
        if method in ['l2maxcut', 'l2maxcutextend', 'l2maxcutselextend']:
            labels, label_cnt = self.max_cut_l2()

            return (labels, label_cnt)

        # max cut l2gap clustering
        if method in ['l2brutecut', 'l2brutecutextend', 'l2brutecutselextend']:
            labels, label_cnt = self.max_brutecut_l2()

            return (labels, label_cnt)


class BioNet():
    """docstring for BioNet"""

    def __init__(self, org1, org2, similarity_mode, power_alpha=cs.ALPHA_BIAS):
        self.org1 = org1
        self.org2 = org2
        self.similarity_mode = similarity_mode
        self.power_alpha = power_alpha

        # dimension of similarity matrix (stored as vector)
        self.dim_sim = (org1.node_count * org2.node_count)

        self.alpha_rec = ''
        self.similarity = None

        if similarity_mode in ['blast_power', 'just_power']:
            self.alpha_rec = '-<alpha={}>'.format(power_alpha)

        file_name = '{}-{}-{}{}_raw_scores.npy'.format(
            org1.org_id, org2.org_id, similarity_mode, self.alpha_rec)
        self.raw_np_file = utils.join_path(cs.NP_PATH, file_name)

        file_name = '{}-{}-{}{}_scores.npy'.format(
            org1.org_id, org2.org_id, similarity_mode, self.alpha_rec)
        self.np_file = utils.join_path(cs.NP_PATH, file_name)


        self.status = self.alpha_rec
        self.calculate_rel_blast_matrix()

        if utils.file_exists(file_name, path_name=cs.NP_PATH):
            message = 'using calculated similarity from {}'.format(file_name)
            utils.print_log(message)

            self.similarity = utils.load_np(self.np_file)
            self.blast_sim = utils.load_np(self.raw_np_file)

        else:
            message = 'calculating similarity ({})'.format(similarity_mode)
            utils.print_log(message)

            if similarity_mode == 'raw_blast':
                self.calculate_blast_matrix()
                self.similarity = self.blast_sim_n
                self.store_similarity_matrix()
            elif similarity_mode == 'rel_blast':
                self.similarity = self.blast_sim_n_rel
                self.store_similarity_matrix(self.np_file)
            elif similarity_mode == 'blast_power':
                self.calculate_blast_matrix()
                self.calculate_power_method(power_alpha, self.np_file)
                self.similarity = self.power_met_sim
                self.store_similarity_matrix(self.np_file)
            elif similarity_mode == 'just_power':
                self.generate_dummy_matrix()
                self.blast_sim_n = self.dummy_sim
                self.calculate_power_method(power_alpha, self.np_file)
                self.similarity = self.power_met_sim
                self.store_similarity_matrix(self.np_file)
            elif similarity_mode == 'no_sim':
                self.generate_dummy_matrix()
                self.similarity = self.dummy_sim
                self.store_similarity_matrix(self.np_file)

    def v_ind(self, i, j):
        return ((i * self.org2.node_count) + j)

    # generate dummy similarity score
    def store_similarity_matrix(self):
        utils.write_np(self.similarity, self.np_file)
        utils.write_np(self.blast_sim, self.raw_np_file)

        message = 'calculated similarity stored in "{}"'.format(self.np_file)
        utils.print_log(message)

    # generate dummy similarity score
    def generate_dummy_matrix(self):
        # self.dummy_sim = np.ones(self.dim_sim) / self.dim_sim
        self.dummy_sim = utils.normalize(
            self.org1.degree.reshape(1, -1) *
            self.org2.degree.reshape(-1, 1)).reshape(1, -1)[0]

    # calculate the normalized blast matrix from blast scores
    @utils.time_it
    def calculate_blast_matrix(self):
        # blast similarity measure
        self.blast_sim = interface.blast_xml_to_matrix(self)

        # normalize blast matrix
        self.blast_sim_n = utils.normalize(self.blast_sim)

    # calculate the normalized relative blast matrix from blast scores
    @utils.time_it
    def calculate_rel_blast_matrix(self):
        file_name = '{}-{}-{}_scores.npy'.format(
            self.org1.org_id, self.org2.org_id, 'rel_blast')

        if utils.file_exists(file_name, path_name=cs.NP_PATH):
            message = 'using saved relative blast from {}'.format(file_name)
            utils.print_log(message)

            self.blast_sim_n_rel = utils.load_np(self.np_file)
            self.blast_sim = utils.load_np(self.raw_np_file)
            # self.blast_sim = interface.blast_xml_to_matrix(self)

        else:
            # blast similarity measure
            blast_sim = interface.blast_xml_to_matrix(self)
            self.blast_sim = blast_sim

            blast_1 = interface.self_blast_xml_to_vec(self.org1)
            blast_1[blast_1 == 0] = 1
            blast_1 = np.array([blast_1]).repeat(self.org2.node_count,
                                                 axis=0).T

            blast_2 = interface.self_blast_xml_to_vec(self.org2)
            blast_2[blast_2 == 0] = 1
            blast_2 = np.array([blast_2]).repeat(self.org1.node_count, axis=0)

            blast_sim = blast_sim.reshape(self.org1.node_count,
                                          self.org2.node_count)
            blast_sim = blast_sim / np.power((blast_1 * blast_2), 0.5)
            blast_sim = blast_sim.reshape(self.dim_sim)

            # normalize blast matrix
            self.blast_sim_n_rel = utils.normalize(blast_sim)
            np_file = utils.join_path(cs.NP_PATH, file_name)
            utils.write_np(self.blast_sim_n_rel, np_file)

    @utils.time_it
    def calculate_power_method(self, alpha, np_file):
        # if not self.blast_sim_n:
        try:
            self.blast_sim_n
        except Exception:
            raise Exception('blast scores are absent in network object, '
                            'run calculate blast matrix before power method.')

        # power method on blast similarity measure
        self.power_met_sim = self.blast_sim_n

        message = ('Starting power method iterations on blast output')
        utils.print_log(message)

        total = len(self.org1.edges) * len(self.org2.edges)

        iteration_count = 0

        while True:

            calculations = 0
            iteration_count += 1

            temp = np.zeros(self.dim_sim)

            for e1 in self.org1.edges:
                for e2 in self.org2.edges:

                    i, u = e1
                    j, v = e2
                    Ni = self.org1.degree[i]
                    Nj = self.org2.degree[j]
                    Nu = self.org1.degree[u]
                    Nv = self.org2.degree[v]

                    # suppose i,j pair and reveres
                    u_v = self.power_met_sim[self.v_ind(u, v)] / (Nu * Nv)
                    temp[self.v_ind(i, j)] += u_v

                    i_j = self.power_met_sim[self.v_ind(i, j)] / (Ni * Nj)
                    temp[self.v_ind(u, v)] += i_j

                    # suppose i,v pair and reveres
                    u_j = self.power_met_sim[self.v_ind(u, j)] / (Nu * Nj)
                    temp[self.v_ind(i, v)] += u_j

                    i_v = self.power_met_sim[self.v_ind(i, v)] / (Ni * Nv)
                    temp[self.v_ind(u, j)] += i_v

                    calculations += 1

                    if calculations % cs.POWER_METHOD_REPORT_FREQ == 0:
                        prog = calculations / total
                        message = (('Iteration {} of power method, '
                                    '{:.2f}%').format(iteration_count,
                                                      (prog * 100)))
                        utils.print_log(message, mode='progress')

            # finish iteration
            temp = (alpha * temp) + ((1 - alpha) * self.blast_sim_n)
            diff = temp - self.power_met_sim
            error = sum(np.multiply(diff, diff))
            self.power_met_sim = temp

            message = (('Iteration {} of power method finished, '
                        'error: {}').format(iteration_count, error))
            utils.print_log(message, mode='end_progress')

            if ((error < cs.POWER_METHOD_ERROR_THR) and
                    (iteration_count > cs.MIN_POWER_METHOD_ITERS)):
                break

            if (iteration_count > cs.MAX_POWER_METHOD_ITERS):
                break

        message = (('power method ended after {} iterations,'
                    ' with total error: {}').format(iteration_count, error))
        utils.print_log(message)
