import igraph
import numpy as np
import numpy.linalg
import os

class ForensicGraph():
    def __init__(self, patch_size=256, overlap=0.5, model_weights='../model/cam_256x256/-30'):
        """ Initiate a new Forensic Similarity Graph.
            Args:
                patch_size (int): the size of the patches.
                overlap (float): percentage of overlap. It must be a number between 0 and 1.
                model_weights (string): the path of the pre-trained model.
                """
        assert patch_size in [256, 128] and overlap < 1.0

        self.graph = igraph.Graph()
        self.patch_size = patch_size
        self.overlap = int(patch_size * overlap)
        self.model_weights = model_weights  # path to pretrained CNN weights
        self.similarity_matrix = None
        self.num_vertices = None

    def random_tiles(self, tiles, N=100):
        """ Randomly select N tiles. Returns the selected patches and their corresponding indices.
            Args:
                tiles (list): the list of tiles i.e. patches.
                N (int): number of patches that you want to randomly select.
                """
        assert len(tiles.shape) == 4

        inds = np.random.randint(0, len(tiles), size=N)  # select random indices
        rand_tiles = tiles[inds]  # vector of randomly selected image tiles

        return rand_tiles, inds  # Patches, indices

    def dense_tiles(self, t0, t1):
        """ Generate (n^2-n)/2 pairs of tiles.
            Args:
                t0 (list): the list of tiles i.e. patches.
                t1 (list): the list of i.e. patches.
                """
        assert len(t0.shape) == 4 and len(t1.shape) == 4

        self.num_vertices = np.max([len(t0), len(t1)])

        inds0 = np.array([])
        inds1 = np.array([])
        for idx1 in range(len(t1)):
            inds0 = np.append(inds0, [idx0 for idx0 in range(len(t0))]).astype(int)
            inds1 = np.append(inds1, np.ones(len(t0), int)*idx1).astype(int)

        tiles0 = t0[inds0]
        tiles1 = t1[inds1]

        return tiles0, inds0, tiles1, inds1  # Patches, indices

    def forensic_similarity_matrix(self, forensic_similarity, ind0, ind1, threshold=0, same=True):
        """ Construct the forensic similarity matrix and add new edges to the graph.
            Args:
                forensic_similarity (list): forensic similarity values.
                ind0 (list): patch indices of the first image.
                ind0 (list): patch indices of the second image.
                threshold (float): similarity threshold. Edges with forensic_similarity < threshold are discarded. Default is None.
                same (bool): indicate whether the input patches come from the same image. Default is True.
                """
        assert len(forensic_similarity) == len(ind0) and len(ind0) == len(ind1)
        assert threshold is 0 and threshold <= 1.0
        self.graph.add_vertices(self.num_vertices)

        W_matrix = np.zeros((np.max(ind0), np.max(ind1)))
        edges = []
        weights = []
        vertices = set()
        for indx in range(len(forensic_similarity)):
            indx0 = ind0[indx] - 1
            indx1 = ind1[indx] - 1

            # Add vertices to the list
            if ind0[indx] not in vertices:
                vertices.add(ind0[indx])
            if ind1[indx] not in vertices:
                vertices.add(ind1[indx])

            # Skip edges betweeen single nodes, i.e. (vi, vi)
            if same and ind0[indx] == ind1[indx]:
                continue

            # Add a new edge if not in the graph
            if (ind0[indx], ind1[indx]) not in edges and (ind1[indx], ind0[indx]) not in edges:
                edges.append((ind0[indx], ind1[indx]))
            else:
                continue

            # Add weights
            if forensic_similarity[indx] >= threshold:
                W_matrix[indx0][indx1] = forensic_similarity[indx]
                weights.append(forensic_similarity[indx])
            else:
                weights.append(0)

        self.similarity_matrix = W_matrix
        self.graph.add_edges(edges)
        self.graph.es['weight'] = weights
        self.graph.vs['patch'] = sorted(list(vertices))

    def spectral_clustering(self, threshold=10):
        """ Calculate spectral clustering.
            Args:
                threshold (int/float): the threshold used to decide whether a patch is authentic or manipulated. Default is 10.
                """
        laplacian_matrix = np.array(self.graph.laplacian(self.graph.es['weight'], False))
        eigvals, eigvecs = numpy.linalg.eig(laplacian_matrix)
        print(f"lambda_2 = {float(eigvals[1])}")

        membership = [0 if vec >= 0 else 1 for vec in eigvecs[1]]

        return eigvals[1] < threshold, membership


    def modularity_optimization(self, threshold=2):
        """ Construct the forensic similarity matrix and add new edges to the graph.
            Args:
                threshold (int/float): the threshold used to decide whether a patch is authentic or manipulated. Default is 10.
            """
        modularity_optim = self.graph.community_fastgreedy(self.graph.es['weight'])
        print(f"Q_opt = {modularity_optim.optimal_count}")

        return modularity_optim.optimal_count >= threshold, modularity_optim.as_clustering()

    def visualize_clusters(self, cluster, membership, out_file='../output/cluster.png', clean=True):
        """ Produce a visualization of the graph.
            Args:
                cluster (VertexClustering): an input cluster that you want to visualize.
                membership (list): a list of memberships.
                clean (bool): set True if you want to filter out irrelevant/low weights. Default is True.
            """
        out_path = out_file.split(out_file.split('/')[-1])[0]
        if not os.path.isdir(out_path):
            os.makedirs(out_path)
        visual_style = {}

        # Define vertex colors
        color_dict = {0: "green", 1: "red", 2: "grey"}
        visual_style["vertex_color"] = [color_dict[k] for k in membership]
        # Set bbox and margin
        # visual_style["bbox"] = (3000, 3000)
        # visual_style["margin"] = 20
        # Set vertex size
        visual_style["vertex_size"] = 22
        # Set vertex lable size
        visual_style["vertex_label_size"] = 14
        if clean:
            # Set edges width
            visual_style["edge_width"] = [2 * int(weight > 0.7) for weight in self.graph.es["weight"]]
        # Set the layout
        visual_style["layout"] = self.graph.layout_kamada_kawai()

        self.graph.vs['label'] = self.graph.vs['patch']
        igraph.plot(cluster, out_file, **visual_style)

