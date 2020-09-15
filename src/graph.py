import networkx as nx
import igraph
import forensic_similarity as forsim  # forensic similarity tool
from src.utils.blockimage import tile_image  # function to tile image into blocks
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg


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
        for indx in range(len(forensic_similarity)):
            indx0 = ind0[indx] - 1
            indx1 = ind1[indx] - 1

            if same and ind0[indx] == ind1[indx]:
                continue
            if (ind0[indx], ind1[indx]) not in edges and (ind1[indx], ind0[indx]) not in edges:
                edges.append((ind0[indx], ind1[indx]))
            else:
                continue

            if forensic_similarity[indx] >= threshold:
                W_matrix[indx0][indx1] = forensic_similarity[indx]
                weights.append(forensic_similarity[indx])
            else:
                weights.append(0)

        self.similarity_matrix = W_matrix
        self.graph.add_edges(edges)
        self.graph.es['weight'] = weights

    def spectral_clustering(self, threshold=10):
        laplacian_matrix = np.array(self.graph.laplacian(self.graph.es['weight'], False))
        eigvals, eigvecs = numpy.linalg.eig(laplacian_matrix)
        print(f"lambda_2 = {float(eigvals[1])}")


        return eigvals[1] < threshold, [0 if vec >= 0 else 1 for vec in eigvecs[1]]


    def modularity_optimization(self, threshold=2):
        modularity_optim = self.graph.community_fastgreedy(self.graph.es['weight'])
        print(f"Q_opt = {modularity_optim.optimal_count}")

        return modularity_optim.optimal_count >= threshold, modularity_optim

    def visualize_graph(self, membership):
        # out_fig_name = "../output/graph.png"
        # visual_style = {}
        # # Define colors used for outdegree visualization
        # colours = ['#fecc5c', '#a31a1c']
        # # Set bbox and margin
        # visual_style["bbox"] = (3000, 3000)
        # visual_style["margin"] = 17
        # # Set vertex colours
        # visual_style["vertex_color"] = 'grey'
        # # Set vertex size
        # visual_style["vertex_size"] = 20
        # # Set vertex lable size
        # visual_style["vertex_label_size"] = 8
        # # Don't curve the edges
        # visual_style["edge_curved"] = False
        # # Set the layout
        # my_layout = self.graph.layout_kamada_kawai()
        # visual_style["layout"] = my_layout
        # # Plot the graph
        # igraph.plot(self.graph, out_fig_name)
        coords = self.graph.layout_kamada_kawai()
        igraph.plot(self.graph, color=membership, layout=coords)


if __name__ == '__main__':
    # Initiate a new forensic similarity graph
    fs_graph = ForensicGraph(patch_size=256, overlap=0.5, model_weights='../model/cam_256x256/-30')

    """ 0) Load images """
    I0 = plt.imread('../data/columbia/4cam_auth/canong3_02_sub_01.tif')
    I1 = plt.imread('../data/columbia/4cam_auth/canong3_02_sub_02.tif')
    I2 = plt.imread('../data/columbia/4cam_splc/canong3_canonxt_sub_01.tif')
    I3 = plt.imread('../data/columbia/4cam_splc/nikond70_kodakdcs330_sub_19.tif')
    # image0 and image1 are from the same camera model, and have high forensic similarity
    # image0 and image2 are from different camera models, and have low forensic similarity
    # image1 and image2 are from different camera models, and have low forensic similarity

    # patches and xy coordinates of each patch for images 0, 1 and 2
    T0, xy0 = tile_image(I0, width=fs_graph.patch_size, height=fs_graph.patch_size, x_overlap=fs_graph.overlap,
                         y_overlap=fs_graph.overlap)
    T1, xy1 = tile_image(I1, width=fs_graph.patch_size, height=fs_graph.patch_size, x_overlap=fs_graph.overlap,
                         y_overlap=fs_graph.overlap)
    T2, xy2 = tile_image(I2, width=fs_graph.patch_size, height=fs_graph.patch_size, x_overlap=fs_graph.overlap,
                         y_overlap=fs_graph.overlap)
    T3, xy3 = tile_image(I3, width=fs_graph.patch_size, height=fs_graph.patch_size, x_overlap=fs_graph.overlap,
                         y_overlap=fs_graph.overlap)

    """ 1) Sample N patches from the images """
    # X0, ind0 = fs_graph.random_tiles(T0, N=5000)
    # X1, ind1 = fs_graph.random_tiles(T1, N=5000)
    # X2, ind2 = fs_graph.random_tiles(T2, N=5000)

    X0, ind0, X2, ind2 = fs_graph.dense_tiles(T2, T2)
    # X0, ind0, X1, ind1 = fs_graph.dense_tiles(T0, T1)
    # X1, ind1, X2, ind2 = fs_graph.dense_tiles(T1, T2)

    """ 2) Calculate forensic similarity between all pairs od sampled patches """
    # sim_0_1 = forsim.calculate_forensic_similarity(X0, X1, fs_graph.model_weights,
    #                                                fs_graph.patch_size)  # between tiles from image 0 and image 1
    sim_0_2 = forsim.calculate_forensic_similarity(X0, X2, fs_graph.model_weights,
                                                   fs_graph.patch_size)  # between tiles from image 0 and image 2
    # sim_1_2 = forsim.calculate_forensic_similarity(X1, X2, fs_graph.model_weights,
    #                                                 fs_graph.patch_size)  # between tiles from image 1 and image 2

    """ 3) Convert the image into its graph representation """
    graph = fs_graph.graph
    # fs_graph.forensic_similarity_matrix(sim_0_1, ind0, ind1, threshold=0)
    fs_graph.forensic_similarity_matrix(sim_0_2, ind0, ind2, threshold=0)
    # fs_graph.forensic_similarity_matrix(sim_1_2, ind1, ind2, threshold=0)

    """ 4) Perform forgery detection/localization """
    # 4 A) spectral clustering
    lambda2, u2 = fs_graph.spectral_clustering(10)
    print(f'Forged = {lambda2}')
    # fs_graph.visualize_graph(u2)

    # 4 B) modularity optimization
    q_opt, modularity_optim = fs_graph.modularity_optimization(2)
    print(f'Forged = {q_opt}')
    # fs_graph.visualize_graph(modularity_optim)
