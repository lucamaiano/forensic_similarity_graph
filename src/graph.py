import igraph
import forensic_similarity as forsim  # forensic similarity tool
from src.utils.blockimage import tile_image  # function to tile image into blocks
import matplotlib.pyplot as plt
import numpy as np


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
        self.patch_size = 256
        self.overlap = int(patch_size * overlap)
        self.model_weights = model_weights  # path to pretrained CNN weights

    def random_tiles(self, tiles, N=100):
        """ Randomly select N tiles. Returns the selected patches and their corresponding indices.
            Args:
                tiles (list): the list of i.e. patches.
                N (int): number of patches that you want to randomly select.
                """
        assert len(tiles.shape) == 4

        inds = np.random.randint(0, len(tiles), size=N)  # select random indices
        rand_tiles = tiles[inds]  # vector of randomly selected image tiles

        return rand_tiles, inds # Patches, indices

    def add_edges(self, forensic_similarity, ind0, ind1, threshold=None):
        """ An new edges to the graph.
            Args:
                forensic_similarity (list): forensic similarity values.
                ind0 (list): patch indices of the first image.
                ind0 (list): patch indices of the second image.
                threshold (float): similarity threshold. Edges with forensic_similarity < threshold are discarded. Default is None.
                """
        assert len(forensic_similarity) == len(ind0) and len(ind0) == len(ind1)
        assert threshold is None or threshold <= 1.0

        edges = []
        for indx in range(len(forensic_similarity)):
            if threshold is None or forensic_similarity[indx] >= threshold:
                edges.append((ind0[indx], ind1[indx]))
        print(edges)
        self.graph.add_edges(edges)


if __name__ == '__main__':
    # Initiate a new forensic similarity graph
    fs_graph = ForensicGraph(patch_size=256, overlap=0.5)

    """ 0) Load images """
    I0 = plt.imread('../data/0_google_pixel_1.jpg')
    I1 = plt.imread('../data/1_google_pixel_1.jpg')
    I2 = plt.imread('../data/2_asus_zenphone_laser.jpg')

    # patches and xy coordinates of each patch for images 0, 1 and 2
    T0, xy0 = tile_image(I0, width=fs_graph.patch_size, height=fs_graph.patch_size, x_overlap=fs_graph.overlap,
                         y_overlap=fs_graph.overlap)
    T1, xy1 = tile_image(I1, width=fs_graph.patch_size, height=fs_graph.patch_size, x_overlap=fs_graph.overlap,
                         y_overlap=fs_graph.overlap)
    T2, xy2 = tile_image(I2, width=fs_graph.patch_size, height=fs_graph.patch_size, x_overlap=fs_graph.overlap,
                         y_overlap=fs_graph.overlap)

    """ 1) Sample N patches from the images """
    X0, ind0 = fs_graph.random_tiles(T0, N=1000)
    X1, ind1 = fs_graph.random_tiles(T1, N=1000)
    X2, ind2 = fs_graph.random_tiles(T2, N=1000)

    """ 2) Calculate forensic similarity between all pairs od sampled patches """
    sim_0_1 = forsim.calculate_forensic_similarity(X0, X1, fs_graph.model_weights,
                                                   fs_graph.patch_size)  # between tiles from image 0 and image 1
    # sim_0_2 = forsim.calculate_forensic_similarity(X0, X2, fs_graph.model_weights,
    #                                                fs_graph.patch_size)  # between tiles from image 0 and image 2
    # sim_1_2 = forsim.calculate_forensic_similarity(X1, X2, fs_graph.model_weights,
    #                                                 fs_graph.patch_size)  # between tiles from image 1 and image 2

    """ 3) Convert the image into its graph representation """
    graph = fs_graph.graph
    graph.add_vertices(len(sim_0_1))
    fs_graph.add_edges(sim_0_1, ind0, ind1, filter=0.9)
    layout = graph.layout("kk")
    print(graph)


    """ 4) Perform forgery detection/localization """
