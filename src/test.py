import glob
from graph import ForensicGraph
import forensic_similarity as forsim  # forensic similarity tool
from src.utils.blockimage import tile_image  # function to tile image into blocks
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np
import random
from itertools import cycle


def plot_roc_curve(fpr, tpr, roc_auc, n_classes=2, lw=2):
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += metrics.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()



if __name__ == '__main__':
    root_dir = '../data/columbia/*/*.tif'
    spectral_preds = np.array([], dtype=int)
    modularity_preds = np.array([], dtype=int)
    y_true = np.array([], dtype=int)

    """ 0) Load images """
    test_imgs = list(glob.iglob(root_dir))[:100]
    random.shuffle(test_imgs)
    processed = 0
    for test_img in test_imgs:
        # Initiate a new forensic similarity graph
        fs_graph = ForensicGraph(patch_size=256, overlap=0.5, model_weights='../model/cam_256x256/-30')

        processed += 1
        print(f'Processed {processed} images of {len(test_imgs)}')
        print(f'Processing {test_img}...')
        auth = '4cam_auth' in test_img
        y_true = np.append(y_true, int(auth))
        img = plt.imread(test_img)

        # patches and xy coordinates of each patch for images 0, 1 and 2
        tiles, xy = tile_image(img, width=fs_graph.patch_size, height=fs_graph.patch_size, x_overlap=fs_graph.overlap,
                             y_overlap=fs_graph.overlap)


        """ 1) Sample N patches from the images """
        X0, ind0, X1, ind1 = fs_graph.dense_tiles(tiles, tiles)

        """ 2) Calculate forensic similarity between all pairs od sampled patches """
        sim = forsim.calculate_forensic_similarity(X0, X1, fs_graph.model_weights,
                                                       fs_graph.patch_size)  # between tiles from image 0 and image 2

        """ 3) Convert the image into its graph representation """
        graph = fs_graph.graph
        fs_graph.forensic_similarity_matrix(sim, ind0, ind1, threshold=0, same=True)

        """ 4) Perform forgery detection/localization """
        # 4 A) spectral clustering
        lambda2, u2 = fs_graph.spectral_clustering(10)
        print(f'Spectral Forged = {lambda2}')
        # print(f'Spectral clustering:\n {u2}')
        spectral_preds = np.append(spectral_preds, int(lambda2))
        #fs_graph.visualize_clusters(graph, u2, clean=True)

        # 4 B) modularity optimization
        q_opt, modularity_optim = fs_graph.modularity_optimization(2)
        print(f'Modularity Forged = {q_opt}')
        # print(f'Modularity optimization:\n {modularity_optim}')
        modularity_preds = np.append(modularity_preds, int(q_opt))
        #fs_graph.visualize_clusters(modularity_optim, modularity_optim.membership, clean=True)

        print(f'True class = {auth}')

    fpr_spectral, tpr_spectral, thresholds_spectral = metrics.roc_curve(y_true, spectral_preds, pos_label=1)
    auc_spectral = metrics.roc_auc_score(y_true, spectral_preds)
    print(f'Spectral\n fpr: {fpr_spectral}\n tpr: {tpr_spectral}\n tau: {thresholds_spectral}\n auc: {auc_spectral}')
    plot_roc_curve(fpr_spectral, tpr_spectral, auc_spectral)

    fpr_modularity, tpr_modularity, thresholds_modularity = metrics.roc_curve(y_true, modularity_preds, pos_label=1)
    auc_modularity = metrics.roc_auc_score(y_true, modularity_preds)
    print(f'Spectral\n fpr: {fpr_modularity}\n tpr: {tpr_modularity}\n tau: {thresholds_modularity}\n auc: {auc_modularity}')
    plot_roc_curve(fpr_modularity, tpr_modularity, auc_modularity)

