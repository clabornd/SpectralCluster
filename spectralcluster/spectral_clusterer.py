from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from copy import copy as cp
from sklearn.cluster import KMeans
from spectralcluster import refinement
from spectralcluster import utils


DEFAULT_REFINEMENT_SEQUENCE = [
    "CropDiagonal",
    "GaussianBlur",
    "RowWiseThreshold",
    "Symmetrize",
    "Diffuse",
    "RowWiseNormalize",
]


class SpectralClusterer(object):
    def __init__(
            self,
            min_clusters=None,
            max_clusters=None,
            gaussian_blur_sigma=1,
            p_percentile=0.95,
            thresholding_soft_multiplier=0.01,
            stop_eigenvalue=1e-2,
            refinement_sequence=DEFAULT_REFINEMENT_SEQUENCE):
        """Constructor of the clusterer.

        Args:
            min_clusters: minimal number of clusters allowed (only effective
                if not None)
            max_clusters: maximal number of clusters allowed (only effective
                if not None), can be used together with min_clusters to fix
                the number of clusters
            gaussian_blur_sigma: sigma value of the Gaussian blur operation
            p_percentile: the p-percentile for the row wise thresholding
            thresholding_soft_multiplier: the multiplier for soft threhsold,
                if this value is 0, then it's a hard thresholding
            stop_eigenvalue: when computing the number of clusters using
                Eigen Gap, we do not look at eigen values smaller than this
                value
            refinement_sequence: a list of strings for the sequence of
                refinement operations to apply on the affinity matrix
        """
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.gaussian_blur_sigma = gaussian_blur_sigma
        self.p_percentile = p_percentile
        self.thresholding_soft_multiplier = thresholding_soft_multiplier
        self.stop_eigenvalue = stop_eigenvalue
        self.refinement_sequence = refinement_sequence

    def _get_refinement_operator(self, name):
        """Get the refinement operator.

        Args:
            name: operator class name as a string

        Returns:
            object of the operator

        Raises:
            ValueError: if name is an unknown refinement operation
        """
        if name == "CropDiagonal":
            return refinement.CropDiagonal()
        elif name == "GaussianBlur":
            return refinement.GaussianBlur(self.gaussian_blur_sigma)
        elif name == "RowWiseThreshold":
            return refinement.RowWiseThreshold(
                self.p_percentile,
                self.thresholding_soft_multiplier)
        elif name == "Symmetrize":
            return refinement.Symmetrize()
        elif name == "Diffuse":
            return refinement.Diffuse()
        elif name == "RowWiseNormalize":
            return refinement.RowWiseNormalize()
        else:
            raise ValueError("Unknown refinement operation: {}".format(name))
    
    def get_eigen_inputs(self, X):
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy array")
        if len(X.shape) != 2:
            raise ValueError("X must be 2-dimensional")
        #  Compute affinity matrix.
        affinity = utils.compute_affinity_matrix(X)

        # Refinement opertions on the affinity matrix.
        for refinement_name in self.refinement_sequence:
            op = self._get_refinement_operator(refinement_name)
            affinity = op.refine(affinity)
            
        # Perform eigen decomposion.
        (eigenvalues, eigenvectors) = utils.compute_sorted_eigenvectors(
            affinity)
        # Get number of clusters.
        k = utils.compute_number_of_clusters(
            eigenvalues, self.max_clusters, self.stop_eigenvalue)
        if self.min_clusters is not None:
            k = max(k, self.min_clusters)

        return k, affinity, eigenvectors, eigenvalues
        
    def predict(self, X, row_norm = False):
        """Perform spectral clustering on data X.

        Args:
            X: numpy array of shape (n_samples, n_features)

        Returns:
            labels: numpy array of shape (n_samples,)

        Raises:
            TypeError: if X has wrong type
            ValueError: if X has wrong shape
        """
        
        k, affinity, eigenvectors, _ = self.get_eigen_inputs(X)
        

        # Get spectral embeddings.
        spectral_embeddings = eigenvectors[:, :k]
        
        if row_norm:
            length = np.sqrt((spectral_embeddings**2).sum(axis=1))[:,None]
            spectral_embeddings = spectral_embeddings / length
        
        # Run K-Means++ on spectral embeddings.
        # Note: The correct way should be using a K-Means implementation
        # that supports customized distance measure such as cosine distance.
        # This implemention from scikit-learn does NOT, which is inconsistent
        # with the paper.
        kmeans_clusterer = KMeans(
            n_clusters=k,
            init="k-means++",
            max_iter=300,
            random_state=0)
        labels = kmeans_clusterer.fit_predict(spectral_embeddings)
        return labels
    
    def predict_recursive(self, X, row_norm = False, unpack = False):
        """Perform recursive spectral clustering on data X.  Specifically, we run the clustering on X, and if there are more 
        than 1 labels, we re-run the clustering on the subsets of the data corresponding to those labels until no subset 
        returns more than one label.

        Args:
            X: numpy array of shape (n_samples, n_features)

        Returns:
            results:  A dictionary of the recursive clustering if unpack = False, or an array of speaker ids of shape
            (n_samples,)

        Raises:
            TypeError: if X has wrong type
            ValueError: if X has wrong shape
        """
        results = {}

        labels = self.predict(X, row_norm = row_norm)
        
        # remove singleton labels
        labels = np.array([el if len([z for z in labels if z == el]) > 1 else 's' for el in labels])
        
        results['labels'] = labels

        if len(np.unique(labels)) > 1:
            results['sublabels'] = {}
            for lab in np.unique(labels):
                inds = np.where(labels == lab)[0]
                if lab == 's':
                    results['sublabels'][lab] = {'labels':labels[inds]}
                else:
                    results['sublabels'][lab] = self.predict_recursive(X[inds], row_norm, unpack = False)
            
        if unpack:
            results = unpack_labels(results)
        
        return results
    
def unpack_labels(cluster_tree):
    """
    Flatten the results of predict_recursive() into a 1-d result
    
    Args:
        cluster_tree: dictionary result of calling predict_recursive with unpack = False

    Returns:
        labels: numpy array of shape (n_samples,)
    """
    labels = cluster_tree['labels']
    out_labels = cp(labels)
    out_labels = out_labels.astype("str")

    if('sublabels' in cluster_tree.keys()):
        for lab in cluster_tree['sublabels'].keys(): 
            newlabs = unpack_labels(cluster_tree['sublabels'][lab]).astype("str")
            prependlabs = np.array([str(lab)+"-"]*len(newlabs))
            newlabs = np.char.add(prependlabs, newlabs)

            inds = np.where(labels == lab)
            out_labels[inds] = newlabs
    
    return out_labels
        
        
