import copy
import numpy as np

contained_clusters = [[1,2],[1,3],[5,1],[4,6], [4,7], [3,8]]

def create_cluster_heirarchy(contained_clusters):

    all_clusters = np.unique(np.array(contained_clusters).flatten())
    cluster_children = {cluster:[] for cluster in all_clusters}

    all_parents = [j[0] for j in contained_clusters]
    all_children = [j[1] for j in contained_clusters]
    for i in contained_clusters:
        for j in all_parents:
            if i[0] == j:
                cluster_children[j].append(all_children[all_parents.index(j)])
                all_children.remove(all_children[all_parents.index(j)])
                all_parents.remove(j)

    for cluster, children in cluster_children.items():
        all_children = [cluster_children[j] for j in all_clusters]
        for child in all_children:
            if cluster in child:
                parent = all_children.index(child)+1
                print(f'{cluster} is a child of {all_children.index(child)+1}')
                cluster_children[parent].extend(cluster_children[cluster])
                cluster_children[cluster] = []

    print(cluster_children)

