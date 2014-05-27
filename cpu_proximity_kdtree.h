#ifndef __CPU_PROXIMITY_KDTREE_H_
#define __CPU_PROXIMITY_KDTREE_H_

#include <vector>
#include <iostream>
#include <algorithm>

template <size_t N, typename T>
struct Node
{
	T coords[N];
	int id;
};

template<size_t N, typename T>
struct KdTreeNode
{
	KdTreeNode* left;
	KdTreeNode* right;
	Node<N, T> p;
};

struct KNNResultInfo
{
	int id;
	double dist;
};

template <size_t N, typename T>
struct Node_compare
{
	bool operator()(Node<N, T> const& lhs, Node<N, T> const& rhs)
	{
		return lhs.coords[COMPARE_COORD_IDX] < rhs.coords[COMPARE_COORD_IDX];
	}
	
	void set_compared_coord(size_t D)
	{
		COMPARE_COORD_IDX = D;
	}
	
	size_t COMPARE_COORD_IDX;
};

struct KNNResult_compare
{
	bool operator()(KNNResultInfo const& lhs, KNNResultInfo const& rhs)
	{
		return lhs.dist < rhs.dist;
	}
};

template <size_t N, typename T>
double distance_sqr2(Node<N, T> const& lhs, Node<N, T> const& rhs)
{
	double dist = 0;
	for(int i = 0; i < N; ++i)
	{
		dist += (lhs.coords[i] - rhs.coords[i]) * (lhs.coords[i] - rhs.coords[i]);
	}
	
	return dist;
}

template <size_t N, typename T>
void knn_check(std::vector<KNNResultInfo>& knn_result, KdTreeNode<N, T>* node, Node<N, T> const& query, int K, T mindist)
{
	double d = distance_sqr2(node->p, query);
	if(d <= mindist) return;

	KNNResultInfo r;
	r.id = node->p.id;
	r.dist = d;
	if(knn_result.size() < K)
		knn_result.push_back(r);
	else if(r.dist < knn_result[knn_result.size() - 1].dist)
		knn_result[knn_result.size() - 1] = r;
	sort(knn_result.begin(), knn_result.end(), KNNResult_compare());
}

template <size_t N, typename T>
void prepare_nodes(T* samples, unsigned int nSamples, std::vector<Node<N, T> >& nodes)
{
	for(int i = 0; i < nSamples; ++i)
	{
		Node<N, T> p;
		for(int j = 0; j < N; ++j)
			p.coords[j] = samples[i * N + j];
		p.id = i;
		nodes.push_back(p);
	}
}

template <size_t N, typename T>
KdTreeNode<N, T>* build_kdtree(std::vector<Node<N, T> > const& nodes, int depth)
{
	if(nodes.size() == 0) return NULL;
	
	int axis = depth % N;
	
	std::vector<Node<N, T> > sorted_nodes(nodes.size());
	copy(nodes.begin(), nodes.end(), sorted_nodes.begin());
	
	Node_compare<N, T> sort_compared_functor;
	sort_compared_functor.set_compared_coord(axis);
	std::sort(sorted_nodes.begin(), sorted_nodes.end(), sort_compared_functor);
	
	int median = (nodes.size() - 1) / 2;
	
	std::vector<Node<N, T> > left_nodes, right_nodes;
	if(median >= 0)
	{
		left_nodes.resize(median);
		copy(sorted_nodes.begin(), sorted_nodes.begin() + median, left_nodes.begin());
	}
	if(median + 1 < nodes.size())
	{
		right_nodes.resize(nodes.size() - median - 1);
		copy(sorted_nodes.begin() + median + 1, sorted_nodes.end(), right_nodes.begin());
	}
	
	KdTreeNode<N, T>* kdnode = new KdTreeNode<N, T>();
	kdnode->p = *(sorted_nodes.begin() + median);
	kdnode->left = build_kdtree(left_nodes, depth + 1);
	kdnode->right = build_kdtree(right_nodes, depth + 1);
	
	return kdnode;
}

template <size_t N, typename T>
void traverse_kdtree(KdTreeNode<N, T>* root)
{
	std::vector<KdTreeNode<N, T>* > v1, v2;
	v1.push_back(root);
	while(v1.size() > 0)
	{
		for(int i = 0; i < v1.size(); ++i)
		{
			std::cout << v1[i]->p.id << " ";
			for(int j = 0; j < N; ++j)
			{
				std::cout << v1[i]->p.coord[j] << " ";
			}
			std::cout << std::endl;
			if(v1[i]->left) v2.push_back(v1[i]->left);
			if(v1[i]->right) v2.push_back(v2[i]->right);
		}
		v1.clear();
		for(int i = 0; i < v2.size(); ++i)
			v1.push_back(v2[i]);
		v2.clear();
	}
}

template <size_t N, typename T>
void knn(std::vector<KNNResultInfo>& knn_result, KdTreeNode<N, T>* node, Node<N, T>& query, int depth, int K, T mindist)
{
	int axis = depth % N;
	if(node->left == NULL && node->right == NULL)
	{
		knn_check(knn_result, node, query, K, mindist);
		return;
	}
	
	KdTreeNode<N, T>* near_node = NULL;
	KdTreeNode<N, T>* far_node = NULL;
	
	if(node->right == NULL || (node->left && query.coords[axis] <= node->p.coords[axis]))
	{
		near_node = node->left;
		far_node = node->right;
	}
	else
	{
		near_node = node->right;
		far_node = node->left;
	}
	
	knn(knn_result, near_node, query, depth + 1, K, mindist);
	
	if(far_node)
	{
		if(knn_result.size() < K || (query.coords[axis] - node->p.coords[axis]) *(query.coords[axis] - node->p.coords[axis]) < knn_result[knn_result.size() - 1].dist)
		{
			knn(knn_result, far_node, query, depth + 1, K, mindist);
		}
	}
	
	knn_check(knn_result, node, query, K, mindist);
}

template <size_t N, typename T>
void proximityComputation_kdtree(T* samples, unsigned int nSamples, T* queries, unsigned int nQueries, unsigned int K, T mindist, unsigned int* KNNResult)
{
	std::vector<Node<N, T> > nodes;
	prepare_nodes<N, T>(samples, nSamples, nodes);
	KdTreeNode<N, T>* root = NULL;
	root = build_kdtree(nodes, 0);

	if(samples == queries && nQueries <= nSamples)
	{
		std::vector<KNNResultInfo> results;
		for(int i = 0; i < nQueries; ++i)
		{
			results.clear();
			knn<N, T>(results, root, nodes[i], 0, K, mindist);

			for(int j = 0; j < K; ++j)
			{
				KNNResult[j * nQueries + i] = results[j].id;
			}
		}
	}
	else
	{
		std::vector<Node<N, T> > query_nodes;
		prepare_nodes<N, T>(queries, nQueries, query_nodes);

		std::vector<KNNResultInfo> results;
		for(int i = 0; i < nQueries; ++i)
		{
			results.clear();
			knn<N, T>(results, root, query_nodes[i], 0, K, mindist);

			for(int j = 0; j < K; ++j)
			{
				KNNResult[j * nQueries + i] = results[j].id;
			}
		}
	}
}


#endif
