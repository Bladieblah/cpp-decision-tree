#ifndef NB_DECISION_TREE_H
#define NB_DECISION_TREE_H

#include <iostream>
#include <map>
#include <set>
#include <vector>

#include "csr_matrix.hpp"

enum ImpurityType {
    Gini,
    Entropy,
};

typedef struct Node {
    uint32_t leftChild = -1;
    uint32_t rightChild = -1;
    uint32_t feature = -2;
    double threshold = -2;
    double impurity;
    uint32_t nSamples;
    uint32_t label;

    void print() {
        std::cerr << "(" << leftChild << ", " << rightChild << ", " << feature << ", " << threshold << ", " << impurity << ", " << nSamples << ", " << label << ")" << std::endl;
    }
} Node;

typedef struct SplitResult {
    uint32_t feature;
    double score = HUGE_VALF, threshold = -1;
    double impurityLeft = -1, impurityRight = -1;
    size_t samplesLeft = -1, samplesRight = -1;
} SplitResult;

typedef struct Matrix {
    std::vector<uint32_t> data;
    size_t w, h;
} Matrix;

class DecisionTree {
public:
    DecisionTree(
        uint32_t criterion = ImpurityType::Gini, 
        uint32_t maxDepth = UINT32_MAX, 
        uint32_t minSamplesSplit = 2
    ) : _criterion(criterion), _maxDepth(maxDepth), _minSamplesSplit(minSamplesSplit) {};

    void fit(CSRMatrix &xTrain, std::vector<uint32_t> &yTrain, uint32_t nClasses);
    
    std::vector<uint32_t> predict(CSRMatrix &xPredict);
    std::vector<uint32_t> predictNodes(CSRMatrix &xPredict);
    Matrix predictProba(CSRMatrix &xPredict);
    std::vector< std::vector<uint32_t> > decisionPaths(CSRMatrix &xPredict);

    void printNodes();

    uint32_t getCriterion() {return _criterion;}
    uint32_t getMaxDepth() {return _maxDepth;}
    uint32_t getMinSamplesSplit() {return _minSamplesSplit;}
    uint32_t getNumClasses() {return _numClasses;}
    uint32_t getNumFeatures() {return _numFeatures;}

    std::vector<Node> getNodes() {return _nodes;}
    std::vector< std::vector<uint32_t> > getValues() {return _values;};

    void setCriterion(uint32_t criterion) {_criterion = criterion;}
    void setMaxDepth(uint32_t maxDepth) {_maxDepth = maxDepth;}
    void setMinSamplesSplit(uint32_t minSamplesSplit) {_minSamplesSplit = minSamplesSplit;}
    void setNumClasses(uint32_t numClasses) {_numClasses = numClasses;}
    void setNumFeatures(uint32_t numFeatures) {_numFeatures = numFeatures;}

    void setNodes(std::vector<Node> nodes) {_nodes = nodes;}
    void setValues(std::vector< std::vector<uint32_t> > values) {_values = values;}

private:
    void fitNode(CSRMatrix &xTrain, std::vector<uint32_t> &yTrain, std::vector<uint32_t> &sampleIndices, uint32_t nodeId, uint32_t depth);
    SplitResult checkSplit(CSRMatrix &xTrain, std::vector<uint32_t> &yTrain, std::vector<uint32_t> &sampleIndices, uint32_t feature);
    void createLabel(std::vector<uint32_t> &yTrain, std::vector<uint32_t> &sampleIndices);
    
    uint32_t predictSample(CSRMatrix &xPredict, uint32_t sample, uint32_t nodeId);
    void decisionPath(CSRMatrix &xPredict, uint32_t sample, uint32_t nodeId, std::vector<uint32_t> &path);
    
    double impurity(std::vector<uint32_t> labelCounts);
    double entropy(std::vector<uint32_t> labelCounts);
    double gini(std::vector<uint32_t> labelCounts);

    std::vector<Node> _nodes;
    std::vector< std::vector<uint32_t> > _values;

    uint32_t _criterion;
    uint32_t _maxDepth;
    uint32_t _minSamplesSplit;
    uint32_t _numClasses;
    uint32_t _numFeatures;
};

#endif
