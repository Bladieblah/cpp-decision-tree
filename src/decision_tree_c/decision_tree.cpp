#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <math.h>
#include <map>
#include <numeric>
#include <set>
#include <vector>

#include "csr_matrix.hpp"
#include "decision_tree.hpp"

using namespace std;

void DecisionTree::fit(CSRMatrix &xTrain, vector<uint32_t> &yTrain, uint32_t nClasses) {
    _numClasses = nClasses;
    _numFeatures = xTrain.numColumns;
    
    vector<uint32_t> labelCounts(_numClasses);
    for (size_t i = 0; i < yTrain.size(); i++) {
        labelCounts[yTrain[i]]++;
    }

    Node root;
    root.nSamples = yTrain.size();
    root.impurity = impurity(labelCounts);
    _nodes.push_back(root);

    vector<uint32_t> sampleIndices(yTrain.size());
    iota(sampleIndices.begin(), sampleIndices.end(), 0);
    
    createLabel(yTrain, sampleIndices);
    fitNode(xTrain, yTrain, sampleIndices, 0, 0);
}

void DecisionTree::fitNode(CSRMatrix &xTrain, vector<uint32_t> &yTrain, vector<uint32_t> &sampleIndices, uint32_t nodeId, uint32_t depth) {
    SplitResult result, bestResult;

    if (sampleIndices.size() < _minSamplesSplit || depth >= _maxDepth) {
        return;
    }

    // Look for the best split
    for (size_t feature = 0; feature < _numFeatures; feature++) {
        result = checkSplit(xTrain, yTrain, sampleIndices, feature);
        if (result.score  < bestResult.score) {
            bestResult = result;
        }
    }

    // Check if a split was found
    if (bestResult.score == HUGE_VALF) {
        return;
    }

    // Split up the samples
    vector<uint32_t> samplesLeft, samplesRight;
    for (size_t i = 0; i < sampleIndices.size(); i++) {
        if (xTrain.get(sampleIndices[i], bestResult.feature) < bestResult.threshold) {
            samplesLeft.push_back(sampleIndices[i]);
        } else {
            samplesRight.push_back(sampleIndices[i]);
        }
    }

    // Set the feature and threshold
    _nodes[nodeId].feature = bestResult.feature;
    _nodes[nodeId].threshold = bestResult.threshold;
    
    // Create and fit the left child
    _nodes[nodeId].leftChild = _nodes.size();
    Node left;
    left.nSamples = bestResult.samplesLeft;
    left.impurity = bestResult.impurityLeft;
    _nodes.push_back(left);
    createLabel(yTrain, samplesLeft);
    fitNode(xTrain, yTrain, samplesLeft, _nodes[nodeId].leftChild, depth + 1);
    
    // Create and fit the right child
    _nodes[nodeId].rightChild = _nodes.size();
    Node right;
    right.nSamples = bestResult.samplesRight;
    right.impurity = bestResult.impurityRight;
    _nodes.push_back(right);
    createLabel(yTrain, samplesRight);
    fitNode(xTrain, yTrain, samplesRight, _nodes[nodeId].rightChild, depth + 1);
}

void DecisionTree::createLabel(std::vector<uint32_t> &yTrain, std::vector<uint32_t> &sampleIndices) {
    vector<uint32_t> labelCounts(_numClasses);
    for (size_t i = 0; i < sampleIndices.size(); i++) {
        labelCounts[yTrain[sampleIndices[i]]]++;
    }

    // Set label to most common class, lowest index when there is a conflict
    _nodes[_values.size()].label = distance(labelCounts.begin(), max_element(labelCounts.begin(), labelCounts.end()));
    _values.push_back(labelCounts);
}

SplitResult DecisionTree::checkSplit(CSRMatrix &xTrain, vector<uint32_t> &yTrain, vector<uint32_t> &sampleIndices, uint32_t feature) {
    SplitResult result;
    result.feature = feature;

    auto comparator = [&xTrain, feature](size_t a, size_t b){ return xTrain.get(a, feature) < xTrain.get(b, feature); };
    sort(sampleIndices.begin(), sampleIndices.end(), comparator);
    
    size_t nSamples = sampleIndices.size();
    vector<uint32_t> leftCounts(_numClasses);
    vector<uint32_t> rightCounts(_numClasses);

    for (size_t i = 0; i < sampleIndices.size(); i++) {
        rightCounts[yTrain[sampleIndices[i]]]++;
    }

    size_t i = 0, iOpt;
    double score, impurityLeft, impurityRight;

    while (i < nSamples - 1) {
        leftCounts[yTrain[sampleIndices[i]]]++;
        rightCounts[yTrain[sampleIndices[i]]]--;

        if (xTrain.get(sampleIndices[i], feature) != xTrain.get(sampleIndices[i + 1], feature)) {
            impurityLeft = impurity(leftCounts);
            impurityRight = impurity(rightCounts);
            score = ((i + 1) * impurityLeft + (nSamples - i - 1) * impurityRight) / nSamples;

            if (score < result.score) {
                iOpt = i;

                result.score = score;
                result.impurityLeft = impurityLeft;
                result.impurityRight = impurityRight;
                result.samplesLeft = i + 1;
                result.samplesRight = nSamples - i - 1;
            }
        }

        i++;
    }

    if (result.score < HUGE_VALF) {
        result.threshold = (xTrain.get(sampleIndices[iOpt], feature) + xTrain.get(sampleIndices[iOpt + 1], feature)) / 2;
    }

    return result;
}

double DecisionTree::impurity(std::vector<uint32_t> labelCounts) {
    switch (_criterion) {
        case ImpurityType::Entropy:
            return entropy(labelCounts);
        case ImpurityType::Gini:
            return gini(labelCounts);
    }

    return gini(labelCounts);
}

double DecisionTree::entropy(std::vector<uint32_t> labelCounts) {
    double normalisation = reduce(labelCounts.begin(), labelCounts.end());
    
    double result = 0;
    for (uint32_t count : labelCounts) {
        if (count == 0) {
            continue;
        }

        double p = (double)count / normalisation;
        result -= p * log(p);
    }

    return result;
}

double DecisionTree::gini(std::vector<uint32_t> labelCounts) {
    double normalisation = reduce(labelCounts.begin(), labelCounts.end());
    
    double result = 0;
    for (uint32_t count : labelCounts) {
        if (count == 0) {
            continue;
        }

        double p = (double)count / normalisation;
        result += pow(p, 2);
    }

    return 1 - result;
}

vector<uint32_t> DecisionTree::predict(CSRMatrix &xPredict) {
    size_t nSamples = xPredict.rowCount();
    vector<uint32_t> predictions(nSamples);

    for (size_t i = 0; i < nSamples; i++) {
        uint32_t id = predictSample(xPredict, i, 0);
        predictions[i] = _nodes[id].label;
    }

    return predictions;
}

vector<uint32_t> DecisionTree::predictNodes(CSRMatrix &xPredict) {
    size_t nSamples = xPredict.rowCount();
    vector<uint32_t> predictions(nSamples);

    for (size_t i = 0; i < nSamples; i++) {
        predictions[i] = predictSample(xPredict, i, 0);
    }

    return predictions;
}

Matrix DecisionTree::predictProba(CSRMatrix &xPredict) {
    size_t nSamples = xPredict.rowCount();
    Matrix predictions;
    predictions.data.reserve(nSamples * _numClasses);
    
    predictions.w = getNumClasses();
    predictions.h = xPredict.rowCount();

    for (size_t i = 0; i < nSamples; i++) {
        uint32_t nodeId = predictSample(xPredict, i, 0);
        predictions.data.insert(predictions.data.end(), _values[nodeId].begin(), _values[nodeId].end());
    }

    return predictions;
}

uint32_t DecisionTree::predictSample(CSRMatrix &xTrain, uint32_t sample, uint32_t nodeId) {
    Node current = _nodes[nodeId];
    if (current.feature == -2) {
        return nodeId;
    }

    if (xTrain.get(sample, current.feature) < current.threshold) {
        return predictSample(xTrain, sample, current.leftChild);
    }

    return predictSample(xTrain, sample, current.rightChild);
}

vector< vector<uint32_t> > DecisionTree::decisionPaths(CSRMatrix &xPredict) {
    size_t nSamples = xPredict.rowCount();
    vector< vector<uint32_t> > paths;

    for (size_t i = 0; i < nSamples; i++) {
        vector<uint32_t> path;
        decisionPath(xPredict, i, 0, path);
        paths.push_back(path);
    }

    return paths;
}

void DecisionTree::decisionPath(CSRMatrix &xPredict, uint32_t sample, uint32_t nodeId, vector<uint32_t> &path) {
    path.push_back(nodeId);
    Node current = _nodes[nodeId];
    if (current.feature == -2) {
        return;
    }

    if (xPredict.get(sample, current.feature) < current.threshold) {
        return decisionPath(xPredict, sample, current.leftChild, path);
    }

    return decisionPath(xPredict, sample, current.rightChild, path);
}

void DecisionTree::printNodes() {
    for (size_t i = 0; i < _nodes.size(); i++) {
        cerr << "Node " << i << ": ";
        _nodes[i].print();
    }
}

int main() {
    return 0;
}
