package com.github.itmodreamteam.ml.labs.lab1.knn;

import com.github.itmodreamteam.ml.classification.Classifier;
import com.github.itmodreamteam.ml.utils.collections.IntList;
import com.github.itmodreamteam.ml.utils.matrixes.Matrix;
import com.github.itmodreamteam.ml.utils.matrixes.Vector;

import java.util.Comparator;
import java.util.List;
import java.util.stream.IntStream;

import static java.util.stream.Collectors.toList;

public class KnnClassifier implements Classifier<Vector, Integer> {
    private final int numberOfNeighbors;
    private final KnnClosestFunction meter;
    private final KnnImportanceFunction importanceFunction;
    private final Matrix train;
    private final IntList classes;
    private final int numberOfClasses;

    public KnnClassifier(int numberOfNeighbors, KnnClosestFunction meter, KnnImportanceFunction importanceFunction, Matrix train, IntList classes, int numberOfClasses) {
        this.numberOfNeighbors = numberOfNeighbors;
        this.meter = meter;
        this.importanceFunction = importanceFunction;
        this.train = train;
        this.classes = classes;
        this.numberOfClasses = numberOfClasses;
    }

    @Override
    public Integer classify(Vector val) {
        List<Neighbor> nearestNeighbors = IntStream.range(0, train.rows())
                .mapToObj(i -> new Neighbor(meter.dist(train.row(i), val), classes.get(i)))
                .sorted(Comparator.comparingDouble(Neighbor::getDist))
                .limit(numberOfNeighbors)
                .collect(toList());
        double maxDist = nearestNeighbors.stream().mapToDouble(Neighbor::getDist).max().getAsDouble();
        List<Neighbor> normalizedNearestNeighbors = nearestNeighbors.stream()
                .map(neighbor -> new Neighbor(neighbor.getDist() / maxDist, neighbor.getLabel()))
                .collect(toList());
        double[] classImportance = new double[numberOfClasses];
        for (Neighbor neighbor : normalizedNearestNeighbors) {
            double importance = importanceFunction.importance(neighbor.dist);
            classImportance[neighbor.label] += importance;
        }
        int maxClassWeightLabel = 0;
        for (int i = 0; i < classImportance.length; ++i) {
            if (classImportance[i] > classImportance[maxClassWeightLabel]) {
                maxClassWeightLabel = i;
            }
        }
        return maxClassWeightLabel;
    }

    private static class Neighbor {
        private final double dist;
        private final int label;

        Neighbor(double dist, int label) {
            this.dist = dist;
            this.label = label;
        }

        double getDist() {
            return dist;
        }

        int getLabel() {
            return label;
        }
    }
}
