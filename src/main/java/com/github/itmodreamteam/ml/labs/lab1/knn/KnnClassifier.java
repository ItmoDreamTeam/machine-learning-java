package com.github.itmodreamteam.ml.labs.lab1.knn;

import com.github.itmodreamteam.ml.classification.Classifier;
import com.github.itmodreamteam.ml.utils.collections.IntList;
import com.github.itmodreamteam.ml.utils.matrixes.Matrix;
import com.github.itmodreamteam.ml.utils.matrixes.Vector;

import java.util.Comparator;
import java.util.List;
import java.util.stream.IntStream;

import static java.util.stream.Collectors.toList;

public class KnnClassifier implements Classifier {
    private final int numberOfNeighbors;
    private final KnnDistMeter meter;
    private final KnnImportanceFunction importanceFunction;
    private final Matrix train;
    private final IntList classes;
    private final int numberOfClassLabels;

    public KnnClassifier(int numberOfNeighbors, KnnDistMeter meter, KnnImportanceFunction importanceFunction, Matrix train, IntList classes, int numberOfClassLabels) {
        this.numberOfNeighbors = numberOfNeighbors;
        this.meter = meter;
        this.importanceFunction = importanceFunction;
        this.train = train;
        this.classes = classes;
        this.numberOfClassLabels = numberOfClassLabels;
    }

    @Override
    public int classify(Vector val) {
        List<Stat> nearestNeighbors = IntStream.range(0, train.rows())
                .mapToObj(i -> new Stat(meter.dist(train.row(i), val), classes.get(i)))
                .sorted(Comparator.comparingDouble(Stat::getDist))
                .limit(numberOfNeighbors)
                .collect(toList());
        double maxDist = nearestNeighbors.stream().mapToDouble(Stat::getDist).max().getAsDouble();
        for (int i = 0; i < numberOfNeighbors; i++) {
            Stat stat = nearestNeighbors.get(i);
            Stat normalizedStat = new Stat(stat.getDist() / maxDist, stat.getLabel());
            nearestNeighbors.set(i, normalizedStat);
        }
        double[] classWeight = new double[numberOfClassLabels];
        for (int serialNumberOfNeighbor = 1; serialNumberOfNeighbor <= nearestNeighbors.size(); ++serialNumberOfNeighbor) {
            double importance = importanceFunction.importance(nearestNeighbors.get(serialNumberOfNeighbor - 1).dist);
            classWeight[nearestNeighbors.get(serialNumberOfNeighbor - 1).label] += importance;
        }
        int maxClassWeightLabel = 0;
        for (int i = 0; i < classWeight.length; ++i) {
            if (classWeight[i] > classWeight[maxClassWeightLabel]) {
                maxClassWeightLabel = i;
            }
        }
        return maxClassWeightLabel;
    }

    private static class Stat {
        private final double dist;
        private final int label;

        Stat(double dist, int label) {
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
