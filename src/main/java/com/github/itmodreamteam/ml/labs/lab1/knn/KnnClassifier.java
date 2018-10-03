package com.github.itmodreamteam.ml.labs.lab1.knn;

import com.github.itmodreamteam.ml.utils.collections.IntList;
import com.github.itmodreamteam.ml.utils.matrixes.Matrix;
import com.github.itmodreamteam.ml.utils.matrixes.Vector;
import com.github.itmodreamteam.ml.classification.Classifier;

import java.util.Comparator;
import java.util.Map;
import java.util.stream.IntStream;

import static java.util.function.Function.identity;
import static java.util.stream.Collectors.counting;
import static java.util.stream.Collectors.groupingBy;

public class KnnClassifier implements Classifier {
    private final int numberOfNeighbors;
    private final KnnDistMeter meter;
    private final Matrix train;
    private final IntList classes;

    public KnnClassifier(int numberOfNeighbors, KnnDistMeter meter, Matrix train, IntList classes) {
        this.numberOfNeighbors = numberOfNeighbors;
        this.meter = meter;
        this.train = train;
        this.classes = classes;
    }

    @Override
    public int classify(Vector val) {
        return IntStream.range(0, train.rows())
                .mapToObj(i -> new Stat(i, meter.dist(train.row(i), val)))
                .sorted(Comparator.comparingDouble(Stat::getDist))
                .limit(numberOfNeighbors)
                .map(stat -> classes.get(stat.getIndex()))
                .collect(groupingBy(identity(), counting()))
                .entrySet()
                .stream()
                .max(Map.Entry.comparingByValue())
                .map(Map.Entry::getKey)
                .get();
    }

    private static class Stat {
        private final int index;
        private final double dist;

        Stat(int index, double dist) {
            this.index = index;
            this.dist = dist;
        }

        int getIndex() {
            return index;
        }

        double getDist() {
            return dist;
        }
    }
}
