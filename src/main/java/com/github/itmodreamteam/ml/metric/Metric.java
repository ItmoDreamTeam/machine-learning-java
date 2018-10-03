package com.github.itmodreamteam.ml.metric;

import java.util.Arrays;
import java.util.List;

public class Metric {
    private final Stats[] stats;

    public static Builder builder(int numberOfClasses) {
        return new Builder(numberOfClasses);
    }

    private Metric(Builder builder) {
        this.stats = new Stats[builder.numberOfClasses];
        for (int classLabel = 0; classLabel < builder.numberOfClasses; ++classLabel) {
            int truePositive = builder.stat[classLabel][classLabel];
            // truePositive + falsePositive
            int totalPredicted = Arrays.stream(builder.stat[classLabel]).sum();
            // truePositive + falseNegative
            int totalLabeled = Arrays.stream(column(builder.stat, builder.numberOfClasses, classLabel)).sum();
            double precision = 1.0 * truePositive / totalPredicted;
            double recall = 1.0 * truePositive / totalLabeled;
            Stats stats = new Stats(precision, recall);
            this.stats[classLabel] = stats;
        }
    }

    private static int[] column(int[][] arr, int size, int columnNumber) {
        int[] col = new int[size];
        for (int i = 0; i < size; ++i) {
            col[i] = arr[i][columnNumber];
        }
        return col;
    }

    private Metric(Stats[] stats) {
        this.stats = stats;
    }

    public double precision(int label) {
        return stats[label].precision;
    }

    public double recall(int label) {
        return stats[label].recall;
    }

    public double f1measure(int label) {
        return fmeasure(label, 1);
    }

    public double fmeasure(int label, double beta) {
        Stats stats = this.stats[label];
        double precision = stats.precision;
        double recall = stats.recall;
        return (beta * beta + 1) * (precision * recall) / (beta * beta * precision + recall);
    }

    @Override
    public String toString() {
        return Arrays.toString(stats);
    }

    public static Metric average(List<Metric> metrics) {
        if (metrics == null || metrics.size() == 0) {
            throw new IllegalArgumentException("metrics cannot be null or empty to compute average metric");
        }
        int numberOfClasses = metrics.get(0).stats.length;
        Stats[] avgMetrics = new Stats[numberOfClasses];
        for (int label = 0; label < numberOfClasses; ++label) {
            double totalPrecision = 0;
            double totalRecall = 0;

            for (Metric metric : metrics) {
                Stats stat = metric.stats[label];
                totalPrecision += stat.precision;
                totalRecall += stat.recall;
            }

            double avgPrecision = totalPrecision / metrics.size();
            double avgRecall = totalRecall / metrics.size();
            avgMetrics[label] = new Stats(avgPrecision, avgRecall);
        }
        return new Metric(avgMetrics);
    }

    private static class Stats {
        private final double precision;
        private final double recall;

        Stats(double precision, double recall) {
            this.precision = precision;
            this.recall = recall;
        }

        @Override
        public String toString() {
            return "Stats{" +
                    "precision=" + precision +
                    ", recall=" + recall +
                    '}';
        }
    }

    public static class Builder {
        private final int numberOfClasses;
        private final int[][] stat;

        public Builder(int numberOfClasses) {
            this.numberOfClasses = numberOfClasses;
            this.stat = new int[numberOfClasses][numberOfClasses];
        }

        public Builder with(int expected, int actual) {
            stat[actual][expected]++;
            return this;
        }

        public Metric build() {
            return new Metric(this);
        }
    }
}
