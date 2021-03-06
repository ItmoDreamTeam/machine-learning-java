package com.github.itmodreamteam.ml.metric;

import java.util.*;

public class Metric<A> {
    private final Stats[] stats;
    private final Map<A, Integer> index;

    public static <A> Builder<A> builder(Set<A> answers) {
        return new Builder<>(answers);
    }

    private Metric(Builder<A> builder) {
        this.stats = new Stats[builder.numberOfClasses];
        this.index = builder.index;
        for (int classLabel = 0; classLabel < builder.numberOfClasses; ++classLabel) {
            int truePositive = builder.stat[classLabel][classLabel];
            // truePositive + falsePositive
            int totalPredicted = Arrays.stream(builder.stat[classLabel]).sum();
            // truePositive + falseNegative
            int totalLabeled = Arrays.stream(column(builder.stat, builder.numberOfClasses, classLabel)).sum();
            int totalCorrect = 0;
            int total = 0;
            for (int i = 0; i < builder.stat.length; ++i) {
                for (int j = 0; j < builder.stat[i].length; ++j) {
                    if (i == j) {
                        totalCorrect += builder.stat[i][j];
                    }
                    total += builder.stat[i][j];
                }
            }
//            System.out.println(classLabel + ": true positive: " + truePositive);
//            System.out.println(classLabel + ": total labeled: " + totalLabeled);
//            System.out.println(classLabel + ": total correct: " + totalCorrect);
            double precision = 1.0 * truePositive / totalPredicted;
            double recall = 1.0 * truePositive / totalLabeled;
            double accuracy = 1.0 * totalCorrect / total;
            Stats stats = new Stats(precision, recall, accuracy);
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

    private Metric(Stats[] stats, Map<A, Integer> index) {
        this.stats = stats;
        this.index = index;
    }

    public double precision(A label) {
        return stats[index.get(label)].precision;
    }

    public double recall(A label) {
        return stats[index.get(label)].recall;
    }

    public double f1measure(A label) {
        return fmeasure(label, 1);
    }

    public double fmeasure(A label, double beta) {
        Stats stats = this.stats[index.get(label)];
        double precision = stats.precision;
        double recall = stats.recall;
        return (beta * beta + 1) * (precision * recall) / (beta * beta * precision + recall);
    }

    public double accuracy(A label) {
        return stats[index.get(label)].accuracy;
    }

    @Override
    public String toString() {
        return Arrays.toString(stats);
    }

    public static <A> Metric<A> average(List<Metric<A>> metrics) {
        if (metrics == null || metrics.size() == 0) {
            throw new IllegalArgumentException("metrics cannot be null or empty to compute average metric");
        }
        int numberOfClasses = metrics.get(0).stats.length;
        Map<A, Integer> index = metrics.get(0).index; // TODO check
        Stats[] avgMetrics = new Stats[numberOfClasses];
        for (int label = 0; label < numberOfClasses; ++label) {
            double totalPrecision = 0;
            double totalRecall = 0;
            double totalAccuracy = 0;

            for (Metric metric : metrics) {
                Stats stat = metric.stats[label];
                totalPrecision += stat.precision;
                totalRecall += stat.recall;
                totalAccuracy += stat.accuracy;
            }

            double avgPrecision = totalPrecision / metrics.size();
            double avgRecall = totalRecall / metrics.size();
            double avgAccuracy = totalAccuracy / metrics.size();
            avgMetrics[label] = new Stats(avgPrecision, avgRecall, avgAccuracy);
        }
        return new Metric<>(avgMetrics, index);
    }

    private static class Stats {
        private final double precision;
        private final double recall;
        private final double accuracy;

        Stats(double precision, double recall, double accuracy) {
            this.precision = precision;
            this.recall = recall;
            this.accuracy = accuracy;
        }

        @Override
        public String toString() {
            return "Stats{" +
                    "precision=" + precision +
                    ", recall=" + recall +
                    ", accuracy=" + accuracy +
                    '}';
        }
    }

    public static class Builder<A> {
        private final int numberOfClasses;
        private final int[][] stat;
        private final Map<A, Integer> index = new HashMap<>();

        public Builder(Set<A> labels) {
            this.numberOfClasses = labels.size();
            int currentIndex = 0;
            for (A label : labels) {
                index.put(label, currentIndex);
                currentIndex++;
            }
            this.stat = new int[numberOfClasses][numberOfClasses];
        }

        public Builder with(A expected, A actual) {
            int expectedIndex = index.get(expected);
            int actualIndex = index.get(actual);
            stat[actualIndex][expectedIndex]++;
            return this;
        }

        public Metric<A> build() {
            return new Metric<>(this);
        }

        @Override
        public String toString() {
            return Arrays.deepToString(stat);
        }
    }
}
