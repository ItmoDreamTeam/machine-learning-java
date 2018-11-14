package com.github.itmodreamteam.ml.labs.lab1;

import com.github.itmodreamteam.ml.classification.knn.*;
import com.github.itmodreamteam.ml.metric.Metric;
import com.github.itmodreamteam.ml.utils.SliceUtils;
import com.github.itmodreamteam.ml.utils.io.Csv;
import com.github.itmodreamteam.ml.utils.matrixes.Matrix;
import com.github.itmodreamteam.ml.utils.matrixes.Matrixes;
import com.github.itmodreamteam.ml.utils.matrixes.Vector;
import com.github.itmodreamteam.ml.validation.ClassifierCrossValidator;
import com.github.itmodreamteam.ml.validation.Samples;
import com.github.sh0nk.matplotlib4j.Plot;
import com.github.sh0nk.matplotlib4j.PythonExecutionException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.Reader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static java.util.stream.Collectors.toList;

public class Lab1 {
    private final static Logger LOG = LoggerFactory.getLogger(Lab1.class);
    private final Csv csv;
    private final List<Chip> chips;

    public Lab1() throws Exception {
        Reader chipsFile = new FileReader(new File(KnnClassifier.class.getClassLoader().getResource("chips1.csv").toURI()));
        csv = Csv.read(chipsFile, ",", false);
        chips = readDataset();
    }

    public void run() throws Exception {
        Metric bestMetric = null;
        int bestNumberOfNeighbors = 0;
        int bestNumberOfBatches = 0;
        KnnClosestFunction bestMeter = null;
        KnnImportanceFunction bestImportanceFunction = null;
        DimensionTransformer bestTransformer = null;

        List<KnnClosestFunction> meters = new ArrayList<>();
        meters.add(KnnClosestFunction.euclidian());
        meters.add(KnnClosestFunction.manhattan());
        meters.add(KnnClosestFunction.cosSimilarity());

        Matrix features = Matrixes.dense(csv.doubles("X", "Y"));
        List<Integer> classes = Arrays.stream(csv.ints("Class")).boxed().collect(toList());

        List<DimensionTransformer> transformers = new ArrayList<>();
        transformers.add(DimensionTransformer.IDENTITY);
        transformers.add(DimensionTransformer.CARTESIAN_TO_POLAR);

        for (DimensionTransformer transformer : transformers) {
            LOG.info("transformer: {}", transformer);
            Matrix transformedFeatures = transformer.transform(features);
            for (int numberOfNeighbors = 2; numberOfNeighbors < 12; ++numberOfNeighbors) {
                LOG.info("number of neighbors: {}", numberOfNeighbors);
                for (int numberOfBatches = 3; numberOfBatches < 8; ++numberOfBatches) {
                    LOG.debug("number of batches: {}", numberOfBatches);
                    for (KnnClosestFunction meter : meters) {
                        LOG.debug("meter: {}", meter);
                        for (KnnImportanceFunction importanceFunction : KnnImportanceFunctions.values()) {
                            ClassifierCrossValidator<Vector, Integer> validator = new ClassifierCrossValidator<>(
                                    Knns.of(numberOfNeighbors, meter, importanceFunction, 2),
                                    2
                            );

                            Metric<Integer> metric = validator.validate(Samples.of(transformedFeatures, classes), numberOfBatches);
                            LOG.debug("neighbor: {}, batches: {}, meter: {}, kernel: {}, transformer: {}, f1mesure(0): {}",
                                    numberOfNeighbors,
                                    numberOfBatches,
                                    meter,
                                    importanceFunction,
                                    transformer,
                                    metric.f1measure(0));
                            LOG.debug("neighbor: {}, batches: {}, meter: {}, kernel: {}, transformer: {}, f1mesure(1): {}",
                                    numberOfNeighbors,
                                    numberOfBatches,
                                    meter,
                                    importanceFunction,
                                    transformer,
                                    metric.f1measure(1));
                            if (bestMetric == null || metric.f1measure(1) > bestMetric.f1measure(1)) {
                                bestMetric = metric;
                                bestNumberOfBatches = numberOfBatches;
                                bestNumberOfNeighbors = numberOfNeighbors;
                                bestMeter = meter;
                                bestImportanceFunction = importanceFunction;
                                bestTransformer = transformer;
                            }
                        }
                    }
                }
            }
        }
        LOG.info("best: neighbor: {}, batches: {}, meter: {}, kernel: {}, transformer: {}",
                bestNumberOfNeighbors,
                bestNumberOfBatches,
                bestMeter,
                bestImportanceFunction,
                bestTransformer);

        LOG.info("f1 measure(1): {}", bestMetric.f1measure(1));
        LOG.info("f1 measure(0): {}", bestMetric.f1measure(0));
        LOG.info("recall(1): {}", bestMetric.recall(1));
        LOG.info("recall(0): {}", bestMetric.recall(0));
        LOG.info("precision(1): {}", bestMetric.precision(1));
        LOG.info("precision(0): {}", bestMetric.precision(0));
        LOG.info("accuracy(1): {}", bestMetric.accuracy(1));
        LOG.info("accuracy(0): {}", bestMetric.accuracy(0));

        Knns knns = Knns.of(bestNumberOfNeighbors, bestMeter, bestImportanceFunction, 2);
        visualizeKnn(features, classes, knns, 0.8);
    }

    // TODO refactor
    //*
    private void visualizeKnn(Matrix features, List<Integer> classes, Knns knns, double trainPart) throws Exception {
        Matrix trainFeatures = features.slice(row -> 1.0 * row / features.rows() <= trainPart, true);
        List<Integer> trainClasses = SliceUtils.slice(classes, row -> 1.0 * row / features.rows() <= trainPart);
        KnnClassifier knn = knns.build(Samples.of(trainFeatures, trainClasses));
        Matrix testFeatures = features.slice(row -> 1.0 * row / features.rows() > trainPart, true);
        List<Integer> testClasses = SliceUtils.slice(classes, row -> 1.0 * row / features.rows() > trainPart);
        List<Chip> trainZeroClass = new ArrayList<>();
        List<Chip> trainOneClass = new ArrayList<>();
        List<Chip> testZeroSuccess = new ArrayList<>();
        List<Chip> testZeroFailed = new ArrayList<>();
        List<Chip> testOneSuccess = new ArrayList<>();
        List<Chip> testOneFailed = new ArrayList<>();

        for (int trainNumber = 0; trainNumber < trainClasses.size(); ++trainNumber) {
            Vector train = trainFeatures.row(trainNumber);
            Chip chip = new Chip(train.get(0), train.get(1), trainClasses.get(trainNumber));
            if (trainClasses.get(trainNumber) == 1) {
                trainOneClass.add(chip);
            } else {
                trainZeroClass.add(chip);
            }
        }

        for (int testNumber = 0; testNumber < testClasses.size(); ++testNumber) {
            Vector test = testFeatures.row(testNumber);
            int actual = knn.classify(test);
            int expected = testClasses.get(testNumber);
            Chip chip = new Chip(test.get(0), test.get(1), testClasses.get(testNumber));
            if (actual == expected) {
                if (expected == 1) {
                    testOneSuccess.add(chip);
                } else {
                    testZeroSuccess.add(chip);
                }
            } else {
                if (actual == 1) {
                    testOneFailed.add(chip);
                } else {
                    testZeroFailed.add(chip);
                }
            }
        }

        Plot plt = Plot.create();
        plt.plot()
                .add(
                        trainOneClass.stream()
                                .map(Chip::getX)
                                .collect(toList()),
                        trainOneClass.stream()
                                .map(Chip::getY)
                                .collect(toList()),
                        "bo")
                .add(
                        trainZeroClass.stream()
                                .map(Chip::getX)
                                .collect(toList()),
                        trainZeroClass.stream()
                                .map(Chip::getY)
                                .collect(toList()),
                        "ro")
                .add(
                        testOneSuccess.stream()
                                .map(Chip::getX)
                                .collect(toList()),
                        testOneSuccess.stream()
                                .map(Chip::getY)
                                .collect(toList()),
                        "b+")
                .add(
                        testZeroSuccess.stream()
                                .map(Chip::getX)
                                .collect(toList()),
                        testZeroSuccess.stream()
                                .map(Chip::getY)
                                .collect(toList()),
                        "r+")
                .add(
                        testOneFailed.stream()
                                .map(Chip::getX)
                                .collect(toList()),
                        testOneFailed.stream()
                                .map(Chip::getY)
                                .collect(toList()),
                        "bx")
                .add(
                        testZeroFailed.stream()
                                .map(Chip::getX)
                                .collect(toList()),
                        testZeroFailed.stream()
                                .map(Chip::getY)
                                .collect(toList()),
                        "rx");
        plt.xlabel("x");
        plt.ylabel("y");
        plt.title("Chips");
        plt.legend();
        plt.show();
    }
    //*/

    public static void main(final String... args) throws Exception {
        Lab1 lab = new Lab1();
        lab.visualizeDataset();
        lab.run();
    }

    private List<Chip> readDataset() {
        List<Chip> chips = new ArrayList<>();
        double[] x = csv.doubles("X");
        double[] y = csv.doubles("Y");
        int[] label = csv.ints("Class");
        for (int i = 0; i < csv.records(); ++i) {
            chips.add(new Chip(x[i], y[i], label[i]));
        }
        return chips;
    }

    private void visualizeDataset() throws IOException, PythonExecutionException {
        Plot plt = Plot.create();
        plt.plot().label("Dataset")
                .add(
                        chips.stream()
                                .filter(chip -> chip.getLabel() == 0)
                                .map(Chip::getX)
                                .collect(toList()),
                        chips.stream()
                                .filter(chip -> chip.getLabel() == 0)
                                .map(Chip::getY)
                                .collect(toList()),
                        "ro")
                .add(
                        chips.stream()
                                .filter(chip -> chip.getLabel() == 1)
                                .map(Chip::getX)
                                .collect(toList()),
                        chips.stream()
                                .filter(chip -> chip.getLabel() == 1)
                                .map(Chip::getY)
                                .collect(toList()),
                        "bo");
        plt.xlabel("x");
        plt.ylabel("y");
        plt.title("Chips");
        plt.legend();
        plt.show();
    }
}
