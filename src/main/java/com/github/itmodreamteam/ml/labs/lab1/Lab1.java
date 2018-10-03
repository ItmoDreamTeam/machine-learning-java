package com.github.itmodreamteam.ml.labs.lab1;

import com.github.itmodreamteam.ml.labs.lab1.knn.*;
import com.github.itmodreamteam.ml.metric.Metric;
import com.github.itmodreamteam.ml.utils.collections.IntList;
import com.github.itmodreamteam.ml.utils.collections.Lists;
import com.github.itmodreamteam.ml.utils.io.Csv;
import com.github.itmodreamteam.ml.utils.matrixes.Matrix;
import com.github.itmodreamteam.ml.utils.matrixes.Matrixes;
import com.github.itmodreamteam.ml.validation.ClassifierCrossValidator;
import com.github.sh0nk.matplotlib4j.Plot;
import com.github.sh0nk.matplotlib4j.PythonExecutionException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.Reader;
import java.util.ArrayList;
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

    public void run() {
        double bestF1Measure = 0.0;
        int bestNumberOfNeighbor = 0;
        int bestNumberOfBatches = 0;
        KnnDistMeter bestMeter = null;
        KnnImportanceFunction bestImportanceFunction = null;

        List<KnnDistMeter> meters = new ArrayList<>();
        meters.add(KnnDistMeter.euclidian());
        meters.add(KnnDistMeter.manhattan());
        meters.add(KnnDistMeter.mahalanobis());
        meters.add(KnnDistMeter.cosSimilarity());
        Matrix features = Matrixes.dense(csv.doubles("X", "Y"));
        IntList classes = Lists.of(csv.ints("Class"));

        for (int numberOfNeighbors = 2; numberOfNeighbors < 10; ++numberOfNeighbors) {
            for (int numberOfBatches = 3; numberOfBatches < 6; ++numberOfBatches) {
                for (KnnDistMeter meter : meters) {
                    for (KnnImportanceFunction importanceFunction : KnnImportanceFunctions.values()) {
                        ClassifierCrossValidator validator = ClassifierCrossValidator.of(
                                Knns.of(numberOfNeighbors, meter, importanceFunction, 2), features, classes
                        );

                        Metric metric = validator.validate(numberOfBatches);
                        LOG.debug("neighbor: {}, batches: {}, meter: {}, kernel: {}, f1mesure(0): {}", numberOfNeighbors, numberOfBatches, meter, importanceFunction, metric.f1measure(0));
                        LOG.debug("neighbor: {}, batches: {}, meter: {}, kernel: {}, f1mesure(1): {}", numberOfNeighbors, numberOfBatches, meter, importanceFunction, metric.f1measure(1));
                        if (metric.f1measure(1) > bestF1Measure) {
                            bestF1Measure = metric.f1measure(1);
                            bestNumberOfBatches = numberOfBatches;
                            bestNumberOfNeighbor = numberOfNeighbors;
                            bestMeter = meter;
                            bestImportanceFunction = importanceFunction;
                        }
                    }
                }
            }
        }
        LOG.info("best f1 measure: {}, neighbor: {}, batches: {}, meter: {}, kernel: {}", bestF1Measure, bestNumberOfNeighbor, bestNumberOfBatches, bestMeter, bestImportanceFunction);
    }

    public static void main(final String... args) throws Exception {
        Lab1 lab = new Lab1();
        lab.visualizeDataset();
        lab.run();
    }

    private List<Chip> readDataset () {
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
