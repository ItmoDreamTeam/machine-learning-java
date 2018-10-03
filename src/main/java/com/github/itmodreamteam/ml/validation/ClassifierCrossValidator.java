package com.github.itmodreamteam.ml.validation;

import com.github.itmodreamteam.ml.utils.collections.IntList;
import com.github.itmodreamteam.ml.utils.matrixes.Matrix;
import com.github.itmodreamteam.ml.classification.Classifier;
import com.github.itmodreamteam.ml.classification.ClassifierFactory;
import com.github.itmodreamteam.ml.metric.Metric;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;

public class ClassifierCrossValidator {
    private final Logger LOG = LoggerFactory.getLogger(ClassifierCrossValidator.class);
    private final ClassifierFactory classifierFactory;
    private final Matrix features;
    private final IntList labels;

    public ClassifierCrossValidator(ClassifierFactory classifierFactory, Matrix features, IntList labels) {
        this.classifierFactory = classifierFactory;
        this.features = features;
        this.labels = labels;
    }

    public static ClassifierCrossValidator of(ClassifierFactory classifierFactory, Matrix features, IntList labels) {
        return new ClassifierCrossValidator(classifierFactory, features, labels);
    }

    public Metric validate(int numberOfBatches) {
        int size = features.rows();
        int numberOfClasses = (int) IntStream.of(labels.toArray()).distinct().count();
        int batchSize = size / numberOfBatches;
        List<Metric> metrics = new ArrayList<>();
        for (int iteration = 0; iteration < numberOfBatches; ++iteration) {
            int testBatch = iteration;
            Matrix trains = features.slice(row -> row / batchSize != testBatch,true);
            Matrix tests = features.slice(row -> row / batchSize == testBatch, true);
            IntList trainLabels = labels.slice(row -> row / batchSize != testBatch);
            IntList testLabels = labels.slice(row -> row / batchSize == testBatch);
            Classifier classifier = classifierFactory.build(trains, trainLabels);
            Metric.Builder metricBuilder = Metric.builder(numberOfClasses);

            for (int rowNumber = 0; rowNumber < tests.rows(); ++rowNumber) {
                int classified = classifier.classify(tests.row(rowNumber));
                int expected = testLabels.get(rowNumber);
                metricBuilder.with(expected, classified);
            }
            Metric metric = metricBuilder.build();
            LOG.debug("iteration #{}, metric: {}", iteration, metric);
            metrics.add(metric);
        }
        return Metric.average(metrics);
    }
}
