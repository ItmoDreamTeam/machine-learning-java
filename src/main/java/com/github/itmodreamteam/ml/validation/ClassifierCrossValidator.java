package com.github.itmodreamteam.ml.validation;

import com.github.itmodreamteam.ml.classification.Classifier;
import com.github.itmodreamteam.ml.classification.ClassifierFactory;
import com.github.itmodreamteam.ml.metric.Metric;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;

public class ClassifierCrossValidator<F, A> {
    private static final Logger LOG = LoggerFactory.getLogger(ClassifierCrossValidator.class);
    private final ClassifierFactory<F, A> factory;
    private final int numberOfClasses;

    public ClassifierCrossValidator(ClassifierFactory<F, A> factory, int numberOfClasses) {
        this.factory = factory;
        this.numberOfClasses = numberOfClasses;
    }

    public Metric<A> validate(Samples<F, A> samples, int numberOfBatches) {
        int batchSize = samples.size() / numberOfBatches;
        List<Metric<A>> metrics = new ArrayList<>();
        for (int iteration = 0; iteration < numberOfBatches; ++iteration) {
            int testBatch = iteration;
            Samples<F, A> train = samples.slice(row -> row / batchSize != testBatch);
            Samples<F, A> test = samples.slice(row -> row / batchSize == testBatch);
            Classifier<F, A> classifier = factory.build(train);
            Metric.Builder<A> metricBuilder = Metric.builder(numberOfClasses);

            for (int sampleNumber = 0; sampleNumber < test.size(); ++sampleNumber) {
                A classified = classifier.classify(test.getFeatures(sampleNumber));
                A expected = test.getAnswer(sampleNumber);
                metricBuilder.with(expected, classified);
            }
            Metric<A> metric = metricBuilder.build();
            metrics.add(metric);
        }
        return Metric.average(metrics);
    }
}
