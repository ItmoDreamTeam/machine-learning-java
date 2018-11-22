package com.github.itmodreamteam.ml.validation;

import com.github.itmodreamteam.ml.classification.Classifier;
import com.github.itmodreamteam.ml.classification.ClassifierFactory;
import com.github.itmodreamteam.ml.metric.Metric;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

public class ClassifierCrossValidator<F, A> {
    private static final Logger LOG = LoggerFactory.getLogger(ClassifierCrossValidator.class);
    private final ClassifierFactory<F, A> factory;
    private final Set<A> classes;

    public ClassifierCrossValidator(ClassifierFactory<F, A> factory, Set<A> classes) {
        this.factory = factory;
        this.classes = classes;
    }

    public Metric<A> validate(Samples<F, A> samples, int numberOfBatches) {
        int batchSize = samples.size() / numberOfBatches;
        List<Metric<A>> metrics = new ArrayList<>();
        for (int iteration = 0; iteration < numberOfBatches; ++iteration) {
            int testBatch = iteration;
            Samples<F, A> train = samples.slice(row -> row / batchSize != testBatch);
            Samples<F, A> test = samples.slice(row -> row / batchSize == testBatch);
            Classifier<F, A> classifier = factory.build(train);
            Metric.Builder<A> metricBuilder = Metric.builder(classes);

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
