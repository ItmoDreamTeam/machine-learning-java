package com.github.itmodreamteam.ml.classification.bayes;

import com.github.itmodreamteam.ml.classification.Classifier;
import com.github.itmodreamteam.ml.classification.ClassifierFactory;
import com.github.itmodreamteam.ml.validation.Samples;

import java.util.List;

public class BinaryNaiveBayesClassifierFactory implements ClassifierFactory<List<Integer>, Boolean> {
    private final double threshold;

    public BinaryNaiveBayesClassifierFactory(double threshold) {
        this.threshold = threshold;
    }

    @Override
    public Classifier<List<Integer>, Boolean> build(Samples<List<Integer>, Boolean> samples) {
        return new BinaryNaiveBayesClassifier(samples.getFeatures(), samples.getAnswers(), threshold);
    }
}
