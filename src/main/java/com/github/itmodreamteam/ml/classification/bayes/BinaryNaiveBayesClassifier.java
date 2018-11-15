package com.github.itmodreamteam.ml.classification.bayes;

import com.github.itmodreamteam.ml.classification.Classifier;

import java.util.*;

public class BinaryNaiveBayesClassifier<T> implements Classifier<List<T>, Boolean> {
    private final double alpha;
    private final double threshold;
    private final double trueClassProbability;
    private final double falseClassProbability;
    private final int numberOfTrueCases;
    private final int numberOfFalseCases;
    private final Map<T, Integer> numberOfContributionsToTrueClass;
    private final Map<T, Integer> numberOfContributionsToFalseClass;

    public BinaryNaiveBayesClassifier(List<List<T>> trainFeatures, List<Boolean> trainClasses, double alpha, double threshold) {
        this.alpha = alpha;
        this.threshold = threshold;
        numberOfContributionsToTrueClass = new HashMap<>();
        numberOfContributionsToFalseClass = new HashMap<>();
        int numberOfTrueCases = 0;
        int numberOfFalseCases = 0;
        int numberOfSamples = trainFeatures.size();
        for (int sampleNumber = 0; sampleNumber < numberOfSamples; ++sampleNumber) {
            boolean label = trainClasses.get(sampleNumber);
            if (label) {
                numberOfTrueCases++;
            } else {
                numberOfFalseCases++;
            }
            for (T feature : trainFeatures.get(sampleNumber)) {
                numberOfContributionsToTrueClass.putIfAbsent(feature, 0);
                numberOfContributionsToFalseClass.putIfAbsent(feature, 0);
                if (label) {
                    numberOfContributionsToTrueClass.merge(feature, 1, (i1, i2) -> i1 + i2);
                } else {
                    numberOfContributionsToFalseClass.merge(feature, 1, (i1, i2) -> i1 + i2);
                }
            }
        }
        this.numberOfFalseCases = numberOfFalseCases;
        this.numberOfTrueCases = numberOfTrueCases;
        trueClassProbability = 1.0 * numberOfTrueCases / numberOfSamples;
        falseClassProbability = 1.0 * numberOfFalseCases / numberOfSamples;
    }

    @Override
    public Boolean classify(List<T> features) {
        double likelihood = Math.log(trueClassProbability / falseClassProbability);
        for (T feature : features) {
            likelihood += Math.log(probability(true, feature) / probability(false, feature));
        }
        return likelihood > threshold;
    }

    private double probability(boolean label, T feature) {
        Map<T, Integer> source = label ? numberOfContributionsToTrueClass : numberOfContributionsToFalseClass;
        int numberOfCases = label ? numberOfTrueCases : numberOfFalseCases;
        double contributions = 0.0;
        if (source.containsKey(feature)) {
            contributions = source.get(feature);
        }
        return (contributions + alpha) / (numberOfCases + alpha);
    }
}
