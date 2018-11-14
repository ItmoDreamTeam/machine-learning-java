package com.github.itmodreamteam.ml.classification.bayes;

import com.github.itmodreamteam.ml.classification.Classifier;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class BinaryNaiveBayesClassifier implements Classifier<List<Integer>, Boolean> {
    private final double threshold;
    private final double trueClassProbability;
    private final double falseClassProbability;
    private final Map<Integer, Double> trueClassWordProbability;
    private final Map<Integer, Double> falseClassWordProbability;

    public BinaryNaiveBayesClassifier(List<List<Integer>> trainFeatures, List<Boolean> trainClasses, double threshold) {
        this.threshold = threshold;
        Map<Integer, Integer> numberOfContributionsToTrueClass = new HashMap<>();
        Map<Integer, Integer> numberOfContributionsToFalseClass = new HashMap<>();
        Map<Integer, Integer> numberOfContributionsToAllClasses = new HashMap<>();
        int numberOfSamples = trainFeatures.size();
        for (int sampleNumber = 0; sampleNumber < numberOfSamples; ++sampleNumber) {
            for (int feature : trainFeatures.get(sampleNumber)) {
                numberOfContributionsToTrueClass.putIfAbsent(feature, 0);
                numberOfContributionsToFalseClass.putIfAbsent(feature, 0);
                numberOfContributionsToAllClasses.putIfAbsent(feature, 0);
                numberOfContributionsToAllClasses.merge(feature, 1, (i1, i2) -> i1 + i2);
                if (trainClasses.get(sampleNumber)) {
                    numberOfContributionsToTrueClass.merge(feature, 1, (i1, i2) -> i1 + i2);
                } else {
                    numberOfContributionsToFalseClass.merge(feature, 1, (i1, i2) -> i1 + i2);
                }
            }
        }
        trueClassWordProbability = new HashMap<>();
        falseClassWordProbability = new HashMap<>();
        for (int feature : numberOfContributionsToAllClasses.keySet()) {
            double trueWeight = 1.0 * numberOfContributionsToTrueClass.get(feature) / numberOfContributionsToAllClasses.get(feature);
            double falseWeight = 1.0 * numberOfContributionsToFalseClass.get(feature) / numberOfContributionsToAllClasses.get(feature);
            trueClassWordProbability.put(feature, trueWeight);
            falseClassWordProbability.put(feature, falseWeight);
        }
        int numberOfTrueClass = (int) trainClasses.stream()
                .filter(b -> b)
                .count();
        int numberOfFalseClass = numberOfSamples - numberOfTrueClass;
        trueClassProbability = 1.0 * numberOfTrueClass;
        falseClassProbability = 1.0 * numberOfFalseClass;
    }

    @Override
    public Boolean classify(List<Integer> features) {
        double totalTrueProbability = 0.0;
        double totalFalseProbability = 0.0;
        for (int feature : features) {
            totalTrueProbability += trueClassProbability * trueClassWordProbability.get(feature);
            totalFalseProbability += falseClassProbability * falseClassWordProbability.get(feature);
        }
        return totalTrueProbability / totalFalseProbability > threshold;
    }
}
