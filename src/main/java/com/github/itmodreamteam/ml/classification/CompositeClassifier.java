package com.github.itmodreamteam.ml.classification;

import java.util.List;

import static java.util.stream.Collectors.toList;

public class CompositeClassifier<F, A> implements Classifier<F, A> {
    private final List<Classifier<F, A>> classifiers;
    private final Quorum<A> quorum;

    public CompositeClassifier(List<Classifier<F, A>> classifiers, Quorum<A> quorum) {
        this.classifiers = classifiers;
        this.quorum = quorum;
    }

    @Override
    public A classify(F features) {
        List<A> answers = classifiers.stream()
                .map(classifier -> classifier.classify(features))
                .collect(toList());
        return quorum.decide(answers);
    }

    public interface Quorum<A> {
        A decide(List<A> answers);
    }
}
