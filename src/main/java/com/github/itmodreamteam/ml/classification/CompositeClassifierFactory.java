package com.github.itmodreamteam.ml.classification;

import com.github.itmodreamteam.ml.validation.Samples;

import java.util.List;

public class CompositeClassifierFactory<F, A> implements ClassifierFactory<F, A> {
    private final List<Classifier<F, A>> classifiers;
    private final CompositeClassifier.Quorum<A> quorum;

    public CompositeClassifierFactory(List<Classifier<F, A>> classifiers, CompositeClassifier.Quorum<A> quorum) {
        this.classifiers = classifiers;
        this.quorum = quorum;
    }

    @Override
    public Classifier<F, A> build(Samples<F, A> train) {
        return new CompositeClassifier<>(classifiers, quorum);
    }
}
