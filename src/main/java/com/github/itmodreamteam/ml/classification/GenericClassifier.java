package com.github.itmodreamteam.ml.classification;

import java.util.function.Function;

public class GenericClassifier<F, A, F1, A1> implements Classifier<F, A> {
    private final Classifier<F1, A1> delegate;
    private final Function<F, F1> featuresMapper;
    private final Function<A1, A> answersMapper;

    public static <F, A, F1, A1> Classifier<F, A> wrap(Classifier<F1, A1> delegate, Function<F, F1> featuresMapper, Function<A1, A> answersMapper) {
        return new GenericClassifier<>(delegate, featuresMapper, answersMapper);
    }

    public static <F, A, F1> Classifier<F, A> wrap(Classifier<F1, A> delegate, Function<F, F1> featuresMapper) {
        return new GenericClassifier<>(delegate, featuresMapper, Function.identity());
    }

    public GenericClassifier(Classifier<F1, A1> delegate, Function<F, F1> featuresMapper, Function<A1, A> answersMapper) {
        this.delegate = delegate;
        this.featuresMapper = featuresMapper;
        this.answersMapper = answersMapper;
    }

    @Override
    public A classify(F features) {
        return answersMapper.apply(delegate.classify(featuresMapper.apply(features)));
    }
}
