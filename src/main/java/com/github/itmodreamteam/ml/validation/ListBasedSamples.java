package com.github.itmodreamteam.ml.validation;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.function.IntPredicate;

public class ListBasedSamples<F, A> implements Samples<F, A> {
    private final List<F> features;
    private final List<A> answers;
    private final int size;

    public ListBasedSamples(List<F> features, List<A> answers, int size) {
        this.features = features;
        this.answers = answers;
        this.size = size;
    }

    @Override
    public int size() {
        return size;
    }

    @Override
    public F getFeatures(int index) {
        return features.get(index);
    }

    @Override
    public A getAnswer(int index) {
        return answers.get(index);
    }

    @Override
    public Samples<F, A> slice(IntPredicate indexes) {
        List<F> features = new ArrayList<>();
        List<A> answers = new ArrayList<>();
        int size = 0;
        for (int i = 0; i < this.size; ++i) {
            if (indexes.test(i)) {
                features.add(this.features.get(i));
                answers.add(this.answers.get(i));
                size++;
            }
        }
        return new ListBasedSamples<>(features, answers, size);
    }

    @Override
    public List<A> getAnswers() {
        return Collections.unmodifiableList(answers);
    }

    @Override
    public List<F> getFeatures() {
        return Collections.unmodifiableList(features);
    }
}
