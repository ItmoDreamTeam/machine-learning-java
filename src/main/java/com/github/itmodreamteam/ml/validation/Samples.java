package com.github.itmodreamteam.ml.validation;

import com.github.itmodreamteam.ml.utils.matrixes.Matrix;
import com.github.itmodreamteam.ml.utils.matrixes.Vector;

import java.util.List;
import java.util.function.IntPredicate;

public interface Samples<F, A> {
    int size();

    F getFeatures(int index);

    A getAnswer(int index);

    Samples<F, A> slice(IntPredicate indexes);

    List<A> getAnswers();

    List<F> getFeatures();

    static <F, A> Samples<F, A> of(List<F> features, List<A> answers) {
        return new ListBasedSamples<>(features, answers, features.size());
    }

    static <A> Samples<Vector, A> of(Matrix features, List<A> answers) {
        return new MatrixBasedSamples<>(features, answers, features.rows());
    }
}
