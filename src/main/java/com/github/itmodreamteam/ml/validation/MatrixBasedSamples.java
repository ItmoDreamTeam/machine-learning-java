package com.github.itmodreamteam.ml.validation;

import com.github.itmodreamteam.ml.utils.SliceUtils;
import com.github.itmodreamteam.ml.utils.matrixes.Matrix;
import com.github.itmodreamteam.ml.utils.matrixes.Vector;

import java.util.Collections;
import java.util.List;
import java.util.function.IntPredicate;

public class MatrixBasedSamples<A> implements Samples<Vector, A> {
    private final Matrix features;
    private final List<A> answers;
    private final int size;

    public MatrixBasedSamples(Matrix features, List<A> answers, int size) {
        this.features = features;
        this.answers = answers;
        this.size = size;
    }

    @Override
    public int size() {
        return size;
    }

    @Override
    public Vector getFeatures(int index) {
        return features.row(index);
    }

    @Override
    public A getAnswer(int index) {
        return answers.get(index);
    }

    @Override
    public Samples<Vector, A> slice(IntPredicate indexes) {
        Matrix slicedFeatures = features.slice(indexes, true);
        List<A> slicedAnswers = SliceUtils.slice(answers, indexes);
        int slicedSize = slicedFeatures.rows();
        return new MatrixBasedSamples<>(slicedFeatures, slicedAnswers, slicedSize);
    }

    @Override
    public List<A> getAnswers() {
        return Collections.unmodifiableList(answers);
    }

    @Override
    public List<Vector> getFeatures() {
        return features.getRows();
    }
}
