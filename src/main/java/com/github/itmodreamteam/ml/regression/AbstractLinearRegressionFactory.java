package com.github.itmodreamteam.ml.regression;

import com.github.itmodreamteam.ml.utils.matrixes.*;

import java.util.ArrayList;
import java.util.List;

public abstract class AbstractLinearRegressionFactory implements LinearRegressionFactory {
    public LinearRegression make(Matrix features, Vector expected, boolean normalize, boolean appendBias) {
        if (normalize) {
            List<Operation> normalizers = new ArrayList<>(features.cols());
            for (int featureNumber = 0; featureNumber < features.cols(); ++ featureNumber) {
                Vector feature = features.col(featureNumber);
                double min = feature.min();
                double max = feature.max();
                normalizers.add(el -> (el - min) / (max - min) - 0.5);
            }
            features = features.forEachColumnWithIndex((column, index) -> column.assign(normalizers.get(index)));
            if (appendBias) {
                features = Matrixes.joinColumns(
                        Vectors.ones(features.rows()),
                        features
                );
            }
            return LinearRegression.of(doMake(features, expected), normalizers, appendBias);
        } else {
            return LinearRegression.of(doMake(features, expected), appendBias);
        }
    }

    protected abstract Vector doMake(Matrix features, Vector expected);
}
