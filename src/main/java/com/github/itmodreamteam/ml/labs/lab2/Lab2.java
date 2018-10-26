package com.github.itmodreamteam.ml.labs.lab2;

import com.github.itmodreamteam.ml.regression.GradientDescentLinearRegressionFactory;
import com.github.itmodreamteam.ml.regression.LinearRegression;
import com.github.itmodreamteam.ml.regression.LinearRegressionFactory;
import com.github.itmodreamteam.ml.regression.NormalEquationSolverLinearRegressionFactory;
import com.github.itmodreamteam.ml.utils.ClassPathResources;
import com.github.itmodreamteam.ml.utils.io.Csv;
import com.github.itmodreamteam.ml.utils.matrixes.Matrix;
import com.github.itmodreamteam.ml.utils.matrixes.Matrixes;
import com.github.itmodreamteam.ml.utils.matrixes.Vector;
import com.github.itmodreamteam.ml.utils.matrixes.Vectors;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Lab2 {
    private static final Logger LOGGER = LoggerFactory.getLogger(Lab2.class);

    public static void main(final String... args) throws Exception {
        Csv csv = Csv.read(ClassPathResources.getFile("prices.txt"), ",", false);
        Matrix features = Matrixes.dense(csv.doubles("area", "rooms"));
        Vector prices = Vectors.dense(csv.doubles("price"));
        LinearRegressionFactory factory1 = new GradientDescentLinearRegressionFactory(1000000, 0.001);
        LinearRegressionFactory factory2 = new NormalEquationSolverLinearRegressionFactory();
        LinearRegression regression2 = factory2.make(features, prices);
        LinearRegression regression1 = factory1.make(features, prices);
        LOGGER.info("{}", regression1);
        LOGGER.info("{}", regression2);
    }
}
