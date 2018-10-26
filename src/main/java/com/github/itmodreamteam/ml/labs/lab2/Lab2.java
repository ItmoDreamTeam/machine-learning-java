package com.github.itmodreamteam.ml.labs.lab2;

import com.github.itmodreamteam.ml.regression.GradientDescentLinearRegressionFactory;
import com.github.itmodreamteam.ml.regression.LinearRegression;
import com.github.itmodreamteam.ml.utils.ClassPathResources;
import com.github.itmodreamteam.ml.utils.io.Csv;
import com.github.itmodreamteam.ml.utils.matrixes.Matrix;
import com.github.itmodreamteam.ml.utils.matrixes.Matrixes;
import com.github.itmodreamteam.ml.utils.matrixes.Vector;
import com.github.itmodreamteam.ml.utils.matrixes.Vectors;

public class Lab2 {
    public static void main(final String... args) throws Exception {
        Csv csv = Csv.read(ClassPathResources.getFile("prices.txt"), ",", false);
        Matrix features = Matrixes.dense(csv.doubles("area", "rooms"));
        Vector prices = Vectors.dense(csv.doubles("price"));
        GradientDescentLinearRegressionFactory factory = new GradientDescentLinearRegressionFactory(10000000, 0.0005);
        LinearRegression regression = factory.make(features, prices);
    }
}
