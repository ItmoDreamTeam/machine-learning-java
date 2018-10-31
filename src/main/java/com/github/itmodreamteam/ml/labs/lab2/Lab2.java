package com.github.itmodreamteam.ml.labs.lab2;

import com.github.itmodreamteam.ml.regression.*;
import com.github.itmodreamteam.ml.utils.ClassPathResources;
import com.github.itmodreamteam.ml.utils.io.Csv;
import com.github.itmodreamteam.ml.utils.matrixes.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.Scanner;

public class Lab2 {
    private static final Scanner SCANNER = new Scanner(System.in);

    private static final Logger LOGGER = LoggerFactory.getLogger(Lab2.class);

    public static void main(final String... args) throws Exception {
        Csv csv = Csv.read(ClassPathResources.getFile("prices.txt"), ",", false);
        Matrix features = Matrixes.dense(csv.doubles("area", "rooms"));
        Operation[] normalizers = new Operation[features.cols()];
        for (int featureNumber = 0; featureNumber < features.cols(); ++ featureNumber) {
            Vector feature = features.col(featureNumber);
            double min = feature.min();
            double max = feature.max();
            normalizers[featureNumber] = el -> (el - min) / (max - min) - 0.5;
        }
        features = features.forEachColumnWithIndex((column, index) -> column.assign(normalizers[index]));
        Vector prices = Vectors.dense(csv.doubles("price"));
        LinearRegressionFactory factory1 = new GradientDescentLinearRegressionFactory(50000, 0.001);
        LinearRegressionFactory factory2 = new NormalEquationSolverLinearRegressionFactory();
        LinearRegressionFactory genetic1 = new GeneticLinearRegressionFactory(
                100,
                80,
                20,
                false,
                20
        );
        LinearRegression regression1 = factory1.make(features, prices);
        LinearRegression regression2 = factory2.make(features, prices);
        LinearRegression regression3 = genetic1.make(features, prices);

        LOGGER.info("{}", regression1);
        LOGGER.info("{}", regression2);
        LOGGER.info("{}", regression3);
        test(regression1, normalizers);
        test(regression2, normalizers);
        test(regression3, normalizers);
    }

    private static void test(LinearRegression regression, Operation[] normalizers) {
        while (true) {
            String line = SCANNER.nextLine();
            if (line.equalsIgnoreCase("quit")) {
                break;
            } else {
                double[] features = parseDoubles(line);
                for (int featureNumber = 0; featureNumber < features.length; ++featureNumber) {
                    features[featureNumber] = normalizers[featureNumber].apply(features[featureNumber]);
                }
                double[] extendedFeatures = new double[features.length + 1];
                System.arraycopy(features, 0, extendedFeatures, 1, 2);
                extendedFeatures[0] = 1;
                System.out.println(Arrays.toString(extendedFeatures));
                System.out.println(regression.answer(Vectors.dense(extendedFeatures)));
            }
        }
    }

    private static double[] parseDoubles(String line) {
        return Arrays.stream(line.split(","))
                .map(String::trim)
                .mapToDouble(Double::parseDouble)
                .toArray();
    }
}
