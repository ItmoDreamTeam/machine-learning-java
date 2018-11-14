package com.github.itmodreamteam.ml.labs.lab3;

import com.github.itmodreamteam.ml.classification.ClassifierFactory;
import com.github.itmodreamteam.ml.classification.bayes.BinaryNaiveBayesClassifierFactory;
import com.github.itmodreamteam.ml.metric.Metric;
import com.github.itmodreamteam.ml.validation.ClassifierCrossValidator;
import com.github.itmodreamteam.ml.validation.Samples;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.filefilter.RegexFileFilter;
import org.apache.commons.io.filefilter.TrueFileFilter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

import static java.util.stream.Collectors.toList;

public class Lab3 {
    private static final Logger LOG = LoggerFactory.getLogger(Lab3.class);

    public static void main(final String... args) throws Exception {
        Collection<File> messages = FileUtils.listFiles(new File("."), new RegexFileFilter(".*legit.*|.*spmsg.*"), TrueFileFilter.INSTANCE);
        List<Boolean> spamOrNot = new ArrayList<>();
        List<List<Integer>> words = new ArrayList<>();
        for (File message : messages) {
            boolean spam = isSpamMessage(message);
            List<String> lines = FileUtils.readLines(message, Charset.defaultCharset());
            String subject = lines.get(0).substring("Subject: ".length());
            String body = lines.get(2);
            List<Integer> subjectCodes = getCodes(subject);
            List<Integer> bodyCodes = getCodes(body);
            List<Integer> all = new ArrayList<>();
            all.addAll(subjectCodes);
            all.addAll(bodyCodes);
            spamOrNot.add(spam);
            words.add(all);
        }

        ClassifierFactory<List<Integer>, Boolean> factory = new BinaryNaiveBayesClassifierFactory<>(1.3);
        ClassifierCrossValidator<List<Integer>, Boolean> validator = new ClassifierCrossValidator<>(factory, 2);
        Metric metric = validator.validate(Samples.of(words, spamOrNot), 10);
        LOG.info("metric = {}", metric);
        LOG.info("f1 measure(1) = {}", metric.f1measure(1));
        LOG.info("f1 measure(0) = {}", metric.f1measure(0));
    }

    private static List<Integer> getCodes(String line) {
        return Arrays.stream(line.split(" "))
                .map(String::trim)
                .filter(str -> !str.isEmpty())
                .mapToInt(Integer::parseInt)
                .boxed()
                .collect(toList());
    }

    private static boolean isSpamMessage(File message) {
        return message.getName().contains("spmsg");
    }
}
