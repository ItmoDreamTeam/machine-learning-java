package com.github.itmodreamteam.ml.labs.lab3;

import java.util.ArrayList;
import java.util.List;

public class Message {
    private final List<Integer> subject;
    private final List<Integer> body;
    private final List<Integer> all;

    public Message(List<Integer> subject, List<Integer> body) {
        this.subject = subject;
        this.body = body;
        all = new ArrayList<>();
        all.addAll(subject);
        all.addAll(body);
    }

    public List<Integer> getSubject() {
        return subject;
    }

    public List<Integer> getBody() {
        return body;
    }

    public List<Integer> getAllWords() {
        return all;
    }
}
