package com.project.parkingbackend.model;

import com.fasterxml.jackson.annotation.JsonProperty;

public class Obstacle {
    @JsonProperty("id")
    private String id;

    @JsonProperty("rect")
    private Rectangle rect;

    @JsonProperty("isDynamic")
    private boolean isDynamic;

    public Obstacle() {
    }

    public Obstacle(String id, Rectangle rect, boolean isDynamic) {
        this.id = id;
        this.rect = rect;
        this.isDynamic = isDynamic;
    }

    public String getId() {
        return id;
    }

    public void setId(String id) {
        this.id = id;
    }

    public Rectangle getRect() {
        return rect;
    }

    public void setRect(Rectangle rect) {
        this.rect = rect;
    }

    public boolean isDynamic() {
        return isDynamic;
    }

    public void setDynamic(boolean dynamic) {
        isDynamic = dynamic;
    }

    @Override
    public String toString() {
        return "Obstacle{" +
                "id='" + id + '\'' +
                ", rect=" + rect +
                ", isDynamic=" + isDynamic +
                '}';
    }
}

