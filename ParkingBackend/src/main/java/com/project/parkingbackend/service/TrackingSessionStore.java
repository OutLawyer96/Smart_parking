package com.project.parkingbackend.service;

import org.springframework.stereotype.Service;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

@Service
public class TrackingSessionStore {

    private final Map<String, Integer> plateToTrackingId = new ConcurrentHashMap<>();
    private final Map<Integer, String> trackingIdToPlate = new ConcurrentHashMap<>();
    private final Map<Integer, String> trackingIdToAssignedSlot = new ConcurrentHashMap<>();
    private final AtomicInteger nextTrackingId = new AtomicInteger(1);

    public int getOrCreateTrackingId(String plate) {
        return plateToTrackingId.computeIfAbsent(plate, p -> {
            int newId = nextTrackingId.getAndIncrement();
            trackingIdToPlate.put(newId, p);
            return newId;
        });
    }

    public void bindTrackingId(String plate, int trackingId) {
        plateToTrackingId.put(plate, trackingId);
        trackingIdToPlate.put(trackingId, plate);
    }

    public void setAssignedSlot(int trackingId, String slotId) {
        if (slotId != null && !slotId.isBlank()) {
            trackingIdToAssignedSlot.put(trackingId, slotId);
        }
    }

    public String getAssignedSlot(int trackingId) {
        return trackingIdToAssignedSlot.get(trackingId);
    }
}

