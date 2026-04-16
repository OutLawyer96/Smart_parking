package com.project.parkingbackend.service;

import com.project.parkingbackend.model.PinSession;
import com.project.parkingbackend.model.Vehicle;
import com.project.parkingbackend.model.dto.AssignedSlotDto;
import com.project.parkingbackend.model.dto.RouteWaypointDto;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;

import java.security.SecureRandom;
import java.time.Instant;
import java.time.temporal.ChronoUnit;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;

@Service
public class PinSessionStore {

    private static final Logger logger = LoggerFactory.getLogger(PinSessionStore.class);
    private static final int PIN_TTL_MINUTES = 30;

    private final ConcurrentHashMap<String, PinSession> store = new ConcurrentHashMap<>();
    private final SecureRandom random = new SecureRandom();

    /**
     * Generate a unique 4-digit PIN and store a session for the given vehicle/slot/route.
     * Any previous PIN for the same tracking_id is removed first.
     */
    public String createPin(int trackingId, String plate, Vehicle vehicle,
                             AssignedSlotDto assignedSlot, List<RouteWaypointDto> route) {
        // Remove stale entry for the same tracking_id (re-entry case)
        store.values().removeIf(s -> s.getTrackingId() == trackingId);

        String pin;
        int attempts = 0;
        do {
            pin = String.format("%04d", random.nextInt(10_000));
            if (++attempts > 1000) {
                throw new IllegalStateException("Could not generate a unique PIN — store may be full");
            }
        } while (store.containsKey(pin));

        Instant now = Instant.now();
        store.put(pin, new PinSession(pin, trackingId, plate, vehicle, assignedSlot, route,
                now, now.plus(PIN_TTL_MINUTES, ChronoUnit.MINUTES)));

        logger.info("PIN {} created for plate {} (tracking_id={}, valid for {} min)",
                pin, plate, trackingId, PIN_TTL_MINUTES);
        return pin;
    }

    /**
     * Look up a PIN.  Returns empty if PIN is unknown or has expired.
     */
    public Optional<PinSession> findByPin(String pin) {
        if (pin == null || pin.isBlank()) return Optional.empty();
        PinSession session = store.get(pin.trim());
        if (session == null) return Optional.empty();
        if (Instant.now().isAfter(session.getExpiresAt())) {
            store.remove(pin);
            logger.debug("PIN {} was expired on lookup — removed", pin);
            return Optional.empty();
        }
        return Optional.of(session);
    }

    public int getPinTtlMinutes() {
        return PIN_TTL_MINUTES;
    }

    /** Background cleanup every 10 minutes. */
    @Scheduled(fixedRate = 10, timeUnit = TimeUnit.MINUTES)
    public void evictExpired() {
        Instant now = Instant.now();
        int before = store.size();
        store.values().removeIf(s -> now.isAfter(s.getExpiresAt()));
        int removed = before - store.size();
        if (removed > 0) logger.info("Evicted {} expired PIN session(s)", removed);
    }
}

